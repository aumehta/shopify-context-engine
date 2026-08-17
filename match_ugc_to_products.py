"""
match_ugc_to_products.py

Matches scraped UGC (TikTok or Reddit transcripts) to products in a brand's catalog.
Supports posts that discuss multiple products (returns one row per product mentioned).

Pipeline:
    1. Load + clean product catalog (Shopify /products.json format)
    2. Batch-embed all products and all UGC transcripts (one API call each)
    3. Retrieve top-k candidate products per transcript via cosine similarity
    4. If there's one dominant match (high similarity + clear margin over 2nd place),
       auto-accept it and just ask the LLM for a summary + sentiment.
       Otherwise (ambiguous, or multiple close candidates suggesting a multi-product
       post), ask the LLM which of the candidates are genuinely discussed, and score each.
    5. Filter out anything below the confidence threshold
    6. Write one row per UGC-to-SKU match to output_path, overwriting it fresh each run
       (no result caching here -- only the scrape step in scrape_tiktoks.py /
       scrape_reddit.py is cached)

CLI usage:
    python3 match_ugc_to_products.py --brand "Rare Beauty" --source tiktok
    python3 match_ugc_to_products.py --brand "Rare Beauty" --source reddit
    python3 match_ugc_to_products.py --brand "Rare Beauty" --source tiktok --days 7 --top-k 5

By default, paths are derived from --brand, --source, and --days using the same
slug convention as scrape_catalog.py / scrape_tiktoks.py / scrape_reddit.py, so as
long as you scrape with the same brand name and window, the matcher finds the right
files automatically:
    catalog/{slug}_products.json
    cache/{slug}_tiktoks_{days}d.json        (source=tiktok)
    cache/{slug}_reddit_{days}d.json         (source=reddit)
    output/{slug}_{source}_{days}d_ugc_matches.json
Override any of these individually with --catalog-path / --ugc-path / --output-path.

Can also be imported and called directly, e.g. from a future UI/backend:
    from match_ugc_to_products import run
    run(brand="Rare Beauty", source="reddit", days=7, top_k=5)

Env vars:
    OPENAI_API_KEY

Install:
    pip install openai numpy beautifulsoup4 --break-system-packages
"""

import argparse
import os
import re
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Maps --source to the human-readable label stored in ugc_source on every
# output record. Add an entry here when a new platform gets scraped.
SOURCE_LABELS = {"tiktok": "TikTok", "reddit": "Reddit"}

# Each platform's scraper names the "when was this posted" field differently
# (Reddit: date_posted, TikTok: create_time). Add an entry here per new source.
POST_DATE_FIELDS = {"tiktok": "create_time", "reddit": "date_posted"}


def brand_to_slug(brand: str) -> str:
    """Turns a brand name into the filename-safe slug used by every script
    in the pipeline (catalog, cache, and output filenames all match)."""
    return re.sub(r"[^a-z0-9]+", "_", brand.lower()).strip("_")


# ======================================================
# DATA MODEL
# ======================================================

@dataclass
class ProductRecord:
    sku: str
    title: str
    text: str  # what gets embedded


@dataclass
class MatchRecord:
    brand: str
    sku: str
    product_title: str
    confidence_score: int
    ugc_source: str
    ugc_link: str
    ugc_summary: str
    sentiment: str
    ugc_transcript: str
    post_date: str  # when the UGC was originally posted (platform-reported)
    matched_at: str  # when this matching run produced the record


# ======================================================
# STEP 1: LOAD + CLEAN PRODUCT CATALOG
# ======================================================

def strip_html(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def load_products(path: str) -> list[ProductRecord]:
    with open(path, "r") as f:
        raw = json.load(f)

    products = raw["products"] if isinstance(raw, dict) else raw
    records = []

    for p in products:
        title = p.get("title", "")
        description = strip_html(p.get("body_html", ""))
        tags = p.get("tags", "")
        product_type = p.get("product_type", "")

        sku = None
        variants = p.get("variants", [])
        if variants and variants[0].get("sku"):
            sku = variants[0]["sku"]
        if not sku:
            sku = str(p.get("id", title))

        text = f"{title}. {product_type}. {description} Tags: {tags}"
        records.append(ProductRecord(sku=sku, title=title, text=text.strip()))

    return records


# ======================================================
# STEP 2: EMBEDDINGS (batched)
# ======================================================

def embed_texts(texts: list[str], batch_size: int = 96) -> np.ndarray:
    """Embed a list of strings, batched, returns (n, dim) array."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


# ======================================================
# STEP 3/4: LLM MATCHING + SUMMARIZATION
# ======================================================

MATCH_PROMPT = """You are matching a piece of user-generated content (UGC) about a brand \
to the specific product(s) it is discussing, if any. A single piece of UGC can discuss multiple products.

Brand: {brand}

UGC transcript:
\"\"\"{transcript}\"\"\"

Candidate products (retrieved by semantic similarity, ranked best-first):
{candidates}

For EACH distinct SKU genuinely discussed in the transcript (not just mentioned in passing, and \
not the brand in general), return ONE entry with:
- sku
- confidence_score (0-100): how clearly and specifically the transcript discusses this exact product
- summary: 1-2 sentences on what's said about THIS product specifically
- sentiment: "Positive", "Neutral", or "Negative" toward THIS product

IMPORTANT: Each SKU must appear AT MOST ONCE in your output, even if the transcript refers to \
it in multiple ways or mentions multiple components/features of the same product (e.g. a "blush \
and luminizer duo" product is still ONE SKU, even if the speaker talks about the blush and the \
luminizer as separate things). If a transcript discusses several aspects of the same SKU, combine \
them into a single entry with one summary covering all of it.

If no candidate is genuinely discussed, return an empty list.

Respond with ONLY valid JSON, no markdown fences, in this exact shape:
{{"matches": [{{"sku": "...", "confidence_score": <int 0-100>, "summary": "...", "sentiment": "Positive"|"Neutral"|"Negative"}}]}}
"""

SUMMARY_ONLY_PROMPT = """Summarize this UGC about {product_title} in 1-2 sentences, and \
classify overall sentiment toward the product.

Transcript:
\"\"\"{transcript}\"\"\"

Respond with ONLY valid JSON, no markdown fences, in this exact shape:
{{"summary": "<string>", "sentiment": "Positive" | "Neutral" | "Negative"}}
"""


def format_candidates(candidates: list[ProductRecord]) -> str:
    return "\n".join(
        f"- SKU: {c.sku} | Title: {c.title} | Description: {c.text[:300]}"
        for c in candidates
    )


def llm_match_multi(brand: str, transcript: str, candidates: list[ProductRecord]) -> list[dict]:
    """Ambiguous / possibly-multi-product path: LLM decides which candidates
    are genuinely discussed, and scores/summarizes each one it keeps."""
    prompt = MATCH_PROMPT.format(
        brand=brand,
        transcript=transcript[:4000],
        candidates=format_candidates(candidates),
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        matches = json.loads(resp.choices[0].message.content).get("matches", [])
    except (json.JSONDecodeError, IndexError):
        return []

    return dedupe_by_sku(matches)


def dedupe_by_sku(matches: list[dict]) -> list[dict]:
    """Safeguard against the LLM returning the same SKU twice for one post
    (e.g. treating two components of a bundle/duo product as separate mentions).
    Keeps the highest-confidence entry per SKU."""
    best_by_sku: dict[str, dict] = {}
    for m in matches:
        sku = m.get("sku")
        if not sku:
            continue
        existing = best_by_sku.get(sku)
        if existing is None or m.get("confidence_score", 0) > existing.get("confidence_score", 0):
            best_by_sku[sku] = m
    return list(best_by_sku.values())


def llm_summarize_only(transcript: str, product: ProductRecord) -> Optional[dict]:
    """Cheap path for the auto-accept case: match is already decided by vector
    search's clear margin, just need a summary + sentiment."""
    prompt = SUMMARY_ONLY_PROMPT.format(
        product_title=product.title,
        transcript=transcript[:4000],
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        return None


# ======================================================
# HELPERS
# ======================================================

def load_ugc(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def extract_transcript(item: dict) -> str:
    """Adjust field names to match whatever Bright Data actually returns."""
    for key in ("transcript", "video_transcript", "subtitles", "description", "caption"):
        if item.get(key):
            return item[key]
    return ""


def extract_link(item: dict) -> str:
    for key in ("url", "video_url", "post_url", "link"):
        if item.get(key):
            return item[key]
    return ""


def extract_post_date(item: dict, source: str) -> str:
    """Original post/publish timestamp, read from whichever field this
    platform's scraper uses (see POST_DATE_FIELDS). Falls back to checking
    every known field, in case a record came from a different source than
    the --source flag claims."""
    primary_key = POST_DATE_FIELDS.get(source)
    if primary_key and item.get(primary_key):
        return item[primary_key]
    for key in POST_DATE_FIELDS.values():
        if item.get(key):
            return item[key]
    return ""


# ======================================================
# MAIN PIPELINE
# ======================================================

def run(
    brand: str,
    source: str = "tiktok",
    days: int = 7,
    catalog_path: Optional[str] = None,
    ugc_path: Optional[str] = None,
    output_path: Optional[str] = None,
    top_k: int = 5,
    confidence_threshold: int = 50,
    auto_accept_sim: float = 0.85,
    auto_accept_margin: float = 0.10,
):
    """
    Match UGC to products for a given brand.

    source: which platform this UGC came from ("tiktok" or "reddit"). Drives
        the default ugc_path, the post-date field to read, and the
        ugc_source label stored on every output record.
    days: recency window the UGC was scraped with -- only used to build the
        default cache/output filenames (cache/{slug}_{source}_{days}d.json),
        so it matches whatever --days you scraped with.
    """
    slug = brand_to_slug(brand)
    source = source.lower()

    if source not in SOURCE_LABELS:
        raise ValueError(f"Unknown source '{source}', expected one of {list(SOURCE_LABELS)}")
    ugc_source_label = SOURCE_LABELS[source]

    catalog_path = catalog_path or f"catalog/{slug}_products.json"
    ugc_path = ugc_path or f"cache/{slug}_{source}_{days}d.json"
    output_path = output_path or f"output/{slug}_{source}_{days}d_ugc_matches.json"

    print(f"Brand: {brand}")
    print(f"Source: {ugc_source_label}")
    print(f"Catalog: {catalog_path}")
    print(f"UGC: {ugc_path}")
    print(f"Output: {output_path}\n")

    print("Loading product catalog...")
    products = load_products(catalog_path)
    print(f"Loaded {len(products)} products")

    print("Loading UGC...")
    ugc_items = load_ugc(ugc_path)
    print(f"Loaded {len(ugc_items)} UGC items")

    pending = [
        (item, extract_transcript(item))
        for item in ugc_items
        if extract_transcript(item).strip()
    ]

    if not pending:
        print("No UGC items with transcripts to process.")
        return []

    print("Embedding products...")
    product_vecs = embed_texts([p.text for p in products])

    print(f"Embedding {len(pending)} new transcripts (batched)...")
    transcript_vecs = embed_texts([t for _, t in pending])

    sims = cosine_sim_matrix(transcript_vecs, product_vecs)  # (n_pending, n_products)

    # One matched_at timestamp for the whole run, so every record produced
    # by this invocation is stamped identically.
    matched_at = datetime.now(timezone.utc).isoformat()

    matches: list[MatchRecord] = []

    for i, (item, transcript) in enumerate(pending):
        row = sims[i]
        top_idx = np.argsort(row)[::-1][:top_k]
        candidates = [products[j] for j in top_idx]
        best_sim, second_sim = row[top_idx[0]], row[top_idx[1]]

        if best_sim >= auto_accept_sim and (best_sim - second_sim) >= auto_accept_margin:
            print(f"[{i}] one dominant match (sim={best_sim:.2f}), skipping LLM match decision")
            result = llm_summarize_only(transcript, candidates[0])
            found = [{
                "sku": candidates[0].sku,
                "confidence_score": int(best_sim * 100),
                "summary": result.get("summary", "") if result else "",
                "sentiment": result.get("sentiment", "Neutral") if result else "Neutral",
            }] if result else []
        else:
            print(f"[{i}] ambiguous or multi-product (sim={best_sim:.2f}, "
                  f"margin={best_sim - second_sim:.2f}), asking LLM...")
            found = llm_match_multi(brand, transcript, candidates)

        post_date = extract_post_date(item, source)

        for m in found:
            confidence = int(m.get("confidence_score", 0))
            if confidence < confidence_threshold:
                print(f"[{i}] dropping {m.get('sku')} (confidence {confidence}% < threshold)")
                continue

            matched = next((p for p in products if p.sku == m.get("sku")), None)
            if not matched:
                print(f"[{i}] unknown SKU {m.get('sku')}, dropping")
                continue

            matches.append(MatchRecord(
                brand=brand,
                sku=matched.sku,
                product_title=matched.title,
                confidence_score=confidence,
                ugc_source=ugc_source_label,
                ugc_link=extract_link(item),
                ugc_summary=m.get("summary", ""),
                sentiment=m.get("sentiment", "Neutral"),
                ugc_transcript=transcript,
                post_date=post_date,
                matched_at=matched_at,
            ))

        time.sleep(0.1)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(m) for m in matches], f, indent=2)

    print(f"\nSaved {len(matches)} total matched UGC records to {output_path}")
    return matches


def main():
    parser = argparse.ArgumentParser(description="Match UGC to a brand's product catalog.")
    parser.add_argument("--brand", required=True, help='Brand name, e.g. "Rare Beauty"')
    parser.add_argument("--source", default="tiktok", choices=sorted(SOURCE_LABELS.keys()),
                         help="Which platform this UGC came from (default: tiktok)")
    parser.add_argument("--days", type=int, default=7,
                         help="Recency window used to locate the cached UGC file (default: 7)")
    parser.add_argument("--catalog-path", default=None, help="Override catalog JSON path")
    parser.add_argument("--ugc-path", default=None, help="Override UGC JSON path")
    parser.add_argument("--output-path", default=None, help="Override output JSON path")
    parser.add_argument("--top-k", type=int, default=5, help="Candidate products per UGC item (default: 5)")
    parser.add_argument("--confidence-threshold", type=int, default=50,
                         help="Minimum confidence to keep a match (default: 50)")
    parser.add_argument("--auto-accept-sim", type=float, default=0.85,
                         help="Similarity bar for skipping the LLM match decision (default: 0.85)")
    parser.add_argument("--auto-accept-margin", type=float, default=0.10,
                         help="Required margin over 2nd-best candidate for auto-accept (default: 0.10)")
    args = parser.parse_args()

    run(
        brand=args.brand,
        source=args.source,
        days=args.days,
        catalog_path=args.catalog_path,
        ugc_path=args.ugc_path,
        output_path=args.output_path,
        top_k=args.top_k,
        confidence_threshold=args.confidence_threshold,
        auto_accept_sim=args.auto_accept_sim,
        auto_accept_margin=args.auto_accept_margin,
    )


if __name__ == "__main__":
    main()