"""
analyze_sentiment_trends.py

Tracks how sentiment toward each product shifts over time, using the UGC-to-SKU
matches already produced by match_ugc_to_products.py (which carries post_date,
sentiment, sku, and ugc_source on every record).

For each product, sentiment is bucketed by month and kept separate per source
(TikTok vs Reddit), since the two audiences can behave differently and merging
them would hide that. Within each (product, source, month) bucket we count how
many matches were Positive / Neutral / Negative.

CLI usage:
    python3 analyze_sentiment_trends.py --brand "Rare Beauty"
    python3 analyze_sentiment_trends.py --brand "Rare Beauty" --sources tiktok
    python3 analyze_sentiment_trends.py --brand "Rare Beauty" --input-paths output/rare_beauty_tiktok_ugc_matches.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

SENTIMENT_KEYS = ("Positive", "Neutral", "Negative")


def brand_to_slug(brand: str) -> str:
    """Same slug convention used across the pipeline (scrape/catalog/match
    scripts), so default paths line up automatically."""
    return re.sub(r"[^a-z0-9]+", "_", brand.lower()).strip("_")


def match_key(record: dict) -> tuple:
    """Same dedupe key as match_ugc_to_products.py's merge step, so combining
    several match files here (e.g. a 7d run and a 365d run) doesn't double-
    count a match that appears in both."""
    return (record.get("ugc_link") or record.get("ugc_transcript", ""), record.get("sku"))


def month_bucket(date_str: str) -> Optional[str]:
    """Converts an ISO timestamp to a "YYYY-MM" bucket. Returns None for
    missing/unparseable dates so callers can decide how to handle them."""
    if not date_str:
        return None
    try:
        parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.strftime("%Y-%m")


def find_input_files(slug: str, sources: list[str]) -> list[str]:
    """Globs output/{slug}_{source}_*_ugc_matches.json per source, since
    match_ugc_to_products.py's default output filenames are days-scoped."""
    paths = []
    for source in sources:
        pattern = f"output/{slug}_{source}_*_ugc_matches.json"
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"  No files matched {pattern}")
        paths.extend(found)
    return paths


def load_records(paths: list[str]) -> list[dict]:
    """Loads and dedupes records across every input file."""
    by_key: dict[tuple, dict] = {}
    for path in paths:
        with open(path, "r") as f:
            records = json.load(f)
        print(f"  Loaded {len(records)} records from {path}")
        for r in records:
            by_key[match_key(r)] = r  # later files win on overlap, same as match_ugc_to_products.py
    return list(by_key.values())


def build_trends(records: list[dict]) -> dict:
    """Groups records into {sku: {product_title, sources: {source: {month: counts}}}}."""
    products: dict[str, dict] = {}
    skipped_no_date = 0

    for r in records:
        sku = r.get("sku")
        source = r.get("ugc_source")
        sentiment = r.get("sentiment")
        month = month_bucket(r.get("post_date", ""))

        if not sku or not source or sentiment not in SENTIMENT_KEYS:
            continue
        if month is None:
            skipped_no_date += 1
            continue

        product = products.setdefault(sku, {
            "product_title": r.get("product_title", ""),
            "sources": defaultdict(lambda: defaultdict(lambda: {k.lower(): 0 for k in SENTIMENT_KEYS} | {"total": 0})),
        })
        bucket = product["sources"][source][month]
        bucket[sentiment.lower()] += 1
        bucket["total"] += 1

    if skipped_no_date:
        print(f"  Skipped {skipped_no_date} record(s) with missing/unparseable post_date")

    # collapse defaultdicts into plain dicts, sorted by month, for clean JSON output
    output_products = {}
    for sku, product in products.items():
        sources_out = {}
        for source, months in product["sources"].items():
            sources_out[source] = {month: months[month] for month in sorted(months.keys())}
        output_products[sku] = {
            "product_title": product["product_title"],
            "sources": sources_out,
        }
    return output_products


def run(
    brand: str,
    sources: Optional[list[str]] = None,
    input_paths: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> dict:
    sources = sources or ["tiktok", "reddit"]
    slug = brand_to_slug(brand)
    output_path = output_path or f"output/{slug}_sentiment_trends.json"

    print(f"Brand: {brand}")
    print(f"Sources: {sources}")

    if input_paths:
        paths = input_paths
    else:
        print("Locating match files...")
        paths = find_input_files(slug, sources)

    if not paths:
        print("No input files found -- nothing to analyze.")
        return {}

    print("Loading records...")
    records = load_records(paths)
    print(f"{len(records)} unique matched records across all input files")

    print("Building monthly sentiment trends per product/source...")
    products = build_trends(records)

    result = {
        "brand": brand,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "products": products,
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote sentiment trends for {len(products)} product(s) to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Track monthly sentiment shifts per product, split by source."
    )
    parser.add_argument("--brand", required=True, help='Brand name, e.g. "Rare Beauty"')
    parser.add_argument("--sources", nargs="+", default=["tiktok", "reddit"],
                         choices=["tiktok", "reddit"],
                         help="Which sources to include (default: both)")
    parser.add_argument("--input-paths", nargs="+", default=None,
                         help="Explicit match-output files to use instead of auto-locating them")
    parser.add_argument("--output-path", default=None, help="Override output JSON path")
    args = parser.parse_args()

    run(
        brand=args.brand,
        sources=args.sources,
        input_paths=args.input_paths,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()