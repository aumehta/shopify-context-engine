# shopify-context-engine

Pulls TikTok and Reddit posts mentioning a brand, matches them to specific products in that brand's Shopify catalog, and tracks how sentiment toward each product shifts over time.

## What it does

1. **Scrape UGC** — Finds TikToks and Reddit posts that mention a brand (via Bright Data)
2. **Scrape catalog** — Pulls the brand's live product catalog from Shopify
3. **Match** — Uses embeddings + an LLM to figure out which specific product each post is talking about, and how the poster feels about it
4. **Analyze trends** — Rolls all matches up into monthly sentiment trends, per product, per platform

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your API keys:

```bash
export BRIGHTDATA_API_KEY=...
export OPENAI_API_KEY=...
```

## Usage

Run the pipeline in order. Using the same `--brand` name throughout is what lets each script automatically find the files the previous step produced.

**1. Scrape UGC (TikTok + Reddit)**

```bash
python3 scrape_tiktok.py --keyword "Rare Beauty" --days 365 --limit 100 --refresh
python3 scrape_reddit.py --keyword "Rare Beauty" --days 365 --limit 100 --refresh
```

- `--days` — how far back to pull posts from
- `--limit` — max posts to return
- `--refresh` — ignore the cache and re-scrape fresh (otherwise cached results are reused)

**2. Scrape the product catalog**

```bash
python3 scrape_catalog.py --brand "Rare Beauty" --domain rarebeauty.com
```

**3. Match UGC to products**

Run once per source:

```bash
python3 match_ugc_to_products.py --brand "Rare Beauty" --source tiktok --days 365
python3 match_ugc_to_products.py --brand "Rare Beauty" --source reddit --days 365
```

**4. Analyze sentiment trends**

```bash
python3 analyze_sentiment_trends.py --brand "Rare Beauty"
```

This produces `output/rare_beauty_sentiment_trends.json`, showing sentiment (positive/neutral/negative) per product, per platform, broken down by month.

## Running it on a new brand

Swap the brand name, keyword, and domain — everything else works the same:

```bash
python3 scrape_tiktok.py --keyword "mmLaFleur" --days 90 --limit 100 --refresh
python3 scrape_reddit.py --keyword "mmLaFleur" --days 90 --limit 100 --refresh
python3 scrape_catalog.py --brand "mmLaFleur" --domain mmlafleur.com
python3 match_ugc_to_products.py --brand "mmLaFleur" --source tiktok --days 90
python3 match_ugc_to_products.py --brand "mmLaFleur" --source reddit --days 90
python3 analyze_sentiment_trends.py --brand "mmLaFleur"
```

## Project structure

```
requirements.txt                # Python dependencies
brightdata_utils.py            # shared Bright Data scrape/poll/download logic
scrape_tiktok.py                # scrapes TikTok posts for a keyword
scrape_reddit.py                # scrapes Reddit posts for a keyword
scrape_catalog.py               # scrapes a Shopify brand's product catalog
match_ugc_to_products.py        # matches UGC to catalog products via embeddings + LLM
analyze_sentiment_trends.py     # builds monthly sentiment trends per product/source

cache/                          # cached scrape results, per keyword + date window
catalog/                        # scraped product catalogs, per brand
output/                         # match results and sentiment trend reports
```

## Notes

- All scripts default to file paths derived from `--brand`/`--keyword` (slugified), so as long as you're consistent with naming, each step automatically finds the right input from the previous one. Every script also supports explicit path overrides if you need to point somewhere custom.
- Scrape results are cached by keyword + date window; matching and analysis are not cached and re-run fresh each time.
- `analyze_sentiment_trends.py` can take multiple match-output files at once via `--input-paths`, and will dedupe overlapping records automatically — useful for combining runs from different time windows.
