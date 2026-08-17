"""
scrape_tiktoks.py

Scrapes TikToks mentioning a keyword via Bright Data, filters to a configurable
recency window, and caches results.

CLI usage:
    python3 scrape_tiktoks.py --keyword "Rare Beauty" --days 7 --limit 100
    python3 scrape_tiktoks.py --keyword "Rare Beauty" --days 30 --limit 200 --refresh

Env vars:
    BRIGHTDATA_API_KEY
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
import time
from brightdata_utils import run_brightdata_job, slugify

DATASET_ID = "gd_lu702nij2f790tmv9h"


def scrape_brightdata(keyword: str, scrape_limit: int = 100) -> list[dict]:
    """Runs a Bright Data scrape for `keyword`."""
    payload = {
        "input": [{"search_keyword": keyword}],
        "limit_per_input": scrape_limit,
    }
    params = {
        "notify": "false",
        "include_errors": "true",
        "type": "discover_new",
        "discover_by": "keyword",
    }
    return run_brightdata_job(DATASET_ID, payload, params)


def filter_by_recency(results: list[dict], days: int) -> list[dict]:
    """Keep only TikToks posted within the last `days` days."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    filtered = []
    for result in results:
        create_time = result.get("create_time")
        if not create_time:
            continue

        try:
            post_date = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
        except ValueError:
            print(f"Could not parse date: {create_time}")
            continue

        if post_date >= cutoff:
            filtered.append(result)

    return filtered


def get_results(
    keyword: str,
    days: int = 7,
    limit: int = 100,
    refresh: bool = False,
    cache_dir: str = "cache",
) -> list[dict]:
    """
    Get up to `limit` TikToks mentioning `keyword`,
    posted within the last `days` days.

    The scraper automatically fetches extra raw results from
    Bright Data because some results may fall outside the date window.

    Args:
        keyword: Search term, e.g. "Rare Beauty".
        days: Recency window in days.
        limit: Maximum number of results to return.
        refresh: If True, ignore the cache and re-scrape.
        cache_dir: Directory to cache results in.

    Returns:
        List of TikTok result dicts.
    """

    # Fetch extra results because some will be filtered out by date.
    scrape_pool = limit * 2

    # Cache is specific to the keyword + date window.
    cache_file = os.path.join(
        cache_dir,
        f"{slugify(keyword)}_tiktok_{days}d.json",
    )

    # Use cache unless explicitly refreshing.
    if os.path.exists(cache_file) and not refresh:
        print(f"Loading cached results from {cache_file}...")

        with open(cache_file, "r") as f:
            cached_results = json.load(f)

        print(f"Loaded {len(cached_results)} cached results")

        return cached_results[:limit]

    # Fresh scrape.
    print(f"Scraping Bright Data for {keyword}...")
    print(f"Requesting up to {scrape_pool} raw results...")

    results = scrape_brightdata(
        keyword,
        scrape_limit=scrape_pool,
    )

    print(f"Bright Data returned {len(results)} results")

    # Apply date filter.
    results = filter_by_recency(results, days=days)

    print(
        f"{len(results)} results were posted "
        f"within the last {days} days"
    )

    # Cache ALL filtered results, not just the final `limit`.
    os.makedirs(cache_dir, exist_ok=True)

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Cached {len(results)} results to {cache_file}")

    # Return at most `limit`.
    return results[:limit]


def main():
    parser = argparse.ArgumentParser(description="Scrape TikToks mentioning a keyword via Bright Data.")
    parser.add_argument("--keyword", required=True, help='Search keyword, e.g. "Rare Beauty"')
    parser.add_argument("--days", type=int, default=7, help="Recency window in days (default: 7)")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max results to return (default: 100)",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-scrape")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory (default: cache)")
    args = parser.parse_args()

    results = get_results(
        keyword=args.keyword,
        days=args.days,
        limit=args.limit,
        refresh=args.refresh,
        cache_dir=args.cache_dir,
    )
    print(f"\nFinal result count: {len(results)}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")