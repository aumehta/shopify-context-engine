"""
scrape_reddit.py

Discovers Reddit posts mentioning a keyword via Bright Data
(discover_by=keyword, same pattern as scrape_tiktoks.py/scrape_youtube.py),
filters to a configurable recency window, and caches results.

Bright Data's reddit_posts dataset (gd_lvz8ah06191smkebj4) supports keyword
discovery directly -- no separate search step needed. Its `date` param takes
a category (Past hour/day/week/month/year/All time) rather than a day count,
so `--days` is mapped to the closest matching category and then results are
filtered precisely to `--days` locally, same as the other scrapers.

CLI usage:
    python3 scrape_reddit.py --keyword "Rare Beauty" --days 30 --limit 50
    python3 scrape_reddit.py --keyword "Rare Beauty" --days 30 --limit 50 --refresh

Can also be imported:
    from scrape_reddit import get_results
    results = get_results(keyword="Rare Beauty", days=30, limit=50)

Env vars:
    BRIGHTDATA_API_KEY
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
import time
from brightdata_utils import run_brightdata_job, slugify

DATASET_ID = "gd_lvz8ah06191smkebj4"


def days_to_date_filter(days: int) -> str:
    """Maps a day count to Bright Data's nearest `date` category. We pick the
    smallest category that still covers `days`, then filter precisely to
    `days` locally in filter_by_recency() -- this param just controls how
    much Bright Data pulls before we trim it."""
    if days <= 1:
        return "Past day"
    if days <= 7:
        return "Past week"
    if days <= 31:
        return "Past month"
    if days <= 366:
        return "Past year"
    return "All time"


def scrape_brightdata(keyword: str, days: int, num_posts: int = 100) -> list[dict]:
    """Runs a Bright Data keyword-discovery scrape for `keyword`."""
    payload = {
        "input": [{
            "keyword": keyword,
            "date": days_to_date_filter(days),
            "num_of_posts": num_posts,
        }],
    }
    params = {"notify": "false", "include_errors": "true", "type": "discover_new", "discover_by": "keyword"}
    return run_brightdata_job(DATASET_ID, payload, params)


def filter_by_recency(results: list[dict], days: int) -> list[dict]:
    """Keep only posts posted within the last `days` days -- Bright Data's
    `date` filter is a coarse category, so trim precisely here."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    filtered = []
    for result in results:
        date_posted = result.get("date_posted")
        if not date_posted:
            continue
        try:
            post_date = datetime.fromisoformat(date_posted.replace("Z", "+00:00"))
        except ValueError:
            print(f"Could not parse date: {date_posted}")
            continue
        if post_date >= cutoff:
            filtered.append(result)

    return filtered


def normalize(item: dict) -> dict:
    """
    Flattens Bright Data's Reddit post schema into the shape
    match_ugc_to_products.py expects.

    For Reddit, use only the original post's title and body as the
    "transcript" for product matching. Comments are intentionally
    excluded because they may introduce unrelated products or opinions
    that could skew the match.
    """
    title = item.get("title", "")
    body = item.get("description", "")

    transcript = " ".join(filter(None, [title, body])).strip()

    return {
        **item,
        "transcript": transcript,
        "url": item.get("url", ""),
    }


def get_results(
    keyword: str,
    days: int = 30,
    limit: int = 50,
    refresh: bool = False,
    cache_dir: str = "cache",
) -> list[dict]:
    """
    Get up to `limit` Reddit posts mentioning `keyword`,
    posted within the last `days` days.

    Bright Data fetches extra posts because its date filter is coarse
    (Past day/week/month/year), and some posts will be filtered out
    by the precise local date filter.

    Args:
        keyword: Search term, e.g. "Rare Beauty".
        days: Recency window in days.
        limit: Maximum number of results to return.
        refresh: If True, ignore the cache and re-scrape.
        cache_dir: Directory to cache results in.

    Returns:
        List of Reddit post dicts, normalized with "transcript" and "url".
    """

    # Fetch extra results because some will be filtered out by date.
    scrape_pool = limit * 2

    # Cache is specific to the keyword + date window.
    cache_file = os.path.join(
        cache_dir,
        f"{slugify(keyword)}_reddit_{days}d.json",
    )

    # Use cache unless explicitly refreshing.
    if os.path.exists(cache_file) and not refresh:
        print(f"Loading cached results from {cache_file}...")

        with open(cache_file, "r") as f:
            cached_results = json.load(f)

        print(f"Loaded {len(cached_results)} cached results")

        return cached_results[:limit]

    # Fresh scrape.
    print(f"Scraping Bright Data for Reddit posts about '{keyword}'...")
    print(f"Requesting up to {scrape_pool} raw posts...")

    results = scrape_brightdata(
        keyword,
        days=days,
        num_posts=scrape_pool,
    )

    print(f"Bright Data returned {len(results)} results")

    # Apply precise local date filter.
    results = filter_by_recency(results, days=days)

    print(
        f"{len(results)} results were posted "
        f"within the last {days} days"
    )

    # Normalize after filtering.
    results = [normalize(item) for item in results]

    # Cache ALL filtered results, not just the final `limit`.
    os.makedirs(cache_dir, exist_ok=True)

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Cached {len(results)} results to {cache_file}")

    # Return at most `limit`.
    return results[:limit]


def main():
    parser = argparse.ArgumentParser(description="Scrape Reddit posts mentioning a keyword via Bright Data.")
    parser.add_argument("--keyword", required=True, help='Search keyword, e.g. "Rare Beauty"')
    parser.add_argument("--days", type=int, default=30, help="Recency window in days (default: 30)")
    parser.add_argument("--limit", type=int, default=50, help="Max results to return (default: 50)")
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