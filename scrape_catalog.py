"""
scrape_catalog.py

Scrapes a Shopify brand's product catalog from its public /products.json endpoint.

CLI usage:
    python3 scrape_catalog.py --brand "Rare Beauty" --domain rarebeauty.com
    python3 scrape_catalog.py --brand "mmLaFleur" --domain mmlafleur.com --limit 250

Can also be imported and called directly, e.g. from a future UI/backend:
    from scrape_catalog import scrape_catalog
    scrape_catalog(brand="Rare Beauty", domain="rarebeauty.com")
"""

import argparse
import json
import os
import re

import requests


def slugify(brand: str) -> str:
    """Turn a brand name into a filesystem-safe slug, used consistently
    across all scripts (catalog / scraper / matcher) so paths line up
    automatically whenever you change brands."""
    return re.sub(r"[^a-z0-9]+", "_", brand.lower()).strip("_")


def scrape_catalog(brand: str, domain: str, limit: int = 250, out_dir: str = "catalog") -> list[dict]:
    """
    Fetch a Shopify store's product catalog.

    Args:
        brand: Display name, e.g. "Rare Beauty". Used to name the output file.
        domain: Store domain without protocol, e.g. "rarebeauty.com".
        limit: Max products to fetch (Shopify caps at 250 per request).
        out_dir: Directory to save the catalog JSON into.

    Returns:
        The list of product dicts, and also writes them to
        {out_dir}/{slug}_products.json
    """
    url = f"https://{domain}/products.json"

    response = requests.get(url, params={"limit": limit})
    response.raise_for_status()

    products = response.json()["products"]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{slugify(brand)}_products.json")

    with open(out_path, "w") as f:
        json.dump(products, f, indent=2)

    print(f"Saved {len(products)} products to {out_path}")
    return products


def main():
    parser = argparse.ArgumentParser(description="Scrape a Shopify brand's product catalog.")
    parser.add_argument("--brand", required=True, help='Brand display name, e.g. "Rare Beauty"')
    parser.add_argument("--domain", required=True, help="Store domain, e.g. rarebeauty.com (no https://)")
    parser.add_argument("--limit", type=int, default=250, help="Max products to fetch (default: 250)")
    parser.add_argument("--out-dir", default="catalog", help="Output directory (default: catalog)")
    args = parser.parse_args()

    scrape_catalog(brand=args.brand, domain=args.domain, limit=args.limit, out_dir=args.out_dir)


if __name__ == "__main__":
    main()