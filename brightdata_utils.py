"""
brightdata_utils.py

Shared helpers used by every scrape_*.py script
and the Bright Data "POST scrape -> poll snapshot -> download" flow, which
is identical across the TikTok / Reddit datasets -- only the
payload and dataset_id differ per platform.

"""

import os
import re
import json
import time
import requests

API_KEY = os.environ.get("BRIGHTDATA_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def slugify(text: str) -> str:
    """Shared across every script (catalog / tiktok / reddit / youtube /
    matcher) so cache, catalog, and output filenames line up automatically
    regardless of which script wrote them."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_response(response: requests.Response):
    """Bright Data can return either regular JSON or NDJSON."""
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return [
            json.loads(line)
            for line in response.text.strip().splitlines()
            if line.strip()
        ]


def run_brightdata_job(
    dataset_id: str,
    payload: dict,
    params: dict,
    poll_interval: int = 10,
    max_wait_seconds: int = 900,
) -> list[dict]:
    """
    POSTs a scrape job to Bright Data and returns the results, handling both
    the synchronous (200) and async (202 -> poll snapshot -> download) cases.

    Args:
        dataset_id: Bright Data dataset id (gd_...).
        payload: JSON body, typically {"input": [...], "limit_per_input": N}.
        params: query-string params, e.g. discover_by/type flags. dataset_id
            is added automatically -- don't include it here.
        poll_interval: seconds between snapshot status checks.
        max_wait_seconds: safety cap so a stuck job doesn't poll forever.
    """
    if not API_KEY:
        raise RuntimeError(
            "BRIGHTDATA_API_KEY is not set. Run `export BRIGHTDATA_API_KEY=...` first."
        )

    full_params = {"dataset_id": dataset_id, **params}

    response = requests.post(
        "https://api.brightdata.com/datasets/v3/scrape",
        params=full_params,
        headers=HEADERS,
        json=payload,
    )

    print("Initial response:", response.status_code)
    result = parse_response(response)

    if response.status_code == 200:
        print("Scrape completed immediately!")
        return result

    elif response.status_code == 202:
        snapshot_id = result[0]["snapshot_id"] if isinstance(result, list) else result["snapshot_id"]
        print(f"Snapshot ID: {snapshot_id}")
        print("Waiting for Bright Data...")

        waited = 0
        while waited < max_wait_seconds:
            progress = requests.get(
                f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}",
                headers=HEADERS,
            ).json()
            status = progress.get("status")
            print(f"Status: {status}")

            if status == "ready":
                break
            elif status == "failed":
                raise Exception(f"Bright Data scrape failed: {progress}")

            time.sleep(poll_interval)
            waited += poll_interval
        else:
            raise TimeoutError(f"Snapshot {snapshot_id} not ready after {max_wait_seconds}s")

        print("Downloading snapshot...")
        download_response = requests.get(
            f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
            params={"format": "json"},
            headers=HEADERS,
        )
        download_response.raise_for_status()
        results = parse_response(download_response)
        print(f"Downloaded {len(results)} results")
        return results

    else:
        response.raise_for_status()