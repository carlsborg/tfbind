"""Download the DNA chapter dataset from the Deep Learning for Biology assets."""

import os
from urllib.parse import urljoin

from parfive import Downloader, SessionConfig
import requests


BASE_URL = "https://assets.deep-learning-for-biology.com"
DESTINATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CHUNK = 256 * 1024 * 1024  # 256 MB


def get_dna_prefixes(base_url: str, models: bool = True):
    """Fetch urls.txt and return (size, prefix) pairs for the dna chapter."""
    r = requests.get(urljoin(base_url, "urls.txt"), timeout=60)
    r.raise_for_status()
    sized_prefixes = []
    for raw in r.text.splitlines():
        line = raw.strip()
        parts = line.split(None, 1)
        size, prefix = int(parts[0]), parts[1]
        chapter, kind, _ = prefix.split("/", 2)
        if chapter == "dna" and (kind == "datasets" or (models and kind == "models")):
            sized_prefixes.append((size, prefix))
    return sized_prefixes


def download_dna(
    base_url: str = BASE_URL,
    destination: str = DESTINATION,
    models: bool = True,
    chunk: int = CHUNK,
):
    """Download the DNA chapter dataset (and optionally pretrained models)."""
    sized_prefixes = get_dna_prefixes(base_url, models)
    dl_small = Downloader(
        max_conn=64,
        max_splits=1,
        progress=True,
        config=SessionConfig(file_progress=False),
    )
    dl_big = Downloader(max_conn=4, max_splits=8, progress=True)
    for size, prefix in sized_prefixes:
        url = urljoin(base_url, prefix)
        dest = os.path.dirname(os.path.join(destination, prefix))
        os.makedirs(dest, exist_ok=True)
        if size <= chunk:
            dl_small.enqueue_file(url=url, path=dest)
        else:
            dl_big.enqueue_file(url=url, path=dest)
    if dl_small.queued_downloads:
        dl_small.download()
    if dl_big.queued_downloads:
        dl_big.download()


if __name__ == "__main__":
    download_dna()
