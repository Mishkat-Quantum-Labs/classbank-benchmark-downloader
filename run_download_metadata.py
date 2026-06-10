"""Download metadata and generate dataset documentation."""

import logging

from pipeline.download.metadata import download_all_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

if __name__ == "__main__":
    download_all_metadata()
