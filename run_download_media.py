"""Download media files from TalkBank ClassBank."""

import logging

from pipeline.download.media import download_all_media

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_media.log")],
)

if __name__ == "__main__":
    download_all_media()
