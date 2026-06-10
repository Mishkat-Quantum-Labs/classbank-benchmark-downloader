"""Download CHAT transcripts from TalkBank ClassBank."""

import logging

from pipeline.download.transcripts import download_all_transcripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_transcripts.log")],
)

if __name__ == "__main__":
    download_all_transcripts()
