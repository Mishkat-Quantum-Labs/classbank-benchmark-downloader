"""Download TIMSS media and transcripts from TalkBank ClassBank."""

import logging

from pipeline.download.timss_media import download_all_timss_media
from pipeline.download.timss_transcripts import download_all_timss_transcripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_timss.log")],
)

if __name__ == "__main__":
    download_all_timss_media()
    download_all_timss_transcripts()
