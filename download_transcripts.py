"""
TalkBank ClassBank - Transcript Downloader
==========================================

Downloads CHAT-format transcript files from TalkBank's ClassBank collection.

Transcripts are available as zip files from:
    https://talkbank.org/data/class/{corpus}?f=zip

Requires a free TalkBank account (register at https://class.talkbank.org).

Usage:
    python download_transcripts.py [--output-dir ./dataset] [--corpora APT,Bradford]
"""

import os
import sys
import json
import time
import zipfile
import argparse
import logging
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRANSCRIPT_ZIP_URL = "https://talkbank.org/data/class/{corpus}?f=zip"

ENGLISH_CORPORA = [
    "APT", "Bradford", "CarlaJim", "CogInst", "Crowley", "Curtis",
    "DISPEL", "Frederiksen", "Graesser", "Horowitz", "JLS", "Looney",
    "MacWhinney", "Moschkovich", "Person", "Rahm", "Roth", "Stevens",
    "TIMSS-Math", "TIMSS-Science", "Warren",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_transcripts.log")],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def create_session(email=None, password=None):
    """Authenticate with TalkBank and return session."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json",
    })

    if email is None:
        email = os.environ.get("TALKBANK_EMAIL") or input("TalkBank email: ").strip()
    if password is None:
        password = os.environ.get("TALKBANK_PASSWORD")
        if not password:
            import getpass
            password = getpass.getpass("TalkBank password: ")

    resp = session.post(
        "https://sla2.talkbank.org:443/logInUser",
        data=json.dumps({"email": email, "pswd": password}),
        timeout=30,
    )

    if resp.status_code == 200:
        data = resp.json()
        if data.get("success"):
            session.headers.pop("Content-Type", None)
            logger.info("Authenticated with TalkBank.")
            return session

    logger.error("Authentication failed. Register at https://class.talkbank.org")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Download Logic
# ---------------------------------------------------------------------------


def download_transcripts(corpus, output_dir, session):
    """Download and extract transcript zip for a corpus."""
    transcript_dir = output_dir / "transcripts" / corpus
    transcript_dir.mkdir(parents=True, exist_ok=True)

    zip_url = TRANSCRIPT_ZIP_URL.format(corpus=corpus)
    logger.info(f"Downloading: {corpus} from {zip_url}")

    try:
        resp = session.get(zip_url, timeout=120)

        content_type = resp.headers.get("content-type", "")
        is_zip = (
            "application/zip" in content_type
            or "application/octet-stream" in content_type
            or (resp.status_code == 200 and len(resp.content) > 500
                and resp.content[:4] == b'PK\x03\x04')
        )

        if is_zip:
            zip_path = transcript_dir / f"{corpus}.zip"
            with open(zip_path, "wb") as f:
                f.write(resp.content)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(transcript_dir)
                logger.info(f"  Extracted {corpus} transcripts")
                zip_path.unlink()
            except zipfile.BadZipFile:
                logger.warning(f"  Bad zip file for {corpus}")
            return True
        else:
            if "authModals" in resp.text or "Login" in resp.text[:500]:
                logger.warning(f"  Auth required for {corpus} - check credentials")
            else:
                logger.warning(f"  Unexpected response for {corpus}: HTTP {resp.status_code}")
            return False

    except Exception as e:
        logger.error(f"  Error downloading {corpus}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download CHAT transcripts from TalkBank ClassBank."
    )
    parser.add_argument("--output-dir", type=str, default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--corpora", type=str, default=None,
                        help="Comma-separated corpora to download (default: all)")
    parser.add_argument("--email", type=str, default=None,
                        help="TalkBank email (or set TALKBANK_EMAIL env var)")
    parser.add_argument("--password", type=str, default=None,
                        help="TalkBank password (or set TALKBANK_PASSWORD env var)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    corpora = [c.strip() for c in args.corpora.split(",")] if args.corpora else ENGLISH_CORPORA

    logger.info(f"Output: {output_dir}")
    logger.info(f"Corpora: {len(corpora)}")

    session = create_session(args.email, args.password)

    results = {}
    for corpus in tqdm(corpora, desc="Downloading transcripts"):
        results[corpus] = download_transcripts(corpus, output_dir, session)
        time.sleep(1)

    # Summary
    success = sum(1 for v in results.values() if v)
    logger.info(f"Done: {success}/{len(corpora)} corpora downloaded successfully")


if __name__ == "__main__":
    main()
