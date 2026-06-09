"""
TalkBank ClassBank - TIMSS Transcript Downloader
=================================================

Downloads CHAT-format transcript files from TalkBank's TIMSS-Math and TIMSS-Science collections.

TIMSS data differs from other ClassBank corpora because transcripts are per-country zip files:
    https://talkbank.org/data/class/TIMSS-{subject}/{country}?f=zip

Requires a free TalkBank account (register at https://class.talkbank.org).
Credentials are loaded from a .env file (see .env.example).

Usage:
    python download_timss_transcripts.py [--output-dir ./dataset] [--subjects Math,Science] [--countries USA]
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
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRANSCRIPT_ZIP_URL = "https://talkbank.org/data/class/TIMSS-{subject}/{country}?f=zip"

TIMSS_SUBJECTS = ["Math", "Science"]

TIMSS_COUNTRIES = {
    "Math": ["USA"],
    "Science": ["USA"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download_timss_transcripts.log")],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def create_session():
    """Authenticate with TalkBank and return session. Reads credentials from .env."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json",
    })

    email = os.environ.get("TALKBANK_EMAIL")
    password = os.environ.get("TALKBANK_PASSWORD")

    if not email or not password:
        logger.error("Missing credentials. Set TALKBANK_EMAIL and TALKBANK_PASSWORD in .env file.")
        logger.error("See .env.example for the required format.")
        sys.exit(1)

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


def download_transcripts(subject, country, output_dir, session):
    """Download and extract transcript zip for a TIMSS subject/country."""
    transcript_dir = output_dir / "transcripts" / f"TIMSS-{subject}"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    zip_url = TRANSCRIPT_ZIP_URL.format(subject=subject, country=country)
    logger.info(f"Downloading: TIMSS-{subject}/{country} from {zip_url}")

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
            zip_path = transcript_dir / f"TIMSS-{subject}.zip"
            with open(zip_path, "wb") as f:
                f.write(resp.content)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(transcript_dir)
                logger.info(f"  Extracted TIMSS-{subject}/{country} transcripts")
                zip_path.unlink()
            except zipfile.BadZipFile:
                logger.warning(f"  Bad zip file for TIMSS-{subject}/{country}")
            return True
        else:
            if "authModals" in resp.text or "Login" in resp.text[:500]:
                logger.warning(f"  Auth required for TIMSS-{subject}/{country} - check credentials")
            else:
                logger.warning(f"  Unexpected response for TIMSS-{subject}/{country}: "
                               f"HTTP {resp.status_code}")
            return False

    except Exception as e:
        logger.error(f"  Error downloading TIMSS-{subject}/{country}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download TIMSS CHAT transcripts from TalkBank ClassBank."
    )
    parser.add_argument("--output-dir", type=str, default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated subjects: Math,Science (default: both)")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated countries (default: all available per subject)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else TIMSS_SUBJECTS

    session = create_session()

    results = {}
    for subject in subjects:
        if subject not in TIMSS_COUNTRIES:
            logger.warning(f"Unknown TIMSS subject: {subject}. Use Math or Science.")
            continue

        available_countries = TIMSS_COUNTRIES[subject]
        if args.countries:
            countries = [c.strip() for c in args.countries.split(",")]
            for c in countries:
                if c not in available_countries:
                    logger.warning(f"  Country '{c}' not available for TIMSS-{subject}. "
                                   f"Available: {', '.join(available_countries)}")
            countries = [c for c in countries if c in available_countries]
        else:
            countries = available_countries

        logger.info(f"=== TIMSS-{subject} ===")
        logger.info(f"Countries: {', '.join(countries)}")

        for country in tqdm(countries, desc=f"TIMSS-{subject} transcripts"):
            key = f"TIMSS-{subject}/{country}"
            results[key] = download_transcripts(subject, country, output_dir, session)
            time.sleep(1)

    # Summary
    success = sum(1 for v in results.values() if v)
    logger.info(f"Done: {success}/{len(results)} TIMSS country datasets downloaded successfully")

    if results:
        logger.info("Results:")
        for key, status in results.items():
            logger.info(f"  {key}: {'OK' if status else 'FAILED'}")


if __name__ == "__main__":
    main()
