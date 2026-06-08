"""
TalkBank ClassBank English Classroom Recordings Downloader
===========================================================

Downloads all English classroom recordings (audio/video) and their
CHAT transcriptions from TalkBank's ClassBank collection.

Uses the TBDBpy API (https://github.com/TalkBank/TBDBpy) to query
metadata, then downloads transcripts and media files.

Dataset Structure:
    dataset/
    ├── metadata/
    │   ├── corpus_manifest.json      # Full dataset manifest
    │   └── transcripts_index.csv     # Index of all transcripts
    ├── transcripts/
    │   ├── APT/
    │   │   ├── *.cha                 # CHAT format transcription files
    │   │   └── ...
    │   ├── Bradford/
    │   └── ...
    ├── media/
    │   ├── APT/
    │   │   ├── *.mp4 / *.mp3        # Audio/video recordings
    │   │   └── ...
    │   ├── Bradford/
    │   └── ...
    └── README.md                     # Dataset documentation

Requirements:
    pip install -r requirements.txt

Usage:
    python download_classbank.py [--output-dir ./dataset] [--skip-media] [--corpora APT,Bradford]

Note:
    Media files on TalkBank require a free registered account.
    The script will prompt for credentials if needed.
"""

import os
import sys
import json
import time
import zipfile
import argparse
import logging
from pathlib import Path
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TALKBANK_API_URL = "https://sla2.talkbank.org:1515"
TRANSCRIPT_ZIP_URL = "https://talkbank.org/data/class/{corpus}?f=zip"
MEDIA_BASE_URL = "https://media.talkbank.org/class/{corpus}"

# All English ClassBank corpora (from TalkBankDB path tree)
ENGLISH_CORPORA = [
    "APT",
    "Bradford",
    "CarlaJim",
    "CogInst",
    "Crowley",
    "Curtis",
    "DISPEL",
    "Frederiksen",
    "Graesser",
    "Horowitz",
    "JLS",
    "Looney",
    "MacWhinney",
    "Moschkovich",
    "Person",
    "Rahm",
    "Roth",
    "Stevens",
    "TIMSS-Math",
    "TIMSS-Science",
    "Warren",
]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("download.log")],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TalkBankDB API Functions (inline, no external tbdb dependency needed)
# ---------------------------------------------------------------------------


def tbdb_get_transcripts(corpus_name, corpora_paths):
    """Query TalkBankDB for transcript metadata."""
    url = f"{TALKBANK_API_URL}/getTranscriptSummary"
    query = {
        "corpusName": corpus_name,
        "corpora": corpora_paths,
        "lang": {},
        "media": {},
        "age": {},
        "gender": {},
        "designType": {},
        "activityType": {},
        "groupType": {},
        "respType": "JSON",
    }
    try:
        resp = requests.post(url, json={"queryVals": query}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"API query failed for {corpora_paths}: {e}")
        return {"colHeadings": [], "data": []}


def tbdb_get_utterances(corpus_name, corpora_paths):
    """Query TalkBankDB for utterance data."""
    url = f"{TALKBANK_API_URL}/getUtteranceSummary"
    query = {
        "corpusName": corpus_name,
        "corpora": corpora_paths,
        "lang": {},
        "media": {},
        "age": {},
        "gender": {},
        "designType": {},
        "activityType": {},
        "groupType": {},
        "respType": "JSON",
    }
    try:
        resp = requests.post(url, json={"queryVals": query}, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Utterance query failed for {corpora_paths}: {e}")
        return {"colHeadings": [], "data": []}


# ---------------------------------------------------------------------------
# Authentication for media downloads
# ---------------------------------------------------------------------------


class TalkBankSession:
    """Manages authenticated session for TalkBank downloads.
    
    TalkBank uses cookie-based authentication via the sla2.talkbank.org server.
    After login, a session cookie ('talkbank') is set that grants access to
    transcript and media downloads.
    """

    AUTH_SERVER = "https://sla2.talkbank.org:443"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
        })
        self.authenticated = False

    def authenticate(self, email=None, password=None):
        """Authenticate with TalkBank to access data files.
        
        Uses the logInUser endpoint which sets a session cookie.
        Register for free at https://class.talkbank.org (click Login > New User).
        """
        if email is None:
            email = input("TalkBank email: ").strip()
        if password is None:
            import getpass
            password = getpass.getpass("TalkBank password: ")

        login_url = f"{self.AUTH_SERVER}/logInUser"
        try:
            resp = self.session.post(
                login_url,
                data=json.dumps({"email": email, "pswd": password}),
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    self.authenticated = True
                    # Remove Content-Type for subsequent GET requests
                    self.session.headers.pop("Content-Type", None)
                    logger.info("Successfully authenticated with TalkBank.")
                    return True
                else:
                    msg = data.get("respMsg", "Unknown error")
                    logger.warning(f"Authentication failed: {msg}")
                    return False
            logger.warning(f"Authentication failed: HTTP {resp.status_code}")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Authentication connection error: {e}")
            logger.info("This may indicate incorrect credentials or rate limiting.")
            return False
        except Exception as e:
            logger.warning(f"Authentication error: {e}")
            return False

    def get(self, url, **kwargs):
        """Make an authenticated GET request."""
        kwargs.setdefault("timeout", 120)
        return self.session.get(url, **kwargs)


# ---------------------------------------------------------------------------
# Transcript Downloading
# ---------------------------------------------------------------------------


def download_transcripts(corpus, output_dir, session=None):
    """Download and extract transcript zip for a corpus.
    
    Transcripts are available as zip files from:
    https://talkbank.org/data/class/{corpus}?f=zip
    
    This requires an authenticated session (free registration).
    """
    transcript_dir = output_dir / "transcripts" / corpus
    transcript_dir.mkdir(parents=True, exist_ok=True)

    zip_url = TRANSCRIPT_ZIP_URL.format(corpus=corpus)
    logger.info(f"Downloading transcripts: {corpus} from {zip_url}")

    try:
        s = session.session if session else requests.Session()
        resp = s.get(zip_url, timeout=120)

        # Check if we got actual zip content (not an auth page)
        content_type = resp.headers.get("content-type", "")
        is_zip = (
            "application/zip" in content_type
            or "application/octet-stream" in content_type
            or (resp.status_code == 200 and len(resp.content) > 500 and resp.content[:4] == b'PK\x03\x04')
        )

        if is_zip:
            # Save and extract zip
            zip_path = transcript_dir / f"{corpus}.zip"
            with open(zip_path, "wb") as f:
                f.write(resp.content)

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(transcript_dir)
                logger.info(f"  Extracted {corpus} transcripts to {transcript_dir}")
                # Remove zip after extraction
                zip_path.unlink()
            except zipfile.BadZipFile:
                logger.warning(f"  Bad zip file for {corpus}, keeping raw download.")

            return True
        else:
            # Probably got an auth page back
            if "authModals" in resp.text or "Login" in resp.text[:500]:
                logger.warning(
                    f"  Auth required for {corpus} transcripts. Make sure you're logged in."
                )
            else:
                logger.warning(
                    f"  Unexpected response for {corpus}: HTTP {resp.status_code}, "
                    f"content-type={content_type}, size={len(resp.content)}"
                )
            return False
    except Exception as e:
        logger.error(f"  Error downloading {corpus} transcripts: {e}")
        return False


# ---------------------------------------------------------------------------
# Media Downloading
# ---------------------------------------------------------------------------


def get_media_file_list(corpus, session):
    """Scrape media file listing for a corpus from TalkBank media server.
    
    The media server returns HTML with file listings regardless of auth status
    (auth is enforced client-side via JS). We parse the HTML table to get
    direct download URLs.
    """
    media_url = f"https://media.talkbank.org:443/class/{corpus}/"
    logger.info(f"Fetching media listing: {media_url}")

    media_extensions = (".mp4", ".mp3", ".wav", ".m4v", ".m4a", ".avi", ".mov", ".m4b")

    try:
        resp = session.get(media_url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"  Cannot access media for {corpus}: HTTP {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        files = []
        subdirs = []

        # Parse all links from the page
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Skip save links (duplicates with ?f=save)
            if "?f=save" in href:
                continue

            # Media files
            if any(href.lower().endswith(ext) for ext in media_extensions):
                # URLs are already absolute (https://media.talkbank.org:443/...)
                if href.startswith("http"):
                    files.append(href)
                else:
                    files.append(f"https://media.talkbank.org:443/class/{corpus}/{href}")

            # Subdirectories (for nested corpora like TIMSS)
            elif (
                href.endswith("/")
                and "class" in href
                and "Parent" not in link.get_text()
                and href != media_url
            ):
                if href.startswith("http"):
                    subdirs.append(href)

        # Recursively check subdirectories
        for subdir_url in subdirs:
            try:
                time.sleep(0.3)
                sub_resp = session.get(subdir_url, timeout=30)
                if sub_resp.status_code == 200:
                    sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
                    for sub_link in sub_soup.find_all("a", href=True):
                        sub_href = sub_link["href"]
                        if "?f=save" in sub_href:
                            continue
                        if any(sub_href.lower().endswith(ext) for ext in media_extensions):
                            if sub_href.startswith("http"):
                                files.append(sub_href)
                            else:
                                files.append(f"{subdir_url}{sub_href}")
            except Exception:
                pass

        logger.info(f"  Found {len(files)} media files for {corpus}")
        return files
    except Exception as e:
        logger.error(f"  Error fetching media listing for {corpus}: {e}")
        return []


def download_media_file(url, output_path, session):
    """Download a single media file.
    
    The TalkBank media server requires explicit Range headers for downloads.
    We first do a HEAD-like request to get the file size, then download
    in chunks using Range requests.
    """
    try:
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.debug(f"  Skipping existing: {output_path.name}")
            return True

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # First, get file size with a small range request
        resp = session.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        if resp.status_code not in (200, 206):
            logger.warning(f"  Failed to access {url}: HTTP {resp.status_code}")
            return False

        # Parse total size from Content-Range header
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range:
            total_size = int(content_range.split("/")[-1])
        else:
            # Fallback: try full download
            total_size = None

        # Download the full file using Range header
        if total_size:
            resp = session.get(
                url,
                headers={"Range": f"bytes=0-{total_size - 1}"},
                timeout=600,
                stream=True,
            )
        else:
            resp = session.get(url, timeout=600, stream=True)

        if resp.status_code in (200, 206):
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return True
        else:
            logger.warning(f"  Failed to download {url}: HTTP {resp.status_code}")
            return False
    except Exception as e:
        logger.error(f"  Error downloading {url}: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


def download_corpus_media(corpus, output_dir, session, max_workers=5):
    """Download all media files for a corpus using parallel downloads."""
    media_dir = output_dir / "media" / corpus
    media_dir.mkdir(parents=True, exist_ok=True)

    files = get_media_file_list(corpus, session)
    if not files:
        return 0

    # Build download tasks
    tasks = []
    for url in files:
        filename = url.split("/")[-1].split("?")[0]
        output_path = media_dir / filename
        tasks.append((url, output_path))

    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_media_file, url, path, session): url
            for url, path in tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"  {corpus} media", leave=False
        ):
            if future.result():
                downloaded += 1

    logger.info(f"  Downloaded {downloaded}/{len(files)} media files for {corpus}")
    return downloaded


# ---------------------------------------------------------------------------
# Metadata Collection
# ---------------------------------------------------------------------------


def collect_corpus_metadata(corpus):
    """Collect metadata for a corpus using TalkBankDB API."""
    logger.info(f"Querying metadata for: {corpus}")

    # Query transcripts
    result = tbdb_get_transcripts("class", [["class", corpus]])

    metadata = {
        "corpus": corpus,
        "transcripts": [],
        "total_transcripts": 0,
        "media_types": set(),
        "languages": set(),
    }

    if result.get("data"):
        headers = result.get("colHeadings", [])
        for row in result["data"]:
            entry = dict(zip(headers, row)) if headers else {}
            metadata["transcripts"].append(entry)

            if "media" in entry:
                metadata["media_types"].add(entry["media"])
            if "languages" in entry:
                metadata["languages"].add(entry["languages"])

        metadata["total_transcripts"] = len(result["data"])

    # Convert sets to lists for JSON serialization
    metadata["media_types"] = list(metadata["media_types"])
    metadata["languages"] = list(metadata["languages"])

    return metadata


# ---------------------------------------------------------------------------
# Dataset Manifest and README
# ---------------------------------------------------------------------------


def generate_manifest(all_metadata, output_dir):
    """Generate dataset manifest JSON."""
    manifest = {
        "dataset_name": "TalkBank ClassBank English Classroom Recordings",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "source": "https://class.talkbank.org",
        "license": "CC BY-NC-SA 3.0",
        "description": (
            "English classroom recordings and CHAT transcriptions from TalkBank's "
            "ClassBank collection. Includes science, mathematics, medicine, and reading "
            "classroom interactions ranging from third grade to medical school."
        ),
        "corpora": all_metadata,
        "total_corpora": len(all_metadata),
        "total_transcripts": sum(m["total_transcripts"] for m in all_metadata),
    }

    manifest_path = output_dir / "metadata" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest saved to {manifest_path}")
    return manifest


def generate_transcripts_index(output_dir):
    """Generate a CSV index of all downloaded transcript files."""
    transcript_dir = output_dir / "transcripts"
    records = []

    for cha_file in transcript_dir.rglob("*.cha"):
        rel_path = cha_file.relative_to(transcript_dir)
        corpus = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
        records.append(
            {
                "corpus": corpus,
                "filename": cha_file.name,
                "relative_path": str(rel_path),
                "size_bytes": cha_file.stat().st_size,
            }
        )

    if records:
        df = pd.DataFrame(records)
        index_path = output_dir / "metadata" / "transcripts_index.csv"
        df.to_csv(index_path, index=False)
        logger.info(f"Transcript index saved: {len(records)} files indexed")
        return df

    return pd.DataFrame()


def generate_readme(manifest, output_dir):
    """Generate README for the dataset."""
    readme_content = f"""# TalkBank ClassBank English Classroom Recordings Dataset

## Overview

This dataset contains English classroom recordings and their CHAT-format transcriptions
from TalkBank's ClassBank collection (https://class.talkbank.org).

**Created:** {manifest['created']}
**License:** {manifest['license']}
**Total Corpora:** {manifest['total_corpora']}
**Total Transcripts:** {manifest['total_transcripts']}

## Description

ClassBank provides transcribed filmed interactions in various classrooms. Topics include
science, mathematics, medicine, and reading. Learners range from third grade to medical
school. All interactions are in English (except TIMSS Japanese classroom data which is
excluded from this collection).

## Dataset Structure

```
dataset/
├── metadata/
│   ├── corpus_manifest.json      # Full dataset manifest with metadata
│   └── transcripts_index.csv     # Index of all transcript files
├── transcripts/
│   ├── <corpus_name>/
│   │   └── *.cha                 # CHAT format transcription files
│   └── ...
├── media/
│   ├── <corpus_name>/
│   │   └── *.mp4 / *.mp3        # Audio/video recordings
│   └── ...
└── README.md
```

## Corpora Included

| Corpus | Description |
|--------|-------------|
| APT | Academically productive talk |
| Bradford | School lessons on cultural literacy |
| CogInst | Problem-based learning in medical school |
| Crowley | Exploration of electricity in a science museum |
| Curtis | Second-grade geometry lessons |
| DISPEL | Tutorial game environment for dysrhythmic phonation |
| Frederiksen | Statistics tutoring |
| Graesser | Research methodology tutoring |
| Greeno-VanDeSande | Math lessons |
| Horowitz | Lessons on camels |
| JLS | Lessons on statistical graphing |
| MacWhinney | Lectures on Psychology Research Methods |
| Moschkovich | Math word problem solving |
| Person | Statistics tutoring |
| Rahm | Museum lessons on the color of the sky |
| Roth | Geography lesson |
| Stevens | Architecture discussions |
| TIMSS-Math | Math classroom recordings from multiple countries |
| TIMSS-Science | Science classroom recordings from multiple countries |
| Warren | Teachers' discussion of gravity |

## Transcript Format (CHAT)

Transcripts use the CHAT (Codes for the Human Analysis of Transcripts) format developed
by Brian MacWhinney at Carnegie Mellon University. Key features:

- **Headers:** Lines beginning with `@` contain metadata (participants, date, etc.)
- **Main tiers:** Lines beginning with `*` contain speaker utterances (e.g., `*TEA:` for teacher)
- **Dependent tiers:** Lines beginning with `%` contain annotations (morphology, timing, etc.)
- **Timestamps:** Media alignment info in `%tim` tier or bullet markers

For full CHAT documentation: https://talkbank.org/0info/manuals/CHAT.pdf

## Usage for Benchmarking

This dataset can be used for:
- Speech recognition benchmarking (with audio-aligned transcripts)
- Speaker diarization evaluation
- Classroom discourse analysis
- Turn-taking and overlap detection
- Educational NLP tasks

## Citation

If you use this data, please cite:

```
MacWhinney, B. (2000). The CHILDES Project: Tools for Analyzing Talk (3rd ed.).
Mahwah, NJ: Lawrence Erlbaum Associates.
```

Additionally, cite the specific corpus contributor(s) as documented on each
corpus's page at https://class.talkbank.org.

## License

This data is shared under the Creative Commons BY-NC-SA 3.0 license.
See: https://creativecommons.org/licenses/by-nc-sa/3.0/
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info(f"README saved to {readme_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download TalkBank ClassBank English classroom recordings and transcriptions."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset",
        help="Output directory for the dataset (default: ./dataset)",
    )
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Skip downloading media files (only download transcripts)",
    )
    parser.add_argument(
        "--corpora",
        type=str,
        default=None,
        help="Comma-separated list of specific corpora to download (default: all)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="TalkBank account email for media access",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="TalkBank account password for media access",
    )
    parser.add_argument(
        "--skip-metadata-query",
        action="store_true",
        help="Skip querying TalkBankDB API for metadata (faster, less info)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Number of parallel media downloads (default: 5)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which corpora to download
    if args.corpora:
        corpora = [c.strip() for c in args.corpora.split(",")]
    else:
        corpora = ENGLISH_CORPORA

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Corpora to download: {len(corpora)}")
    logger.info(f"Skip media: {args.skip_media}")

    # -----------------------------------------------------------------------
    # Step 1: Collect metadata from TalkBankDB
    # -----------------------------------------------------------------------
    all_metadata = []
    if not args.skip_metadata_query:
        logger.info("=" * 60)
        logger.info("STEP 1: Collecting corpus metadata from TalkBankDB API")
        logger.info("=" * 60)

        for corpus in tqdm(corpora, desc="Querying metadata"):
            meta = collect_corpus_metadata(corpus)
            all_metadata.append(meta)
            time.sleep(0.5)  # Be respectful to the API
    else:
        # Create minimal metadata
        for corpus in corpora:
            all_metadata.append(
                {
                    "corpus": corpus,
                    "transcripts": [],
                    "total_transcripts": 0,
                    "media_types": [],
                    "languages": ["eng"],
                }
            )

    # -----------------------------------------------------------------------
    # Step 2: Authenticate (required for both transcripts and media)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Authenticating with TalkBank")
    logger.info("=" * 60)
    logger.info(
        "Both transcripts and media require a free TalkBank account.\n"
        "Register at https://class.talkbank.org (click Login > New User)."
    )

    session = TalkBankSession()
    if args.email and args.password:
        session.authenticate(args.email, args.password)
    else:
        logger.info("Please enter your TalkBank credentials:")
        session.authenticate()

    if not session.authenticated:
        logger.warning(
            "Authentication failed. Will attempt downloads anyway (some may fail).\n"
            "Register for free at https://class.talkbank.org if you don't have an account."
        )

    # -----------------------------------------------------------------------
    # Step 3: Download transcripts
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Downloading transcripts (CHAT format)")
    logger.info("=" * 60)

    transcript_results = {}
    for corpus in tqdm(corpora, desc="Downloading transcripts"):
        success = download_transcripts(corpus, output_dir, session)
        transcript_results[corpus] = success
        time.sleep(1)  # Rate limiting

    # -----------------------------------------------------------------------
    # Step 3: Download media files (if not skipped)
    # -----------------------------------------------------------------------
    if not args.skip_media:
        logger.info("=" * 60)
        logger.info("STEP 4: Downloading media files (audio/video)")
        logger.info("=" * 60)

        if session.authenticated:
            for corpus in tqdm(corpora, desc="Downloading media"):
                download_corpus_media(corpus, output_dir, session, max_workers=args.parallel)
                time.sleep(1)
        else:
            logger.warning(
                "Skipping media download - authentication failed. "
                "Register at https://class.talkbank.org to get credentials."
            )

    # -----------------------------------------------------------------------
    # Step 5: Generate metadata files
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Generating dataset metadata")
    logger.info("=" * 60)

    manifest = generate_manifest(all_metadata, output_dir)
    generate_transcripts_index(output_dir)
    generate_readme(manifest, output_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Dataset location: {output_dir}")
    logger.info(f"Corpora processed: {len(corpora)}")
    logger.info(
        f"Transcripts downloaded: {sum(1 for v in transcript_results.values() if v)}/{len(corpora)}"
    )

    # Count actual files
    cha_count = len(list((output_dir / "transcripts").rglob("*.cha"))) if (output_dir / "transcripts").exists() else 0
    media_count = 0
    if (output_dir / "media").exists():
        for ext in ["*.mp4", "*.mp3", "*.wav", "*.m4v", "*.m4a"]:
            media_count += len(list((output_dir / "media").rglob(ext)))

    logger.info(f"Total .cha transcript files: {cha_count}")
    logger.info(f"Total media files: {media_count}")


if __name__ == "__main__":
    main()
