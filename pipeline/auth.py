"""Shared TalkBank authentication for all download scripts."""

import json
import logging
import sys

import requests

from pipeline.config import TALKBANK_AUTH_URL, TALKBANK_EMAIL, TALKBANK_PASSWORD

logger = logging.getLogger(__name__)


def create_session() -> requests.Session:
    """Authenticate with TalkBank and return a configured requests.Session.

    Reads TALKBANK_EMAIL and TALKBANK_PASSWORD from the environment (via .env).
    Exits with code 1 if credentials are missing or authentication fails.

    Returns:
        An authenticated requests.Session with appropriate headers and connection pooling.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=20, pool_maxsize=20, max_retries=3
    )
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json",
    })

    if not TALKBANK_EMAIL or not TALKBANK_PASSWORD:
        logger.error(
            "Missing credentials. Set TALKBANK_EMAIL and TALKBANK_PASSWORD in .env file."
        )
        logger.error("See .env.example for the required format.")
        sys.exit(1)

    resp = session.post(
        TALKBANK_AUTH_URL,
        data=json.dumps({"email": TALKBANK_EMAIL, "pswd": TALKBANK_PASSWORD}),
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
