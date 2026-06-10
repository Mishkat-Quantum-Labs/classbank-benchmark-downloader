"""Pipeline configuration — loads from .env and defines shared constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
MEDIA_DIR: Path = DATASET_DIR / "media"
TRANSCRIPTS_DIR: Path = DATASET_DIR / "transcripts"
PARSED_DIR: Path = DATASET_DIR / "parsed"
METADATA_DIR: Path = DATASET_DIR / "metadata"
LIMITATION_SET_DIR: Path = PROJECT_ROOT / "data-limitation-set" / "dataset"

# ── TalkBank Credentials ────────────────────────────────────────────────────

TALKBANK_EMAIL: str = os.getenv("TALKBANK_EMAIL", "")
TALKBANK_PASSWORD: str = os.getenv("TALKBANK_PASSWORD", "")

# ── TalkBank URLs ───────────────────────────────────────────────────────────

TALKBANK_AUTH_URL: str = "https://sla2.talkbank.org:443/logInUser"
TALKBANK_API_URL: str = "https://sla2.talkbank.org:1515"
TALKBANK_ACCESS_PAGE: str = "https://talkbank.org/class/access/index.html"
MEDIA_BASE_URL: str = "https://media.talkbank.org:443/class/{corpus}/"
TRANSCRIPT_ZIP_URL: str = "https://talkbank.org/data/class/{corpus}?f=zip"
TIMSS_MEDIA_URL: str = "https://media.talkbank.org:443/class/TIMSS-{subject}/{country}/"
TIMSS_TRANSCRIPT_ZIP_URL: str = "https://talkbank.org/data/class/TIMSS-{subject}/{country}?f=zip"

# ── Corpora ─────────────────────────────────────────────────────────────────

ENGLISH_CORPORA: list[str] = [
    "APT", "Bradford", "CarlaJim", "CogInst", "Crowley", "Curtis",
    "DISPEL", "Frederiksen", "Graesser", "Horowitz", "JLS", "Looney",
    "MacWhinney", "Moschkovich", "Person", "Rahm", "Roth", "Stevens",
    "TIMSS-Math", "TIMSS-Science", "Warren",
]

# Corpora ordered by expected size (smallest first) — excludes TIMSS
MEDIA_CORPORA: list[str] = [
    "Bradford", "Horowitz", "Crowley", "Roth", "Warren", "Rahm",
    "Stevens", "Frederiksen", "Moschkovich", "JLS", "Looney", "Person",
    "CogInst", "Graesser", "MacWhinney", "DISPEL", "CarlaJim", "Curtis",
    "APT",
]

# Mapping from access page display names to actual folder names
CORPUS_NAME_MAP: dict[str, str] = {
    "Cognition&Instruction": "CogInst",
    "Greeno/VanDeSande": "CarlaJim",
}

# Corpora handled by separate TIMSS scripts
SKIP_CORPORA: list[str] = ["TIMSS-Math", "TIMSS-Science"]

TIMSS_SUBJECTS: list[str] = ["Math", "Science"]
TIMSS_COUNTRIES: dict[str, list[str]] = {
    "Math": ["USA"],
    "Science": ["USA"],
}

# ── Corpus Descriptions ─────────────────────────────────────────────────────

CORPUS_DESCRIPTIONS: dict[str, str] = {
    "APT": "Academically productive talk",
    "Bradford": "School lessons on cultural literacy",
    "CarlaJim": "Math lessons",
    "CogInst": "Problem-based learning in medical school",
    "Crowley": "Exploration of electricity in a science museum",
    "Curtis": "Second-grade geometry lessons",
    "DISPEL": "Tutorial game environment for dysrhythmic phonation",
    "Frederiksen": "Statistics tutoring",
    "Graesser": "Research methodology tutoring",
    "Horowitz": "Lessons on camels",
    "JLS": "Lessons on statistical graphing",
    "Looney": "Classroom interactions",
    "MacWhinney": "Lectures on Psychology Research Methods",
    "Moschkovich": "Math word problem solving",
    "Person": "Statistics tutoring",
    "Rahm": "Museum lessons on the color of the sky",
    "Roth": "Geography lesson",
    "Stevens": "Architecture discussions",
    "TIMSS-Math": "Math classroom recordings from multiple countries",
    "TIMSS-Science": "Science classroom recordings from multiple countries",
    "Warren": "Teachers' discussion of gravity",
}

# ── Media Settings ──────────────────────────────────────────────────────────

MEDIA_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mp3", ".wav", ".m4v", ".m4a", ".avi", ".mov", ".m4b",
)

# ── Language Mapping (ISO 639-3 → ISO 639-1) ───────────────────────────────

LANGUAGE_MAP: dict[str, str] = {
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "zho": "zh",
    "jpn": "ja",
    "kor": "ko",
}
