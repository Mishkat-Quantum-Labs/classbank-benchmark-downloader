"""Move warn-status files to data-limitation-set."""

import logging

from pipeline.quality_filter import separate_warn_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

if __name__ == "__main__":
    separate_warn_files()
