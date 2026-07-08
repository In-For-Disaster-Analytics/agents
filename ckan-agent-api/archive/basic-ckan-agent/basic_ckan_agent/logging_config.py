from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from basic_ckan_agent.settings import env


LOG_DIR = Path(env("CKAN_AGENT_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"ckan-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)

logger = logging.getLogger("ckan-agent")


def debug_print(title: str, data: Any | None = None) -> None:
    logger.debug("=" * 80)
    logger.debug("DEBUG: %s", title)
    logger.debug("=" * 80)

    if data is None:
        return

    if isinstance(data, str):
        logger.debug(data)
        return

    try:
        logger.debug(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    except TypeError:
        logger.debug(str(data))

