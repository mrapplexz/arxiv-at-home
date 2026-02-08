from pathlib import Path

import cyclopts

from arxiv_at_home.sync.engine import SyncEngine
from arxiv_at_home.sync.settings import SyncSettings


async def main(config_path: Path) -> None:
    config = SyncSettings.model_validate_json(config_path.read_text(encoding="utf-8"))
    engine = SyncEngine(config)
    await engine.sync()


if __name__ == "__main__":
    cyclopts.run(main)
