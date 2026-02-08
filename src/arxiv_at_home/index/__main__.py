from pathlib import Path

import cyclopts

from arxiv_at_home.index.engine import IndexEngine
from arxiv_at_home.index.settings import IndexSettings


async def main(config_path: Path) -> None:
    config = IndexSettings.model_validate_json(config_path.read_text(encoding="utf-8"))
    engine = IndexEngine(config)
    await engine.index()


if __name__ == "__main__":
    cyclopts.run(main)
