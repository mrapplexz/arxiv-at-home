from pathlib import Path

import cyclopts
import uvicorn
from fastapi import FastAPI

from arxiv_at_home.api.dependencies import lifespan_factory
from arxiv_at_home.api.router import router
from arxiv_at_home.api.settings import ApiSettings


def main(config_path: Path) -> None:
    config = ApiSettings.model_validate_json(config_path.read_text(encoding="utf-8"))
    app = FastAPI(title="Arxiv-at-Home API", lifespan=lifespan_factory(config))
    app.include_router(router, prefix="/api/v1")
    uvicorn.run(app, host=config.serving.host, port=config.serving.port)


if __name__ == "__main__":
    cyclopts.run(main)
