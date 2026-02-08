# Arxiv at Home

## How to Run (Local)

* Setup your own infra or just run `docker compose up -d` to launch dependent services
* Edit `.env` (see `.env.example`)
* `uv run python -m arxiv_at_home.migrate`
* `uv run python -m arxiv_at_home.sync --config-path example/sync.json`
* `uv run python -m arxiv_at_home.index --config-path example/index.json`
* `uv run python -m arxiv_at_home.api --config-path example/api.json`
