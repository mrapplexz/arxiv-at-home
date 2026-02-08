import asyncio
from typing import Literal

import backoff
import httpx
from httpx import AsyncClient
from pydantic import BaseModel

from arxiv_at_home.api.component.citation_provider.base import CitationProvider


class SemanticScholarConfig(BaseModel):
    type: Literal["semantic_scholar"] = "semantic_scholar"

    url: str
    api_key: str
    proxy: str = None
    max_batch_size: int = 500


class SemanticScholarProvider(CitationProvider):
    MAX_BATCH_SIZE = 500

    def __init__(self, config: SemanticScholarConfig) -> None:
        self._client = AsyncClient(
            base_url=config.url, headers={"x-api-key": config.api_key} if config.api_key else {}, proxy=config.proxy
        )
        self._config = config

    @staticmethod
    def _normalize_id(paper_id: str) -> str:
        paper_source, paper_id = paper_id.split("/", maxsplit=1)
        # (ARXIV:, ACL:, etc)
        return f"{paper_source.upper()}:{paper_id}"

    @backoff.on_exception(backoff.expo, httpx.HTTPError, max_tries=3)
    async def _fetch_chunk(self, batch_ids: list[str]) -> list[dict]:
        # Prepare S2 specific IDs
        s2_ids = [self._normalize_id(pid) for pid in batch_ids]

        result = await self._client.post(
            "/graph/v1/paper/batch",
            json={"ids": s2_ids},
            params={"fields": "citationCount"},
        )
        result.raise_for_status()
        return result.json()

    async def get_citation_count_batch(self, paper_ids: list[str]) -> dict[str, int | None]:
        if not paper_ids:
            return {}

        chunks = [
            paper_ids[i : i + self._config.max_batch_size]
            for i in range(0, len(paper_ids), self._config.max_batch_size)
        ]

        results_dict: dict[str, int | None] = {}

        tasks = [self._fetch_chunk(chunk) for chunk in chunks]
        batch_responses = await asyncio.gather(*tasks)

        for chunk_index, response_list in enumerate(batch_responses):
            original_ids_chunk = chunks[chunk_index]

            for original_id, paper_data in zip(original_ids_chunk, response_list, strict=True):
                if paper_data and "citationCount" in paper_data:
                    results_dict[original_id] = paper_data["citationCount"]
                else:
                    results_dict[original_id] = None

        return results_dict
