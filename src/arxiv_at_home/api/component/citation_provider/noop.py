from typing import Literal

from pydantic import BaseModel

from arxiv_at_home.api.component.citation_provider.base import CitationProvider


class NoOpCitationProviderConfig(BaseModel):
    type: Literal["noop"] = "noop"


class NoOpCitationProvider(CitationProvider):
    async def get_citation_count_batch(self, paper_ids: list[str]) -> dict[str, int | None]:
        return {x: None for x in paper_ids}
