from typing import Annotated

from pydantic import Field

from arxiv_at_home.api.component.citation_provider.base import CitationProvider
from arxiv_at_home.api.component.citation_provider.noop import NoOpCitationProvider, NoOpCitationProviderConfig
from arxiv_at_home.api.component.citation_provider.semantic_scholar import (
    SemanticScholarConfig,
    SemanticScholarProvider,
)

AnyCitationProviderConfig = Annotated[SemanticScholarConfig | NoOpCitationProviderConfig, Field(discriminator="type")]


def create_citation_provider(config: AnyCitationProviderConfig) -> CitationProvider:
    match config:
        case SemanticScholarConfig():
            return SemanticScholarProvider(config)
        case NoOpCitationProviderConfig():
            return NoOpCitationProvider()
        case _:
            raise ValueError("Unknown citation provider")
