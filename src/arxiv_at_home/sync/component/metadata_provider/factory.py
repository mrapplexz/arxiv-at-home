from typing import Annotated

from pydantic import Field

from arxiv_at_home.sync.component.metadata_provider.base import PaperMetadataProvider
from arxiv_at_home.sync.component.metadata_provider.json import (
    KaggleDumpPaperMetadataProvider,
    KaggleDumpPaperMetadataProviderConfig,
)

AnyPaperProviderConfig = Annotated[KaggleDumpPaperMetadataProviderConfig, Field(discriminator="type")]


def paper_metadata_provider_from_config(config: AnyPaperProviderConfig) -> PaperMetadataProvider:
    match config:
        case KaggleDumpPaperMetadataProviderConfig():
            return KaggleDumpPaperMetadataProvider(config)
        case _:
            raise ValueError(f"Unknown config {type(config)}")
