import datetime
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

import aiofiles
import aiofiles.os
from pydantic import BaseModel, Field, field_validator

from arxiv_at_home.common.dto import PaperMetadata, PaperMetadataVersion
from arxiv_at_home.sync.component.metadata_provider.base import (
    MetadataFetchProgress,
    MetadataFetchResult,
    PaperMetadataProvider,
)


class KaggleDumpPaperMetadataProviderConfig(BaseModel):
    type: Literal["kaggle_json"] = "kaggle_json"

    path: Path


class KaggleDatasetVersion(BaseModel):
    version: str
    created: datetime.datetime

    @field_validator("created", mode="before")
    @classmethod
    def parse_created_date(cls, v: datetime.datetime | str) -> datetime.datetime:
        if isinstance(v, str):
            # Matches format: "Mon, 2 Apr 2007 19:18:42 GMT"
            # Note: strptime creates a naive datetime (no timezone).
            return datetime.datetime.strptime(v, "%a, %d %b %Y %H:%M:%S GMT").astimezone(datetime.UTC)
        return v


class KaggleDatasetRow(BaseModel):
    id: str
    title: str
    abstract: str
    authors: str
    categories: str
    doi: str | None = None
    update_date: datetime.date
    license: str | None = None
    journal_ref: str | None = Field(None, alias="journal-ref")
    versions: list[KaggleDatasetVersion]


class KaggleDumpPaperMetadataProvider(PaperMetadataProvider):
    def __init__(self, config: KaggleDumpPaperMetadataProviderConfig) -> None:
        self._config = config

    @property
    def provides_source(self) -> str:
        return "arxiv"

    async def fetch_metadata(self, since: datetime.datetime | None) -> MetadataFetchResult:
        total_bytes = (await aiofiles.os.stat(self._config.path)).st_size

        return MetadataFetchResult(total_progress=total_bytes, generator=self._stream_rows(since))

    async def _stream_rows(self, since: datetime.datetime | None) -> AsyncGenerator[MetadataFetchProgress, None]:
        current_bytes = 0

        async with aiofiles.open(self._config.path, "rb") as f:
            async for line_bytes in f:
                current_bytes += len(line_bytes)

                row = KaggleDatasetRow.model_validate_json(line_bytes)

                updated_at_comparator = datetime.datetime.combine(row.update_date, datetime.time.min).astimezone(
                    tz=datetime.UTC
                )

                if since is None or updated_at_comparator >= since:
                    yield MetadataFetchProgress(metadata=self._map_to_dto(row), progress=current_bytes)
                else:
                    yield MetadataFetchProgress(metadata=None, progress=current_bytes)

    def _map_to_dto(self, row: KaggleDatasetRow) -> PaperMetadata:
        return PaperMetadata(
            source=self.provides_source,
            id=row.id,
            title=row.title,
            abstract=row.abstract,
            authors=row.authors,
            categories=set(row.categories.split()),
            doi=row.doi,
            license=row.license,
            updated_at=row.update_date,
            journal_ref=row.journal_ref,
            versions=[PaperMetadataVersion(created=ver.created, version=ver.version) for ver in row.versions],
        )
