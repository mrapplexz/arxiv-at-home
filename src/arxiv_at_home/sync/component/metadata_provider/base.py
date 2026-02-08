import abc
import dataclasses
import datetime
from collections.abc import AsyncGenerator

from arxiv_at_home.common.dto import PaperMetadata


@dataclasses.dataclass
class MetadataFetchProgress:
    metadata: PaperMetadata | None

    progress: int


@dataclasses.dataclass
class MetadataFetchResult:
    total_progress: int
    generator: AsyncGenerator[MetadataFetchProgress, None]


class PaperMetadataProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def provides_source(self) -> str: ...

    @abc.abstractmethod
    async def fetch_metadata(self, since: datetime.datetime | None) -> MetadataFetchResult: ...
