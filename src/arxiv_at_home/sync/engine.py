import datetime
from collections.abc import AsyncGenerator

from tqdm import tqdm

from arxiv_at_home.common.database.manager import AsyncDatabaseManager, new_database_manager
from arxiv_at_home.common.database.repository import PaperMetadataRepository, SyncStateRepository
from arxiv_at_home.common.dto import PaperMetadata
from arxiv_at_home.sync.component.metadata_provider.base import MetadataFetchResult, PaperMetadataProvider
from arxiv_at_home.sync.component.metadata_provider.factory import paper_metadata_provider_from_config
from arxiv_at_home.sync.settings import SyncSettings


class SyncEngine:
    def __init__(self, settings: SyncSettings) -> None:
        self._config = settings
        self._metadata_providers = [paper_metadata_provider_from_config(prov) for prov in settings.metadata_providers]
        self._filter_categories = settings.filter_categories
        self._batch_size = settings.batch_size

    async def _stream_batches(
        self, fetch_result: MetadataFetchResult, pbar: tqdm
    ) -> AsyncGenerator[list[PaperMetadata], None]:
        batch: list[PaperMetadata] = []

        async for sample in fetch_result.generator:
            if sample.progress > pbar.n:
                pbar.update(sample.progress - pbar.n)

            meta = sample.metadata

            # Skip empty rows (e.g. skipped by the provider internal logic)
            if not meta:
                continue

            # Skip categories not in filter
            if self._filter_categories and not meta.categories.intersection(self._filter_categories):
                continue

            batch.append(meta)

            # Yield if batch is full
            if len(batch) >= self._batch_size:
                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    async def _sync_provider(self, db: AsyncDatabaseManager, provider: PaperMetadataProvider) -> None:
        source_name = provider.provides_source

        # Get lasy Sync
        async with db.session() as session:
            last_sync_time = await SyncStateRepository(session).get_last_synced_for_source(source_name)

        fetch_result: MetadataFetchResult = await provider.fetch_metadata(since=last_sync_time)

        new_last_sync: datetime.datetime | None = None

        with tqdm(total=fetch_result.total_progress, desc=source_name, unit="bytes") as pbar:
            # Iterate over our helper generator
            async for batch in self._stream_batches(fetch_result, pbar):
                async with db.session() as session:
                    paper_repo = PaperMetadataRepository(session)
                    await paper_repo.batch_upload(batch)
                for paper in batch:
                    if new_last_sync is None or paper.updated_at > new_last_sync:
                        new_last_sync = paper.updated_at

            # Update last sync
            async with db.session() as session:
                await SyncStateRepository(session).set_last_synced(source_name, new_last_sync)

    async def sync(self) -> None:
        async with new_database_manager(self._config.database) as db:
            for provider in self._metadata_providers:
                await self._sync_provider(db, provider)
