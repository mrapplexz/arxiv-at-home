import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence

import torch
from d9d.dataset import PaddingSide1D, pad_stack_1d
from pydantic import BaseModel
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, IterableDataset

from arxiv_at_home.common.database.config import DatabaseConfig
from arxiv_at_home.common.database.manager import new_database_manager
from arxiv_at_home.common.database.repository import PaperMetadataRepository
from arxiv_at_home.common.dense.template import DenseEncodingTemplate
from arxiv_at_home.common.dto import PaperMetadata
from arxiv_at_home.index.component.batch_type import (
    PaperMetadataDatasetBatch,
    PaperMetadataDatasetSample,
)


class PaperMetadataDatasetConfig(BaseModel):
    db_chunk_size: int
    batch_size: int
    num_workers: int


class PaperMetadataDataset(IterableDataset):
    def __init__(
        self,
        db_config: DatabaseConfig,
        db_chunk_size: int,
        dense_tokenizer: Tokenizer,
        dense_template: DenseEncodingTemplate,
        tokenization_prefix: str,
    ) -> None:
        self._db_config = db_config
        self._db_chunk_size = db_chunk_size
        self._tokenizer = dense_tokenizer
        self._tokenization_prefix = tokenization_prefix
        self._template = dense_template

    def __iter__(self) -> Iterator[PaperMetadataDatasetSample]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async_chunk_gen = self._async_chunk_generator()

        try:
            while True:
                try:
                    chunk = loop.run_until_complete(async_chunk_gen.__anext__())
                except StopAsyncIteration:
                    break

                for meta in chunk:
                    yield self._encode_metadata(meta)
        finally:
            if loop.is_running():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            loop.close()

    def _encode_metadata(self, meta: PaperMetadata) -> PaperMetadataDatasetSample:
        template = self._template.template_metadata(meta)
        encoding = self._tokenizer.encode(template)
        return {
            "id": meta.fully_qualified_name,
            "metadata": {
                "dense": {
                    "input_ids": torch.tensor(encoding.ids, dtype=torch.long),
                    "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
                },
                "sparse": {
                    "text": template,
                },
                "json": meta.model_dump_json(),
            },
        }

    async def _async_chunk_generator(self) -> AsyncIterator[Sequence[PaperMetadata]]:
        async with new_database_manager(self._db_config) as db_mgr:
            while True:
                async with db_mgr.session() as session:
                    repo = PaperMetadataRepository(session)
                    papers = await repo.fetch_and_lock_next_batch_for_indexing(batch_size=self._db_chunk_size)

                if not papers:
                    break

                yield papers


class PaperMetadataCollator:
    def __call__(self, batch: Sequence[PaperMetadataDatasetSample]) -> PaperMetadataDatasetBatch:
        return {
            "id": [x["id"] for x in batch],
            "metadata": {
                "dense": {
                    "input_ids": pad_stack_1d(
                        [x["metadata"]["dense"]["input_ids"] for x in batch],
                        pad_value=0,
                        padding_side=PaddingSide1D.right,
                    ),
                    "attention_mask": pad_stack_1d(
                        [x["metadata"]["dense"]["attention_mask"] for x in batch],
                        pad_value=0,
                        padding_side=PaddingSide1D.right,
                    ),
                },
                "sparse": {
                    "text": [x["metadata"]["sparse"]["text"] for x in batch],
                },
                "json": [x["metadata"]["json"] for x in batch],
            },
        }


def create_paper_metadata_data_loader(
    db_config: DatabaseConfig,
    dense_tokenizer: Tokenizer,
    dense_template: DenseEncodingTemplate,
    config: PaperMetadataDatasetConfig,
) -> DataLoader:
    dataset = PaperMetadataDataset(
        db_config=db_config,
        db_chunk_size=config.db_chunk_size,
        dense_tokenizer=dense_tokenizer,
        tokenization_prefix="",
        dense_template=dense_template,
    )

    collator = PaperMetadataCollator()

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collator,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )

    return loader
