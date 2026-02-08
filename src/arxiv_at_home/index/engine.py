from tqdm import tqdm

from arxiv_at_home.common.database.manager import AsyncDatabaseManager, new_database_manager
from arxiv_at_home.common.database.repository import PaperMetadataRepository
from arxiv_at_home.common.dense.vectorizer import DenseVectorizer, create_dense_tokenizer, create_dense_vectorizer
from arxiv_at_home.common.dto import PaperMetadata
from arxiv_at_home.common.qdrant.factory import create_qdrant
from arxiv_at_home.index.component.batch_type import PaperMetadataDatasetBatch
from arxiv_at_home.index.component.dataset import create_paper_metadata_data_loader
from arxiv_at_home.index.component.populator import CollectionPopulator
from arxiv_at_home.index.settings import IndexSettings


class IndexEngine:
    def __init__(self, config: IndexSettings) -> None:
        self._config = config

    async def _process_batch(
        self,
        batch: PaperMetadataDatasetBatch,
        vectorizer: DenseVectorizer,
        populator: CollectionPopulator,
        db_manager: AsyncDatabaseManager,
    ) -> None:
        metadata_inputs = batch["metadata"]
        metadata = [PaperMetadata.model_validate_json(x) for x in metadata_inputs["json"]]

        dense_embeddings = vectorizer(metadata_inputs["dense"])

        await populator.upsert_metadata(
            metadata=metadata, dense_vectors=dense_embeddings, sparse_texts=metadata_inputs["sparse"]["text"]
        )

        # do not catch errors - we will release dangling reservations in index() beginning
        async with db_manager.session() as session:
            repo = PaperMetadataRepository(session)
            await repo.mark_batch_as_indexed(metadata)

    async def index(self) -> None:
        tokenizer = create_dense_tokenizer(self._config.dense_vectorizer)
        populator = CollectionPopulator(create_qdrant(self._config.qdrant))
        with create_dense_vectorizer(self._config.dense_vectorizer) as vectorizer:
            async with new_database_manager(self._config.database) as db_manager:
                async with db_manager.session() as sess:
                    repo = PaperMetadataRepository(sess)
                    await repo.clear_indexing_reservations()
                    estimated_count = await repo.estimate_count_for_indexing()
                data_loader = create_paper_metadata_data_loader(
                    db_config=self._config.database, dense_tokenizer=tokenizer, config=self._config.dataset
                )
                with tqdm(desc="Indexing", total=estimated_count) as pbar:
                    for batch in data_loader:
                        await self._process_batch(
                            batch, vectorizer=vectorizer, populator=populator, db_manager=db_manager
                        )
                        pbar.update(len(batch["id"]))
