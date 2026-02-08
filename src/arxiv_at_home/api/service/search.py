import time

import torch
from qdrant_client import models

from arxiv_at_home.api.component.reranker.template import RerankTemplate
from arxiv_at_home.api.dependencies import AppState
from arxiv_at_home.api.dto import ScoredPaper, SearchRequest, SearchResponse, SearchStats
from arxiv_at_home.api.settings import SearchConfig
from arxiv_at_home.common.database.repository import PaperMetadataRepository
from arxiv_at_home.common.dense.template import DenseEncodingTemplate
from arxiv_at_home.common.dense.vectorizer import VectorizerInputs
from arxiv_at_home.common.dto import PaperMetadata
from arxiv_at_home.common.qdrant.config import QDRANT_SPARSE_MODEL


class SearchService:
    def __init__(
        self, config: SearchConfig, state: AppState, paper_metadata_repository: PaperMetadataRepository
    ) -> None:
        self._config = config
        self._qdrant = state.qdrant
        self._vectorizer = state.dense_vectorizer
        self._dense_tokenizer = state.dense_tokenizer
        self._repo = paper_metadata_repository
        self._template_encode = DenseEncodingTemplate()

        self._template_rerank = RerankTemplate()
        self._reranker = state.reranker
        self._reranker_processor = state.reranker_processor
        self._reranker_tokenizer = state.reranker_tokenizer

    def _vectorize_query(self, text: str) -> list[float]:
        encoding = self._dense_tokenizer.encode(self._template_encode.template_query(text))

        inputs: VectorizerInputs = {
            "input_ids": torch.tensor([encoding.ids], dtype=torch.long),
            "attention_mask": torch.tensor([encoding.attention_mask], dtype=torch.long),
        }

        embeddings = self._vectorizer(inputs)
        return embeddings[0].tolist()

    def _rerank_documents(self, query: str, documents: list[PaperMetadata]) -> list[float]:
        templates = [self._template_rerank.format(query, doc) for doc in documents]
        inputs = self._reranker_processor.encode(templates)
        results = self._reranker(inputs)
        return results

    async def search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.perf_counter()

        dense_vector = self._vectorize_query(request.query)

        prefetch_limit = request.limit * self._config.prefetch_more_times
        search_result = await self._qdrant.query_points(
            collection_name=request.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="metadata/dense",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.Document(text=request.query, model=QDRANT_SPARSE_MODEL),
                    using="metadata/sparse",
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            limit=prefetch_limit,
            # We need payload to get valid IDs to fetch full metadata from Repo
            with_payload=["fully_qualified_name"],
        )

        prefetch_score_map = {point.payload["fully_qualified_name"]: point.score for point in search_result.points}

        papers = await self._repo.get_by_ids(list(prefetch_score_map.keys()))

        paper_ranks = self._rerank_documents(query=request.query, documents=papers)

        results = [ScoredPaper(paper=paper, score=rank) for paper, rank in zip(papers, paper_ranks, strict=True)]

        results.sort(key=lambda x: x.score, reverse=True)

        results = results[: request.limit]

        end_time = time.perf_counter()

        return SearchResponse(results=results, stats=SearchStats(time_taken_seconds=end_time - start_time))
