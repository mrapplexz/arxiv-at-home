import math
import time

import torch
from qdrant_client import models

from arxiv_at_home.api.dependencies import AppState
from arxiv_at_home.api.dto import ScoredPaper, SearchRequest, SearchResponse, SearchStats
from arxiv_at_home.api.settings import SearchConfig
from arxiv_at_home.common.database.repository import PaperMetadataRepository
from arxiv_at_home.common.dense.vectorizer import VectorizerInputs
from arxiv_at_home.common.dto import PaperMetadata
from arxiv_at_home.common.qdrant.config import QDRANT_SPARSE_MODEL


class SearchService:
    def __init__(
        self, config: SearchConfig, state: AppState, paper_metadata_repository: PaperMetadataRepository
    ) -> None:
        self._config = config
        self._qdrant = state.qdrant
        self._dense_vectorizer = state.dense_vectorizer
        self._dense_tokenizer = state.dense_tokenizer
        self._dense_template = state.dense_template

        self._repo = paper_metadata_repository

        self._reranker = state.reranker
        self._reranker_processor = state.reranker_processor
        self._reranker_template = state.reranker_template

        self._citation_provider = state.citation_provider

    def _vectorize_query(self, text: str) -> list[float]:
        encoding = self._dense_tokenizer.encode(self._dense_template.template_query(text))

        inputs: VectorizerInputs = {
            "input_ids": torch.tensor([encoding.ids], dtype=torch.long),
            "attention_mask": torch.tensor([encoding.attention_mask], dtype=torch.long),
        }

        embeddings = self._dense_vectorizer(inputs)
        return embeddings[0].tolist()

    async def _retrieve_candidates(
        self, collection_name: str, query_text: str, query_vector: list[float], limit: int
    ) -> list[models.ScoredPoint]:
        prefetch_limit = limit * self._config.prefetch_more_times

        search_result = await self._qdrant.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="metadata/dense",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.Document(text=query_text, model=QDRANT_SPARSE_MODEL),
                    using="metadata/sparse",
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            limit=prefetch_limit,
            with_payload=["fully_qualified_name"],
        )
        return search_result.points

    async def _hydrate_documents(self, points: list[models.ScoredPoint]) -> list[PaperMetadata]:
        if not points:
            return []

        fqns = [point.payload["fully_qualified_name"] for point in points]
        return await self._repo.get_by_ids(fqns)

    async def _fetch_citation_metadata(self, documents: list[PaperMetadata]) -> dict[str, int | None]:
        if not documents:
            return {}

        counts = await self._citation_provider.get_citation_count_batch([doc.fully_qualified_name for doc in documents])

        return counts

    def _rerank_documents(self, query: str, documents: list[PaperMetadata]) -> list[float]:
        if not documents:
            return []

        templates = [self._reranker_template.format(query, doc) for doc in documents]
        inputs = self._reranker_processor.encode(templates)
        results = self._reranker(inputs)
        return results

    def _calculate_total_score(self, semantic_score: float, citation_count: int | None) -> float:
        if citation_count is None:
            citation_count = 0

        citation_factor = math.log10(citation_count + 1)
        citation_boost = 1 + (self._config.citation_boost_weight * citation_factor)

        return semantic_score * citation_boost

    def _apply_ranking_and_sort(
        self, query: str, documents: list[PaperMetadata], citation_map: dict[str, int | None], limit: int
    ) -> list[ScoredPaper]:
        if not documents:
            return []

        # 1. Get Semantic Ranks (Cross-Encoder)
        semantic_scores = self._rerank_documents(query=query, documents=documents)

        scored_papers = []
        for paper, semantic_score in zip(documents, semantic_scores, strict=True):
            # 2. Lookup Citations
            citations = citation_map[paper.fully_qualified_name]

            # 3. Calculate Final Score
            final_score = self._calculate_total_score(semantic_score, citations)

            scored_papers.append(ScoredPaper(paper=paper, citations=citations, score=final_score))

        # 4. Sort
        scored_papers.sort(key=lambda x: x.score, reverse=True)

        return scored_papers[:limit]

    async def search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.perf_counter()

        # 1. Prepare Query
        dense_vector = self._vectorize_query(request.query)

        # 2. Retrieve Candidates (Qdrant)
        points = await self._retrieve_candidates(
            collection_name=request.collection, query_text=request.query, query_vector=dense_vector, limit=request.limit
        )

        # 3. Hydrate Data (Database)
        papers = await self._hydrate_documents(points)

        # 4. Fetch Citation Metadata (it may be some external provider)
        citation_map = await self._fetch_citation_metadata(papers)

        # 5. Rerank and Sort (Cross-Encoder + Citation Boost)
        results = self._apply_ranking_and_sort(
            query=request.query, documents=papers, citation_map=citation_map, limit=request.limit
        )

        end_time = time.perf_counter()

        return SearchResponse(results=results, stats=SearchStats(time_taken_seconds=end_time - start_time))
