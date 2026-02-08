from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from starlette.types import Lifespan
from tokenizers import Tokenizer

from arxiv_at_home.api.component.reranker.model import (
    GenerativeReranker,
    RerankInputProcessor,
    create_rerank_processor,
    create_rerank_tokenizer,
    create_reranker,
)
from arxiv_at_home.api.settings import ApiSettings
from arxiv_at_home.common.database.manager import AsyncDatabaseManager, new_database_manager
from arxiv_at_home.common.dense.vectorizer import DenseVectorizer, create_dense_tokenizer, create_dense_vectorizer
from arxiv_at_home.common.qdrant.factory import create_qdrant


class AppState:
    settings: ApiSettings
    qdrant: AsyncQdrantClient
    db_manager: AsyncDatabaseManager

    dense_vectorizer: DenseVectorizer
    dense_tokenizer: Tokenizer

    reranker: GenerativeReranker
    reranker_processor: RerankInputProcessor
    reranker_tokenizer: Tokenizer


_state = AppState()


def lifespan_factory(config: ApiSettings) -> Lifespan:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> None:
        _state.settings = config
        _state.qdrant = create_qdrant(config.qdrant)

        async with new_database_manager(config.database) as db_manager:
            _state.db_manager = db_manager
            with (
                create_dense_vectorizer(config.dense_vectorizer) as dense_vectorizer,
                create_reranker(config.reranker) as reranker,
            ):
                _state.dense_vectorizer = dense_vectorizer
                _state.dense_tokenizer = create_dense_tokenizer(config.dense_vectorizer)

                _state.reranker = reranker
                _state.reranker_tokenizer = create_rerank_tokenizer(config.reranker)
                _state.reranker_processor = create_rerank_processor(config.reranker)

                yield

    return lifespan


def get_app_state() -> AppState:
    return _state
