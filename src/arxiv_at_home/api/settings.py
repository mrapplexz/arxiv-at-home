from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from arxiv_at_home.api.component.citation_provider.factory import AnyCitationProviderConfig
from arxiv_at_home.api.component.reranker.model import RerankerConfig
from arxiv_at_home.common.database.config import DatabaseConfig
from arxiv_at_home.common.dense.vectorizer import DenseVectorizationConfig
from arxiv_at_home.common.qdrant.config import QdrantConfig


class ServingConfig(BaseModel):
    host: str
    port: int


class SearchConfig(BaseModel):
    prefetch_more_times: int
    citation_boost_weight: float


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")

    serving: ServingConfig
    database: DatabaseConfig
    qdrant: QdrantConfig
    dense_vectorizer: DenseVectorizationConfig
    reranker: RerankerConfig
    search: SearchConfig
    citation_provider: AnyCitationProviderConfig
