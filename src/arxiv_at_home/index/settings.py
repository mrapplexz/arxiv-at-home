from pydantic_settings import BaseSettings, SettingsConfigDict

from arxiv_at_home.common.database.config import DatabaseConfig
from arxiv_at_home.common.dense.vectorizer import DenseVectorizationConfig
from arxiv_at_home.common.qdrant.config import QdrantConfig
from arxiv_at_home.index.component.dataset import PaperMetadataDatasetConfig


class IndexSettings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")

    database: DatabaseConfig
    qdrant: QdrantConfig
    dataset: PaperMetadataDatasetConfig
    dense_vectorizer: DenseVectorizationConfig
