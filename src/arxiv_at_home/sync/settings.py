from pydantic_settings import BaseSettings, SettingsConfigDict

from arxiv_at_home.common.database.config import DatabaseConfig
from arxiv_at_home.sync.component.metadata_provider.factory import AnyPaperProviderConfig


class SyncSettings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")

    metadata_providers: list[AnyPaperProviderConfig]
    filter_categories: set[str] | None
    batch_size: int
    database: DatabaseConfig
