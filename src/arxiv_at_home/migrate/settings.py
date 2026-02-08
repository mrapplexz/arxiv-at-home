from pydantic_settings import BaseSettings, SettingsConfigDict

from arxiv_at_home.common.database.config import DatabaseConfig


class MigrateSettings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")

    database: DatabaseConfig
