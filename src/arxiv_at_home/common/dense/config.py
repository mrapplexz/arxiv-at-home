from enum import StrEnum

from pydantic import BaseModel


class PoolingMode(StrEnum):
    last_token = "last_token"  # noqa: S105
    first_token = "first_token"  # noqa: S105


class DenseVectorizationConfig(BaseModel):
    device: str
    model: str
    pooling: PoolingMode
    query_template: str
    document_template: str
