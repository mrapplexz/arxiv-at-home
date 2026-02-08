from pydantic import BaseModel

QDRANT_SPARSE_MODEL = "Qdrant/bm25"


class QdrantConfig(BaseModel):
    host: str
    grpc_port: int
