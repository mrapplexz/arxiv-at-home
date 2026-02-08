from qdrant_client import AsyncQdrantClient

from arxiv_at_home.common.qdrant.config import QdrantConfig


def create_qdrant(config: QdrantConfig) -> AsyncQdrantClient:
    return AsyncQdrantClient(host=config.host, grpc_port=config.grpc_port, prefer_grpc=True)
