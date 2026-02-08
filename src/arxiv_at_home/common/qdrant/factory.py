from qdrant_client import QdrantClient

from arxiv_at_home.common.qdrant.config import QdrantConfig


def create_qdrant(config: QdrantConfig) -> QdrantClient:
    return QdrantClient(host=config.host, grpc_port=config.grpc_port, prefer_grpc=True)
