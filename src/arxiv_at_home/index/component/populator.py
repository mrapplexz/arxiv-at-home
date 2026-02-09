import uuid
from typing import Any

import torch
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, Document, Modifier, SparseIndexParams, SparseVectorParams, VectorParams

from arxiv_at_home.common.dto import PaperMetadata
from arxiv_at_home.common.qdrant.config import QDRANT_SPARSE_MODEL
from arxiv_at_home.index.component.batch_type import PaperMetadataDatasetSparseBatch


def metadata_to_uuid(metadata: PaperMetadata) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, metadata.fully_qualified_name)


class CollectionPopulator:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def _ensure_collection(self, source: str, dense_dim: int) -> None:
        if not await self._client.collection_exists(source):
            await self._client.create_collection(
                collection_name=source,
                sparse_vectors_config={
                    "title/sparse": SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF),
                    "abstract/sparse": SparseVectorParams(index=SparseIndexParams(), modifier=Modifier.IDF),
                },
                vectors_config={"metadata/dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
            )

    def _vectors_from_meta(self, sparse_title: str, sparse_abstract: str, dense_vector: torch.Tensor) -> dict[str, Any]:
        return {
            "title/sparse": Document(text=sparse_title, model=QDRANT_SPARSE_MODEL),
            "abstract/sparse": Document(text=sparse_abstract, model=QDRANT_SPARSE_MODEL),
            "metadata/dense": dense_vector.tolist(),
        }

    def _payload_from_meta(self, meta: PaperMetadata) -> dict[str, Any]:
        return {
            "title": meta.title,  # for debugging purposes only
            "n_versions": len(meta.versions),
            "journal_ref": meta.journal_ref,
            "fully_qualified_name": meta.fully_qualified_name,
            "updated_at": meta.updated_at,
            "categories": list(meta.categories),
        }

    async def upsert_metadata(
            self, metadata: list[PaperMetadata], sparse_texts: PaperMetadataDatasetSparseBatch,
            dense_vectors: list[torch.Tensor]
    ) -> None:
        if not metadata:
            return

        collection_name = metadata[0].source
        dense_dim = dense_vectors[0].shape[0]

        await self._ensure_collection(collection_name, dense_dim)
        self._client.upload_collection(
            collection_name=collection_name,
            vectors=[
                self._vectors_from_meta(title, abstract, dense_vec)
                for title, abstract, dense_vec in zip(sparse_texts["title"], sparse_texts["abstract"], dense_vectors, strict=True)
            ],
            payload=[self._payload_from_meta(meta) for meta in metadata],
            ids=[metadata_to_uuid(meta) for meta in metadata],  # deterministic uuidv5 ids
        )
