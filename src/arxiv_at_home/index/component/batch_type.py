from typing import TypedDict

from arxiv_at_home.common.dense.vectorizer import VectorizerInputs


class PaperMetadataDatasetSparseBatch(TypedDict):
    text: list[str]


class PaperMetadataDatasetSparseSample(TypedDict):
    text: str


class PaperMetadataDatasetMetadataBatch(TypedDict):
    dense: VectorizerInputs
    sparse: PaperMetadataDatasetSparseBatch
    json: list[str]


class PaperMetadataDatasetMetadataSample(TypedDict):
    dense: VectorizerInputs
    sparse: PaperMetadataDatasetSparseSample
    json: str


class PaperMetadataDatasetBatch(TypedDict):
    id: list[str]
    metadata: PaperMetadataDatasetMetadataBatch


class PaperMetadataDatasetSample(TypedDict):
    id: str
    metadata: PaperMetadataDatasetMetadataSample
