from arxiv_at_home.common.dense.config import DenseVectorizationConfig
from arxiv_at_home.common.dto import PaperMetadata

_QUERY_REPLACE = "$QUERY"
_DOCUMENT_REPLACE = "$DOCUMENT"


class DenseEncodingTemplate:
    def __init__(self, config: DenseVectorizationConfig) -> None:
        self._query_template = config.query_template
        if _QUERY_REPLACE not in config.query_template:
            raise ValueError(f"Query template should contain '{_QUERY_REPLACE}'")

        self._document_template = config.document_template
        if _DOCUMENT_REPLACE not in config.document_template:
            raise ValueError(f"Document template should contain '{_DOCUMENT_REPLACE}'")

    def template_metadata(self, metadata: PaperMetadata) -> str:
        categories = ", ".join(metadata.categories)
        doc = f"""
Title: {metadata.title}
Categories: {categories}.

Abstract: {metadata.abstract}
        """.strip()

        return self._document_template.replace(_DOCUMENT_REPLACE, doc)

    def template_query(self, query: str) -> str:
        return self._query_template.replace(_QUERY_REPLACE, query)
