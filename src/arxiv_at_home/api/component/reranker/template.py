from arxiv_at_home.api.component.reranker.config import RerankerConfig
from arxiv_at_home.common.dto import PaperMetadata

_QUERY_REPLACE = "$QUERY"
_DOC_REPLACE = "$DOCUMENT"


class RerankTemplate:
    def __init__(self, config: RerankerConfig) -> None:
        self._template = config.template
        if _QUERY_REPLACE not in config.template:
            raise ValueError(f"Invalid template - it should contain '{_QUERY_REPLACE}'")
        if _DOC_REPLACE not in config.template:
            raise ValueError(f"Invalid template - it should contain '{_DOC_REPLACE}'")

    def format(self, query: str, metadata: PaperMetadata) -> str:
        doc = f"""
{metadata.title}
Categories: {metadata.categories}
Abstract: {metadata.abstract}
        """.strip()

        return self._template.replace(_QUERY_REPLACE, query).replace(_DOC_REPLACE, doc)
