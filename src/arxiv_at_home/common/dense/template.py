from arxiv_at_home.common.dto import PaperMetadata


class DenseEncodingTemplate:
    def template_metadata(self, metadata: PaperMetadata) -> str:
        categories = ", ".join(metadata.categories)
        return f"""
Title: {metadata.title}
Categories: {categories}.

Abstract: {metadata.abstract}
        """.strip()
