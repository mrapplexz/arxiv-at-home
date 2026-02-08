from arxiv_at_home.common.dto import PaperMetadata


class RerankTemplate:
    def format(self, query: str, metadata: PaperMetadata) -> str:
        instruction = "Given an academic database search query, retrieve relevant articles that satisfy the query"

        doc = f"""
{metadata.title}
Categories: {metadata.categories}
Abstract: {metadata.abstract}
        """.strip()

        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
