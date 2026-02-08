import abc


class CitationProvider(abc.ABC):
    @abc.abstractmethod
    async def get_citation_count_batch(self, paper_ids: list[str]) -> dict[str, int | None]:
        pass
