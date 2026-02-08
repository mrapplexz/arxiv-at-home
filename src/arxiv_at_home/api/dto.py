from pydantic import BaseModel

from arxiv_at_home.common.dto import PaperMetadata


class SearchRequest(BaseModel):
    collection: str = "arxiv"
    query: str
    limit: int = 10


class ScoredPaper(BaseModel):
    score: float
    paper: PaperMetadata


class SearchStats(BaseModel):
    time_taken_seconds: float


class SearchResponse(BaseModel):
    results: list[ScoredPaper]
    stats: SearchStats
