from typing import Any

from fastapi import APIRouter, Depends

from arxiv_at_home.api.dependencies import AppState, get_app_state
from arxiv_at_home.api.dto import SearchRequest, SearchResponse
from arxiv_at_home.api.service.search import SearchService
from arxiv_at_home.common.database.repository import PaperMetadataRepository

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_papers(
    request: SearchRequest,
    state: AppState = Depends(get_app_state),  # noqa: B008
) -> SearchResponse:
    async with state.db_manager.session() as sess:
        service = SearchService(
            config=state.settings.search,
            state=state,
            paper_metadata_repository=PaperMetadataRepository(sess),
        )

        return await service.search(request)


@router.get("/health")
async def health_check() -> Any:
    return {"status": "ok"}
