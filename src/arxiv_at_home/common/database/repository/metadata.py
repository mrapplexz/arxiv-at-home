import datetime as dt
from collections.abc import Sequence

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as sapg
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from arxiv_at_home.common.database.repository.base import ArxivDeclarativeBase
from arxiv_at_home.common.dto import PaperMetadata


class PaperMetadataStored(ArxivDeclarativeBase):
    __tablename__ = "paper_records"

    fully_qualified_name: Mapped[str] = mapped_column(sa.String(255), primary_key=True)
    paper_metadata: Mapped[dict] = mapped_column(sapg.JSONB)
    abstract_len: Mapped[int] = mapped_column(sa.Integer)

    synced_at: Mapped[dt.datetime] = mapped_column(sa.DateTime(timezone=True))

    indexed_at: Mapped[dt.datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    indexing_reserved_at: Mapped[dt.datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)

    # Index for: SELECT ... WHERE synced_at >= ... ORDER BY abstract_len, synced_at
    __table_args__ = (
        # Partial index for the queue: fast retrieval of unindexed, unreserved items
        sa.Index(
            "idx_papers_queue", "abstract_len", postgresql_where=(indexed_at.is_(None) & indexing_reserved_at.is_(None))
        ),
    )


class PaperMetadataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def batch_upload(self, papers: list[PaperMetadata]) -> int:
        if not papers:
            return 0

        now = dt.datetime.now(dt.UTC)
        values = []

        for p in papers:
            values.append(
                {
                    "fully_qualified_name": p.fully_qualified_name,
                    "synced_at": now,
                    "paper_metadata": p.model_dump(mode="json"),
                    "abstract_len": len(p.abstract),
                }
            )

        stmt = sapg.insert(PaperMetadataStored).values(values)

        stmt = stmt.on_conflict_do_update(
            index_elements=[PaperMetadataStored.fully_qualified_name],
            set_={
                "synced_at": stmt.excluded.synced_at,
                "paper_metadata": stmt.excluded.paper_metadata,
                "abstract_len": stmt.excluded.abstract_len,
                "indexed_at": None,
                "indexing_reserved_at": None,
            },
        )

        result = await self._session.execute(stmt)

        return result.rowcount

    async def fetch_and_lock_next_batch_for_indexing(self, batch_size: int) -> Sequence[PaperMetadata]:
        now = dt.datetime.now(dt.UTC)

        subquery = (
            sa.select(PaperMetadataStored.fully_qualified_name)
            .where(PaperMetadataStored.indexed_at.is_(None))
            .where(PaperMetadataStored.indexing_reserved_at.is_(None))
            .order_by(PaperMetadataStored.abstract_len.asc())
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        ).cte("locked_rows")

        stmt = (
            sa.update(PaperMetadataStored)
            .values(indexing_reserved_at=now)
            .where(PaperMetadataStored.fully_qualified_name == subquery.c.fully_qualified_name)
            .returning(PaperMetadataStored.paper_metadata)
        )

        result = await self._session.execute(stmt)

        results = result.scalars().all()
        return [PaperMetadata.model_validate(x) for x in results]

    async def mark_batch_as_indexed(self, metadata: list[PaperMetadata]) -> int:
        if not metadata:
            return 0

        now = dt.datetime.now(dt.UTC)
        fqns = [meta.fully_qualified_name for meta in metadata]

        stmt = (
            sa.update(PaperMetadataStored)
            .where(PaperMetadataStored.fully_qualified_name.in_(fqns))
            .values(
                indexed_at=now,
                indexing_reserved_at=None,  # Clear reservation
            )
        )

        result = await self._session.execute(stmt)
        return result.rowcount

    async def clear_indexing_reservations(self) -> int:
        stmt = (
            sa.update(PaperMetadataStored)
            .where(PaperMetadataStored.indexing_reserved_at.is_not(None))
            .values(indexing_reserved_at=None)
        )
        result = await self._session.execute(stmt)
        return result.rowcount

    async def estimate_count_for_indexing(self) -> int:
        stmt = (
            sa.select(sa.func.count())
            .select_from(PaperMetadataStored)
            .where(PaperMetadataStored.indexed_at.is_(None))
            .where(PaperMetadataStored.indexing_reserved_at.is_(None))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def get_by_ids(self, fully_qualified_names: list[str]) -> list[PaperMetadata]:
        if not fully_qualified_names:
            return []

        stmt = sa.select(PaperMetadataStored).where(PaperMetadataStored.fully_qualified_name.in_(fully_qualified_names))

        result = await self._session.execute(stmt)
        stored_papers = result.scalars().all()

        paper_map = {p.fully_qualified_name: p for p in stored_papers}

        ordered_papers = []

        for fqn in fully_qualified_names:
            if fqn in paper_map:
                obj = paper_map[fqn]
                ordered_papers.append(PaperMetadata.model_validate(obj.paper_metadata))

        return ordered_papers
