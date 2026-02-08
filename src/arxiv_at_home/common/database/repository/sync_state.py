import datetime as dt

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as sapg
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from arxiv_at_home.common.database.repository.base import ArxivDeclarativeBase


class SyncStateStored(ArxivDeclarativeBase):
    __tablename__ = "sync_state"

    source: Mapped[str] = mapped_column(sa.String(255), primary_key=True)

    last_synced_at: Mapped[dt.datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)


class SyncStateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_last_synced_for_source(self, source: str) -> dt.datetime | None:
        stmt = sa.select(SyncStateStored).where(SyncStateStored.source == source)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        return row.last_synced_at if row else None

    async def set_last_synced(self, source: str, timestamp: dt.datetime | None) -> None:
        stmt = sapg.insert(SyncStateStored).values(source=source, last_synced_at=timestamp)

        stmt = stmt.on_conflict_do_update(
            index_elements=[SyncStateStored.source], set_={"last_synced_at": stmt.excluded.last_synced_at}
        )

        await self._session.execute(stmt)
