from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from arxiv_at_home.common.database.config import DatabaseConfig


class AsyncDatabaseManager:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        session: AsyncSession = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def new_database_manager(config: DatabaseConfig) -> AsyncGenerator[AsyncDatabaseManager, None]:
    engine: AsyncEngine = create_async_engine(
        config.connection_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

    session_factory = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=AsyncSession)

    yield AsyncDatabaseManager(session_factory=session_factory)

    await engine.dispose()
