import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger

from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection
from pgvector.psycopg import register_vector_async

from .settings import settings

# Construct the DB connection string for psycopg
DATABASE_URL = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

# We initialize the pool globally but lazily
_pool: Optional[AsyncConnectionPool] = None

async def init_db_pool():
    global _pool
    if _pool is None:
        logger.info(f"Initializing AsyncConnectionPool to {settings.DB_HOST}:{settings.DB_PORT}...")
        
        async def configure_connection(conn: AsyncConnection):
            # Crucial: Register pgvector dynamically whenever a new connection is spawned in the pool
            await register_vector_async(conn)

        _pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": False}, # We handle transactions explicitly via conn.commit()
            configure=configure_connection
        )
        await _pool.open()
        logger.info("AsyncConnectionPool initialized and pgvector registered.")

async def close_db_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        logger.info("AsyncConnectionPool closed.")

@asynccontextmanager
async def get_db_connection():
    """
    Context Manager to get an async connection from the global pool.
    Yields a transaction-ready AsyncConnection with pgvector capabilities.
    """
    global _pool
    if _pool is None:
        await init_db_pool()
        
    async with _pool.connection() as conn:
        yield conn
