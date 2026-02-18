import psycopg


class MetadataStoreService:
    def __init__(self, database_url: str):
        self._database_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    async def init_schema(self) -> None:
        # Stores file-level ingest progress for delta/index maintenance.
        query = """
        CREATE TABLE IF NOT EXISTS ingestion_file_state (
            source_path TEXT PRIMARY KEY,
            source_kind TEXT NOT NULL,
            last_modified TIMESTAMPTZ,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status TEXT NOT NULL DEFAULT 'indexed',
            checksum TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        async with await psycopg.AsyncConnection.connect(self._database_url, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
