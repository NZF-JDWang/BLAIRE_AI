from app.core.config import Settings
from app.rag.qdrant_bootstrap import QdrantBootstrapService
from app.services.approval_service import ApprovalService
from app.services.metadata_store import MetadataStoreService
from app.services.preferences_service import PreferencesService


class InitService:
    def __init__(self, settings: Settings):
        self._settings = settings
        db_url = settings.database_url.get_secret_value()
        self._approval = ApprovalService(db_url)
        self._preferences = PreferencesService(db_url)
        self._metadata = MetadataStoreService(db_url)
        self._qdrant = QdrantBootstrapService(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            embedding_dim=settings.qdrant_embedding_dim,
        )

    async def run(self) -> dict[str, bool]:
        await self._approval.init_schema()
        await self._preferences.init_schema()
        await self._metadata.init_schema()
        await self._qdrant.ensure_collection()
        return {
            "approval_schema_ready": True,
            "preferences_schema_ready": True,
            "metadata_schema_ready": True,
            "qdrant_collection_ready": True,
        }
