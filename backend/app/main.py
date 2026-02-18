from contextlib import asynccontextmanager

from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes.agents import router as agents_router
from app.api.routes.approvals import router as approvals_router
from app.api.routes.chat import router as chat_router
from app.api.routes.dependencies import router as dependencies_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.mcp import router as mcp_router
from app.api.routes.preferences import router as preferences_router
from app.api.routes.runtime_options import router as runtime_options_router
from app.api.routes.search import router as search_router
from app.api.routes.tools import router as tools_router
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.request_context import RequestContextMiddleware
from app.core.security_headers import SecurityHeadersMiddleware
from app.rag.qdrant_bootstrap import QdrantBootstrapService
from app.services.approval_service import ApprovalService
from app.services.dependency_checks import collect_dependency_status
from app.services.preferences_service import PreferencesService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(component="startup")
    logger.info(
        "app_starting",
        environment=settings.app_env,
        host=settings.api_host,
        port=settings.api_port,
    )
    try:
        await ApprovalService(settings.database_url.get_secret_value()).init_schema()
        logger.info("approval_schema_ready")
    except Exception:
        logger.exception("approval_schema_init_failed")
        raise
    try:
        await PreferencesService(settings.database_url.get_secret_value()).init_schema()
        logger.info("preferences_schema_ready")
    except Exception:
        logger.exception("preferences_schema_init_failed")
        raise
    try:
        await QdrantBootstrapService(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            embedding_dim=settings.qdrant_embedding_dim,
        ).ensure_collection()
        logger.info(
            "qdrant_collection_ready",
            collection=settings.qdrant_collection_name,
            embedding_dim=settings.qdrant_embedding_dim,
        )
    except Exception:
        logger.exception("qdrant_collection_init_failed")
        raise
    try:
        status = await collect_dependency_status(settings)
        logger.info(
            "dependency_status",
            dependencies=[item.model_dump() for item in status.dependencies],
        )
    except Exception:
        logger.exception("dependency_check_failed")
    yield
    logger.info("app_stopping")


def create_app() -> FastAPI:
    settings = get_settings()
    docs_url = "/docs" if settings.api_docs_enabled else None
    redoc_url = "/redoc" if settings.api_docs_enabled else None

    app = FastAPI(
        title="BLAIRE Backend",
        version="0.1.0",
        default_response_class=ORJSONResponse,
        docs_url=docs_url,
        redoc_url=redoc_url,
        lifespan=lifespan,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts_list())
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(runtime_options_router)
    app.include_router(approvals_router)
    app.include_router(tools_router)
    app.include_router(search_router)
    app.include_router(knowledge_router)
    app.include_router(agents_router)
    app.include_router(dependencies_router)
    app.include_router(mcp_router)
    app.include_router(preferences_router)
    return app


app = create_app()
