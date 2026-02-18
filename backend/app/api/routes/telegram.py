from fastapi import APIRouter, Header, HTTPException, Request

from app.core.config import get_settings
from app.core.rate_limit import RateLimitRule, rate_limiter
from app.services.agent_swarm import AgentSwarmService
from app.services.search_service import SearchService
from app.services.telegram_service import TelegramService, TelegramServiceError

router = APIRouter(tags=["telegram"])


@router.post("/telegram/webhook")
async def telegram_webhook(
    update: dict,
    raw_request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None, alias="X-Telegram-Bot-Api-Secret-Token"),
) -> dict[str, str]:
    settings = get_settings()
    token = settings.telegram_bot_token.get_secret_value() if settings.telegram_bot_token else ""
    if not token:
        raise HTTPException(status_code=503, detail="Telegram bot token is not configured")
    webhook_secret = (
        settings.telegram_webhook_secret_token.get_secret_value()
        if settings.telegram_webhook_secret_token
        else ""
    )
    if webhook_secret and x_telegram_bot_api_secret_token != webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret token")

    rate_limiter.check(
        f"telegram:{raw_request.client.host if raw_request.client else 'unknown'}",
        RateLimitRule(30, 60),
    )

    parsed = TelegramService.parse_incoming_text(update)
    if not parsed:
        return {"status": "ignored"}

    chat_id, text = parsed
    query = text.removeprefix("/research").strip() if text.startswith("/research") else text.strip()
    if not query:
        return {"status": "ignored"}

    swarm = AgentSwarmService(SearchService(settings))
    result = await swarm.run_research(query=query, search_mode=settings.search_mode_default)
    outbound = f"BLAIRE summary:\n{result.supervisor_summary}"

    telegram = TelegramService(token)
    try:
        await telegram.send_message(chat_id=chat_id, text=outbound[:3900])
    except TelegramServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"status": "ok"}
