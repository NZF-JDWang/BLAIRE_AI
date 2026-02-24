"""Telegram stub."""


class TelegramInterface:
    """Placeholder Telegram interface contract."""

    def start(self) -> None:
        raise NotImplementedError("Telegram integration is not implemented in v0.1.")

    def stop(self) -> None:
        raise NotImplementedError("Telegram integration is not implemented in v0.1.")

    def send_message(self, chat_id: str, text: str) -> None:
        raise NotImplementedError("Telegram integration is not implemented in v0.1.")

