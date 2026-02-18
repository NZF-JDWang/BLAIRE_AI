from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings

ModelClass = Literal["general", "vision", "embedding", "code"]


@dataclass(frozen=True)
class ModelSelection:
    model_class: ModelClass
    model_name: str
    reason: str


class ModelRouter:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._allowlist: dict[ModelClass, set[str]] = {
            "general": {
                settings.model_general_default,
                "llama3.2:3b",
                "qwen2.5:7b-instruct",
            },
            "vision": {
                settings.model_vision_default,
                "llava:7b",
                "llava:13b",
            },
            "embedding": {
                settings.model_embedding_default,
                "nomic-embed-text:v1.5",
            },
            "code": {
                model
                for model in [settings.model_code_default, "qwen2.5-coder:7b"]
                if model
            },
        }

    def select_model(self, model_class: ModelClass, override: str | None = None) -> ModelSelection:
        if override:
            if override not in self._allowlist.get(model_class, set()):
                raise ValueError(f"Requested model '{override}' is not allowed for class '{model_class}'")
            return ModelSelection(model_class=model_class, model_name=override, reason="user_override")

        defaults: dict[ModelClass, str | None] = {
            "general": self._settings.model_general_default,
            "vision": self._settings.model_vision_default,
            "embedding": self._settings.model_embedding_default,
            "code": self._settings.model_code_default,
        }
        default_model = defaults.get(model_class)
        if not default_model:
            raise ValueError(f"No default model configured for class '{model_class}'")
        return ModelSelection(model_class=model_class, model_name=default_model, reason="class_default")

    def get_allowlist(self) -> dict[ModelClass, list[str]]:
        return {
            key: sorted(value)
            for key, value in self._allowlist.items()
            if value
        }
