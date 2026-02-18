from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings

ModelClass = Literal["general", "vision", "embedding", "code"]


@dataclass(frozen=True)
class ModelSelection:
    model_class: ModelClass
    model_name: str
    reason: str
    fallback_used: bool
    rejected_candidates: list[str]


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

    def is_model_allowed(self, model_class: ModelClass, model_name: str) -> bool:
        return model_name in self._allowlist.get(model_class, set())

    def select_model(
        self,
        model_class: ModelClass,
        request_override: str | None = None,
        preference_override: str | None = None,
    ) -> ModelSelection:
        rejected_candidates: list[str] = []
        allowlist = self._allowlist.get(model_class, set())

        if request_override:
            if request_override in allowlist:
                return ModelSelection(
                    model_class=model_class,
                    model_name=request_override,
                    reason="session_override",
                    fallback_used=False,
                    rejected_candidates=rejected_candidates,
                )
            rejected_candidates.append(f"session_override_disallowed:{request_override}")

        if preference_override:
            if preference_override in allowlist:
                return ModelSelection(
                    model_class=model_class,
                    model_name=preference_override,
                    reason="user_preference_override",
                    fallback_used=bool(rejected_candidates),
                    rejected_candidates=rejected_candidates,
                )
            rejected_candidates.append(f"preference_override_disallowed:{preference_override}")

        defaults: dict[ModelClass, str | None] = {
            "general": self._settings.model_general_default,
            "vision": self._settings.model_vision_default,
            "embedding": self._settings.model_embedding_default,
            "code": self._settings.model_code_default,
        }
        default_model = defaults.get(model_class)
        if default_model and default_model in allowlist:
            return ModelSelection(
                model_class=model_class,
                model_name=default_model,
                reason="class_default",
                fallback_used=bool(rejected_candidates),
                rejected_candidates=rejected_candidates,
            )
        if default_model:
            rejected_candidates.append(f"class_default_disallowed:{default_model}")

        if allowlist:
            fallback_model = sorted(allowlist)[0]
            return ModelSelection(
                model_class=model_class,
                model_name=fallback_model,
                reason="allowlist_fallback",
                fallback_used=True,
                rejected_candidates=rejected_candidates,
            )

        raise ValueError(f"No allowed model configured for class '{model_class}'")

    def get_allowlist(self) -> dict[ModelClass, list[str]]:
        return {
            key: sorted(value)
            for key, value in self._allowlist.items()
            if value
        }
