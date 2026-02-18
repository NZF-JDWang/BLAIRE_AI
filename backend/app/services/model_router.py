from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.services.ollama_client import OllamaModelCatalog

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
        self._catalog = OllamaModelCatalog(settings.ollama_base_url)

    def _fallback_allowlist(self) -> dict[ModelClass, set[str]]:
        return {
            "general": {
                self._settings.model_general_default,
                "llama3.2:3b",
                "qwen2.5:7b-instruct",
            },
            "vision": {
                self._settings.model_vision_default,
                "llava:7b",
                "llava:13b",
            },
            "embedding": {
                self._settings.model_embedding_default,
                "nomic-embed-text:v1.5",
            },
            "code": {
                model
                for model in [self._settings.model_code_default, "qwen2.5-coder:7b"]
                if model
            },
        }

    @staticmethod
    def _is_embedding_model(name: str, model_meta: dict) -> bool:
        lowered = name.lower()
        if lowered.startswith("nomic-embed-text"):
            return True
        details = model_meta.get("details", {}) if isinstance(model_meta.get("details"), dict) else {}
        family = str(details.get("family", "")).lower()
        families = details.get("families", [])
        family_blob = " ".join([family, " ".join(str(item).lower() for item in families if item)])
        return "embed" in lowered or "embed" in family_blob

    @staticmethod
    def _is_vision_model(name: str, model_meta: dict) -> bool:
        lowered = name.lower()
        if "llava" in lowered or "vl" in lowered:
            return True
        details = model_meta.get("details", {}) if isinstance(model_meta.get("details"), dict) else {}
        family = str(details.get("family", "")).lower()
        families = details.get("families", [])
        family_blob = " ".join([family, " ".join(str(item).lower() for item in families if item)])
        return "vision" in family_blob or "vl" in family_blob

    def _dynamic_allowlist(self) -> dict[ModelClass, set[str]]:
        fallback = self._fallback_allowlist()
        models = self._catalog.get_models()
        if not models:
            return fallback

        by_name: dict[str, dict] = {
            str(model.get("name", "")).strip(): model
            for model in models
            if str(model.get("name", "")).strip()
        }
        names = set(by_name.keys())
        allowlist: dict[ModelClass, set[str]] = {
            "general": set(names),
            "code": set(names),
            "embedding": {name for name in names if self._is_embedding_model(name, by_name[name])},
            "vision": (
                set(names)
                if self._settings.allow_any_vision_models
                else {name for name in names if self._is_vision_model(name, by_name[name])}
            ),
        }

        defaults: dict[ModelClass, str | None] = {
            "general": self._settings.model_general_default,
            "vision": self._settings.model_vision_default,
            "embedding": self._settings.model_embedding_default,
            "code": self._settings.model_code_default,
        }
        for model_class, default_model in defaults.items():
            if default_model and default_model in names:
                allowlist[model_class].add(default_model)

        for model_class in ("general", "vision", "embedding", "code"):
            if not allowlist[model_class]:
                allowlist[model_class] = fallback.get(model_class, set())

        return allowlist

    def get_available_models(self) -> list[str]:
        return sorted(self._catalog.get_model_names())

    def is_model_allowed(self, model_class: ModelClass, model_name: str) -> bool:
        return model_name in self._dynamic_allowlist().get(model_class, set())

    def select_model(
        self,
        model_class: ModelClass,
        request_override: str | None = None,
        preference_override: str | None = None,
    ) -> ModelSelection:
        rejected_candidates: list[str] = []
        allowlist = self._dynamic_allowlist().get(model_class, set())

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
        dynamic = self._dynamic_allowlist()
        return {
            key: sorted(value)
            for key, value in dynamic.items()
            if value
        }
