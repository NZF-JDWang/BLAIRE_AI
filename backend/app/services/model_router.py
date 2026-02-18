from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.services.inference_client import fetch_available_model_names

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
        self._installed_models = self._load_installed_models()
        self._allowlist = self._build_allowlist()

    def _baseline_allowlist(self) -> dict[ModelClass, set[str]]:
        return {
            "general": {
                "llama3.2:3b",
                "qwen2.5:7b-instruct",
            },
            "vision": {
                "llava:7b",
                "llava:13b",
            },
            "embedding": {
                "nomic-embed-text:v1.5",
            },
            "code": {
                "qwen2.5-coder:7b",
            },
        }

    def _defaults(self) -> dict[ModelClass, str | None]:
        return {
            "general": self._settings.model_general_default,
            "vision": self._settings.model_vision_default,
            "embedding": self._settings.model_embedding_default,
            "code": self._settings.model_code_default,
        }

    def _load_installed_models(self) -> set[str]:
        try:
            return set(fetch_available_model_names(self._settings.inference_base_url))
        except Exception:  # noqa: BLE001
            return set()

    def _build_allowlist(self) -> dict[ModelClass, set[str]]:
        allowlist = self._baseline_allowlist()
        defaults = self._defaults()
        extras: dict[ModelClass, list[str]] = {
            "general": self._settings.model_allowlist_extra_general_list(),
            "vision": self._settings.model_allowlist_extra_vision_list(),
            "embedding": self._settings.model_allowlist_extra_embedding_list(),
            "code": self._settings.model_allowlist_extra_code_list(),
        }

        for model_class, default_model in defaults.items():
            if default_model:
                allowlist[model_class].add(default_model)

        for model_class, model_names in extras.items():
            allowlist[model_class].update(model_names)

        if self._settings.model_allow_any_inference and self._installed_models:
            for model_class in ("general", "vision", "embedding", "code"):
                allowlist[model_class].update(self._installed_models)

        disallowed = set(self._settings.model_disallowlist_list())
        if disallowed:
            for model_class in ("general", "vision", "embedding", "code"):
                allowlist[model_class].difference_update(disallowed)

        return allowlist

    def get_available_models(self) -> list[str]:
        available: set[str] = set()
        for model_names in self._allowlist.values():
            available.update(model_names)
        return sorted(available)

    def get_installed_models(self) -> list[str]:
        return sorted(self._installed_models)

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

        defaults = self._defaults()
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
