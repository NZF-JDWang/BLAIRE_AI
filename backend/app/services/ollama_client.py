"""Compatibility wrappers for legacy Ollama naming.

Use app.services.inference_client directly for new code.
"""

from app.services.inference_client import (
    InferenceClient as OllamaClient,
    InferenceModelCatalog as OllamaModelCatalog,
    fetch_available_model_names as fetch_installed_model_names,
)
