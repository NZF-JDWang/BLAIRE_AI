# Model Selection Guide

This guide explains how BLAIRE decides which Ollama models are allowed for each model class.

## How selection works
For each class (`general`, `vision`, `embedding`, `code`), BLAIRE builds an allowlist in this order:
1. Baseline built-in models
2. Class defaults (`MODEL_*_DEFAULT`)
3. Class extra allowlist entries (`MODEL_ALLOWLIST_EXTRA_*`)
4. If `MODEL_ALLOW_ANY_OLLAMA=true`, all installed Ollama models from `GET /api/tags`
5. Final removal by `MODEL_DISALLOWLIST`

`POST /chat` with `model_override` is accepted only if the override is in the computed class allowlist.

## Environment variables
- `MODEL_GENERAL_DEFAULT`
- `MODEL_VISION_DEFAULT`
- `MODEL_EMBEDDING_DEFAULT`
- `MODEL_CODE_DEFAULT`
- `MODEL_ALLOW_ANY_OLLAMA` (`true`/`false`, default `false`)
- `MODEL_ALLOWLIST_EXTRA_GENERAL` (comma-separated)
- `MODEL_ALLOWLIST_EXTRA_VISION` (comma-separated)
- `MODEL_ALLOWLIST_EXTRA_EMBEDDING` (comma-separated)
- `MODEL_ALLOWLIST_EXTRA_CODE` (comma-separated)
- `MODEL_DISALLOWLIST` (comma-separated, applied last)

Example:

```env
MODEL_ALLOW_ANY_OLLAMA=true
MODEL_ALLOWLIST_EXTRA_GENERAL=gpt-oss:20b,dolphin-llama3:8b-v2.9-q4_K_M
MODEL_DISALLOWLIST=old-model:7b
```

## Runtime APIs
- `GET /runtime/options`  
  Returns `model_allowlist`, `available_models`, and `available_models_by_class`.
- `GET /models`  
  Returns:
  - `installed_models`
  - `allowlist`
  - `defaults`
  - `model_allow_any_ollama`

## Quick verification
1. Check models:
```bash
curl -s http://localhost:8000/models -H "X-API-Key: <KEY>" | jq
```
2. Test override:
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <KEY>" \
  -d '{
    "messages":[{"role":"user","content":"hello"}],
    "model_override":"dolphin-llama3:8b-v2.9-q4_K_M",
    "stream":false,
    "use_rag":false
  }' | jq
```
