# Code Review: Efficiency & Security

## Scope and approach

This review covers the backend (`FastAPI` services and routes), frontend (`Next.js` client and API proxy), and selected integration points with emphasis on:

- Security controls (auth, secrets handling, input/path validation, network boundaries, defense-in-depth headers)
- Performance/efficiency (memory behavior, scalability bottlenecks, avoidable overhead)

I focused on static analysis of critical paths and validated baseline behavior by running existing tests for related modules.

## Executive summary

The codebase has a strong base: explicit auth checks, typed request models, allowlist-oriented tool policies, and a two-step approval pattern for sensitive MCP actions. The most important gaps are:

1. **API key exposure risk in browser storage/cookies** (High)
2. **Filesystem sandbox symlink handling can be bypassed or misapplied** (High)
3. **Upload endpoint reads whole files into memory before enforcing limits** (Medium)
4. **In-memory per-process rate limiter is not production-resilient and can grow unbounded by key cardinality** (Medium)
5. **Chat streaming path duplicates large response buffers and can inflate memory usage** (Medium)
6. **Security headers are incomplete (no CSP/HSTS/Permissions-Policy)** (Medium)

## Detailed findings

### 1) API keys are stored in `localStorage` and a readable cookie

**Severity:** High  
**Type:** Security (credential exposure)

The frontend writes and reads the API key from `window.localStorage` and also sets a cookie that is readable by JavaScript (`HttpOnly` is not set). Any XSS issue would allow credential theft and persistent account/API compromise.

- `window.localStorage` usage for API key storage/read.  
- Cookie set without `HttpOnly` and without `Secure` flag.

**Evidence:** `frontend/lib/api.ts`.

**Recommendation:**
- Move auth to server-set, short-lived, `HttpOnly; Secure; SameSite=Strict/Lax` session cookies.
- Avoid storing long-lived API keys client-side.
- If API keys must remain, scope to least privilege and rotate aggressively.

### 2) Filesystem sandbox symlink protections are insufficient

**Severity:** High  
**Type:** Security (path/symlink escape risk)

`FilesystemSandbox.validate_target_path()` resolves the user path first and then checks `resolved.is_symlink()`. After `resolve()`, the leaf symlink relationship is effectively flattened; this check is unreliable for preventing symlink tricks. There is also no explicit protection against symlinked parents inside allowlisted roots.

**Evidence:** `backend/app/services/filesystem_sandbox.py`.

**Recommendation:**
- Validate the raw path components before final resolution.
- Reject any path where any segment in the traversal chain is a symlink.
- Use `os.open(..., flags=O_NOFOLLOW|O_CREAT|O_EXCL, dir_fd=...)` patterns (or equivalent) for robust anti-symlink writes.
- Re-check root containment after final canonicalization.

### 3) Upload endpoint reads full request body before max-size enforcement

**Severity:** Medium  
**Type:** Efficiency + Security (memory pressure/DoS)

`upload_to_drop_folder` calls `await file.read()` and only then validates `MAX_UPLOAD_MB`. This creates a per-request memory spike proportional to uploaded size and can be abused for memory exhaustion.

**Evidence:** `backend/app/api/routes/knowledge.py`.

**Recommendation:**
- Stream uploads in chunks (e.g., 1–4MB) to disk while counting bytes.
- Abort and delete partial files once size exceeds configured limit.
- Consider reverse-proxy/body-size caps (NGINX/Traefik) as outer guardrails.

### 4) Rate limiting is process-local and unbounded by key cardinality

**Severity:** Medium  
**Type:** Security + Scalability

The rate limiter is an in-memory `defaultdict(deque)` keyed by user/IP tuple. This is not shared across instances and can grow indefinitely for high-cardinality keys, causing memory growth over time.

**Evidence:** `backend/app/core/rate_limit.py`.

**Recommendation:**
- Replace with distributed limiter (Redis sliding window/token bucket).
- Add global/max-key eviction or TTL compaction strategy.
- Emit metrics on key count and reject rate.

### 5) Chat streaming path stores duplicate full response in memory

**Severity:** Medium  
**Type:** Efficiency

In stream mode, each token is appended to `combined` and then concatenated for a final `done` event. This duplicates data already emitted token-by-token and can increase memory usage on long outputs.

**Evidence:** `backend/app/api/routes/chat.py`.

**Recommendation:**
- Make the final `done` event metadata-only (no full text) or include only a digest/length.
- If full text is required, cap retained output length.
- Consider backpressure/timeout tuning for long streams.

### 6) Security headers middleware lacks several modern hardening headers

**Severity:** Medium  
**Type:** Security hardening

Current middleware sets a basic subset (`nosniff`, `DENY`, `no-referrer`, `no-store`) but does not define CSP/HSTS/Permissions-Policy.

**Evidence:** `backend/app/core/security_headers.py`.

**Recommendation:**
- Add strict `Content-Security-Policy` appropriate to frontend delivery model.
- Add `Strict-Transport-Security` in TLS deployments.
- Add `Permissions-Policy` and consider `Cross-Origin-Opener-Policy`/`Cross-Origin-Resource-Policy` as needed.

## Positive observations

- Role-based auth gating is consistently applied through route dependencies in sensitive endpoints.
- MCP sensitive actions enforce approval flow and payload hash matching before execution.
- Allowlist-based operation checks exist for Home Assistant and homelab operations.

## Prioritized remediation plan

1. **Immediate (High):** remove client-side API key persistence; implement server-side session auth.
2. **Immediate (High):** harden filesystem writes against symlink/path race issues.
3. **Near-term:** chunked upload handling with strict size limits and cleanup.
4. **Near-term:** migrate rate limiting to Redis-backed shared limiter.
5. **Near-term:** reduce chat streaming memory footprint (no full-body accumulation by default).
6. **Ongoing hardening:** expand security headers and validate policy in staging.

## Validation run

Executed targeted backend tests related to reviewed subsystems:

- `pytest -q tests/test_filesystem_sandbox.py tests/test_tool_policy.py tests/test_mcp_client.py tests/test_knowledge_routes.py tests/test_chat_route.py`

Result: **11 passed**.
