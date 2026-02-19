export type ApprovalRecord = {
  id: string;
  status: string;
  action_class: string;
  target_host: string;
  tool_name: string;
  payload_hash: string;
  requested_by: string;
  approved_by: string | null;
  created_at: string;
  updated_at: string;
  token_expires_at: string | null;
  executed_at: string | null;
  rejection_reason: string | null;
};

export type ApprovalAuditEvent = {
  id: number;
  approval_id: string | null;
  event_type: string;
  actor: string;
  details: Record<string, unknown>;
  event_time: string;
};

export type RuntimeOptions = {
  search_modes: string[];
  default_search_mode: string;
  model_allowlist: Record<string, string[]>;
  available_models?: string[];
  available_models_by_class?: Record<string, string[]>;
  sensitive_actions_enabled: boolean;
  approval_token_ttl_minutes: number;
  allowed_network_hosts: string[];
  allowed_network_tools: string[];
  tools: Array<{
    name: string;
    action_class: string;
    description: string;
    requires_target_host: boolean;
  }>;
};

export type UserPreferences = {
  subject: string;
  search_mode: string;
  model_class: string;
  model_override: string | null;
  temperature: number;
  top_p: number;
  max_tokens: number | null;
  context_window_tokens: number | null;
  use_rag: boolean;
  retrieval_k: number;
  updated_at: string;
};

export type RuntimeConfigEffective = {
  search_mode_default: string;
  sensitive_actions_enabled: boolean;
  approval_token_ttl_minutes: number;
  allowed_network_hosts: string[];
  allowed_network_tools: string[];
  allowed_obsidian_paths: string[];
  allowed_ha_operations: string[];
  allowed_homelab_operations: string[];
};

export type RuntimeConfigOverrides = {
  search_mode_default: string | null;
  sensitive_actions_enabled: boolean | null;
  approval_token_ttl_minutes: number | null;
  allowed_network_hosts: string | null;
  allowed_network_tools: string | null;
  allowed_obsidian_paths: string | null;
  allowed_ha_operations: string | null;
  allowed_homelab_operations: string | null;
  updated_by: string | null;
  updated_at: string | null;
};

export type RuntimeConfigBundle = {
  effective: RuntimeConfigEffective;
  overrides: RuntimeConfigOverrides;
};

export type RuntimeConfigAuditEvent = {
  id: number;
  actor: string;
  previous_overrides: RuntimeConfigOverrides;
  new_overrides: RuntimeConfigOverrides;
  event_time: string;
};

export class ApiRequestError extends Error {
  status: number;
  detail: string | null;

  constructor(status: number, detail: string | null, message: string) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.detail = detail;
  }
}

function apiBaseUrl(): string {
  if (typeof window === "undefined") {
    return process.env.INTERNAL_API_BASE_URL ?? "http://backend:8000";
  }
  return "/api";
}

function apiKeyHeader(): Record<string, string> {
  if (typeof window === "undefined") {
    const key = process.env.FRONTEND_PROXY_API_KEY ?? "";
    return key ? { "X-API-Key": key } : {};
  }
  const key = window.localStorage.getItem("blaire_api_key") ?? "";
  return key ? { "X-API-Key": key } : {};
}

export function setBrowserApiKey(key: string) {
  const trimmed = key.trim();
  if (!trimmed) {
    window.localStorage.removeItem("blaire_api_key");
    document.cookie = "blaire_api_key=; Path=/; Max-Age=0; SameSite=Lax";
    window.dispatchEvent(new Event("blaire-api-key-changed"));
    return;
  }
  window.localStorage.setItem("blaire_api_key", trimmed);
  document.cookie = `blaire_api_key=${encodeURIComponent(trimmed)}; Path=/; SameSite=Lax`;
  window.dispatchEvent(new Event("blaire-api-key-changed"));
}

export function getBrowserApiKey(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem("blaire_api_key") ?? "";
}

async function buildApiError(response: Response, label: string): Promise<ApiRequestError> {
  let detail: string | null = null;
  try {
    const data = (await response.json()) as { detail?: string };
    if (typeof data.detail === "string") {
      detail = data.detail;
    }
  } catch {
    detail = null;
  }
  return new ApiRequestError(
    response.status,
    detail,
    detail ? `${label}: ${response.status} (${detail})` : `${label}: ${response.status}`,
  );
}

export function formatApiError(err: unknown, fallback: string): string {
  if (err instanceof ApiRequestError) {
    if (err.status === 401) {
      return "Missing API key. Open Settings and add a valid user or admin key.";
    }
    if (err.status === 403) {
      return err.detail ? `Access denied: ${err.detail}` : "Access denied. Check key role and allowlists.";
    }
    return err.detail ? `${fallback}: ${err.detail}` : `${fallback}: ${err.status}`;
  }
  return err instanceof Error ? err.message : fallback;
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? {});
  for (const [key, value] of Object.entries(apiKeyHeader())) {
    headers.set(key, value);
  }
  return fetch(`${apiBaseUrl()}${path}`, {
    ...init,
    headers,
  });
}

export async function getHealth(): Promise<unknown> {
  const response = await apiFetch("/health", { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Health check failed");
  }
  return response.json();
}

export async function getRuntimeOptions(): Promise<unknown> {
  const response = await apiFetch("/runtime/options", { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Runtime options request failed");
  }
  return response.json();
}

export async function getRuntimeOptionsTyped(): Promise<RuntimeOptions> {
  const data = await getRuntimeOptions();
  return data as RuntimeOptions;
}

export async function getRecentApprovals(limit = 100): Promise<ApprovalRecord[]> {
  const response = await apiFetch(`/approvals/recent?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Recent approvals request failed");
  }
  return response.json();
}

export async function getApprovalAudit(approvalId: string, limit = 200): Promise<ApprovalAuditEvent[]> {
  const response = await apiFetch(`/approvals/${approvalId}/audit?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Approval audit request failed");
  }
  return response.json();
}

export async function approveApproval(approvalId: string): Promise<{
  approval: ApprovalRecord;
  execution_token: string;
  expires_at: string;
}> {
  const response = await apiFetch(`/approvals/${approvalId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" }
  });
  if (!response.ok) {
    throw await buildApiError(response, "Approve request failed");
  }
  return response.json();
}

export async function rejectApproval(approvalId: string, reason: string): Promise<ApprovalRecord> {
  const response = await apiFetch(`/approvals/${approvalId}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason })
  });
  if (!response.ok) {
    throw await buildApiError(response, "Reject request failed");
  }
  return response.json();
}

export async function executeApproval(approvalId: string, executionToken: string, expectedPayloadHash: string): Promise<ApprovalRecord> {
  const response = await apiFetch(`/approvals/${approvalId}/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      execution_token: executionToken,
      expected_payload_hash: expectedPayloadHash
    })
  });
  if (!response.ok) {
    throw await buildApiError(response, "Execute request failed");
  }
  return response.json();
}

export type KnowledgeStatus = {
  drop_folder: string;
  files_detected: number;
  last_scan_at: string | null;
  qdrant_reachable: boolean;
  obsidian_vault_path: string;
  obsidian_files_detected: number;
  obsidian_last_scan_at: string | null;
};

export async function getKnowledgeStatus(): Promise<KnowledgeStatus> {
  const response = await apiFetch("/knowledge/status", { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Knowledge status request failed");
  }
  return response.json();
}

export async function uploadKnowledgeFile(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  const response = await apiFetch("/knowledge/upload", {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw await buildApiError(response, "Upload failed");
  }
}

export async function reindexObsidian(fullRescan = false): Promise<{
  scanned_files: number;
  indexed_files: number;
  unchanged_files: number;
  chunks_indexed: number;
}> {
  const response = await apiFetch("/knowledge/obsidian/reindex", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ full_rescan: fullRescan, limit: 5000 }),
  });
  if (!response.ok) {
    throw await buildApiError(response, "Obsidian reindex failed");
  }
  return response.json();
}

export type ResearchResponse = {
  query: string;
  supervisor_summary: string;
  workers: Array<{
    worker_id: string;
    summary: string;
    sources: string[];
  }>;
  citations?: Array<{
    url: string;
    worker_ids: string[];
    occurrences: number;
  }>;
  trace?: Array<{
    step: string;
    status: "started" | "completed" | "failed" | "skipped";
    timestamp: string;
    details: Record<string, string | number | boolean>;
  }>;
};

export type SwarmLiveResponse = {
  runs: Array<{
    run_id: string;
    query: string;
    created_at: string;
    supervisor_summary: string;
    workers: Array<{
      worker_id: string;
      summary: string;
      sources: string[];
    }>;
    trace: Array<{
      step: string;
      status: "started" | "completed" | "failed" | "skipped";
      timestamp: string;
      details: Record<string, string | number | boolean>;
    }>;
  }>;
};

export async function runResearch(query: string, searchMode?: string): Promise<ResearchResponse> {
  let effectiveSearchMode = searchMode;
  if (!effectiveSearchMode) {
    try {
      const prefs = await getMyPreferences();
      effectiveSearchMode = prefs.search_mode;
    } catch {
      effectiveSearchMode = undefined;
    }
  }
  const response = await apiFetch("/agents/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      search_mode: effectiveSearchMode || null
    })
  });
  if (!response.ok) {
    throw await buildApiError(response, "Research request failed");
  }
  return response.json();
}

export async function getLiveSwarmRuns(limit = 20): Promise<SwarmLiveResponse> {
  const response = await apiFetch(`/agents/swarm/live?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Live swarm request failed");
  }
  return response.json();
}

export async function runSearch(query: string, mode?: string): Promise<{
  mode: string;
  providers_used: string[];
  results: Array<{ title: string; url: string; snippet: string; provider: string }>;
}> {
  const response = await apiFetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode: mode ?? null, limit: 10 }),
  });
  if (!response.ok) {
    throw await buildApiError(response, "Search request failed");
  }
  return response.json();
}

export type DependencyStatus = {
  dependencies: Array<{
    name: string;
    ok: boolean;
    detail: string;
    required: boolean;
    enabled: boolean;
  }>;
};

export async function getDependencyStatus(): Promise<DependencyStatus> {
  const response = await apiFetch("/health/dependencies", { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Dependency status request failed");
  }
  return response.json();
}

export async function getMyPreferences(): Promise<UserPreferences> {
  const response = await apiFetch("/preferences/me", { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Preferences request failed");
  }
  return response.json();
}

export async function updateMyPreferences(update: {
  search_mode: string;
  model_class: string;
  model_override: string | null;
  temperature: number;
  top_p: number;
  max_tokens: number | null;
  context_window_tokens: number | null;
  use_rag: boolean;
  retrieval_k: number;
}): Promise<UserPreferences> {
  const response = await apiFetch("/preferences/me", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!response.ok) {
    throw await buildApiError(response, "Preferences update failed");
  }
  return response.json();
}

export async function getRuntimeConfig(): Promise<RuntimeConfigBundle> {
  const response = await apiFetch("/runtime/config", { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Runtime config request failed");
  }
  return response.json();
}

export async function updateRuntimeConfig(update: {
  search_mode_default: string | null;
  sensitive_actions_enabled: boolean | null;
  approval_token_ttl_minutes: number | null;
  allowed_network_hosts: string | null;
  allowed_network_tools: string | null;
  allowed_obsidian_paths: string | null;
  allowed_ha_operations: string | null;
  allowed_homelab_operations: string | null;
}): Promise<RuntimeConfigBundle> {
  const response = await apiFetch("/runtime/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!response.ok) {
    throw await buildApiError(response, "Runtime config update failed");
  }
  return response.json();
}

export async function getRuntimeConfigAudit(limit = 100): Promise<RuntimeConfigAuditEvent[]> {
  const response = await apiFetch(`/runtime/config/audit?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw await buildApiError(response, "Runtime config audit request failed");
  }
  return response.json();
}
