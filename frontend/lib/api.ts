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
  updated_at: string;
};

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
    return;
  }
  window.localStorage.setItem("blaire_api_key", trimmed);
  document.cookie = `blaire_api_key=${encodeURIComponent(trimmed)}; Path=/; SameSite=Lax`;
}

export function getBrowserApiKey(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem("blaire_api_key") ?? "";
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
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
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

export async function getRuntimeOptions(): Promise<unknown> {
  const response = await apiFetch("/runtime/options", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Runtime options request failed: ${response.status}`);
  }
  return response.json();
}

export async function getRuntimeOptionsTyped(): Promise<RuntimeOptions> {
  const data = await getRuntimeOptions();
  return data as RuntimeOptions;
}

export async function getPendingApprovals(limit = 50): Promise<ApprovalRecord[]> {
  const response = await apiFetch(`/approvals/pending?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Pending approvals request failed: ${response.status}`);
  }
  return response.json();
}

export async function getRecentApprovals(limit = 100): Promise<ApprovalRecord[]> {
  const response = await apiFetch(`/approvals/recent?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Recent approvals request failed: ${response.status}`);
  }
  return response.json();
}

export async function getApprovalAudit(approvalId: string, limit = 200): Promise<ApprovalAuditEvent[]> {
  const response = await apiFetch(`/approvals/${approvalId}/audit?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Approval audit request failed: ${response.status}`);
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
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor: "ui-admin" })
  });
  if (!response.ok) {
    throw new Error(`Approve request failed: ${response.status}`);
  }
  return response.json();
}

export async function rejectApproval(approvalId: string, reason: string): Promise<ApprovalRecord> {
  const response = await apiFetch(`/approvals/${approvalId}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor: "ui-admin", reason })
  });
  if (!response.ok) {
    throw new Error(`Reject request failed: ${response.status}`);
  }
  return response.json();
}

export async function executeApproval(approvalId: string, executionToken: string, expectedPayloadHash: string): Promise<ApprovalRecord> {
  const response = await apiFetch(`/approvals/${approvalId}/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      actor: "ui-admin",
      execution_token: executionToken,
      expected_payload_hash: expectedPayloadHash
    })
  });
  if (!response.ok) {
    throw new Error(`Execute request failed: ${response.status}`);
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
    throw new Error(`Knowledge status request failed: ${response.status}`);
  }
  return response.json();
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
    throw new Error(`Obsidian reindex failed: ${response.status}`);
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
    throw new Error(`Research request failed: ${response.status}`);
  }
  return response.json();
}

export async function getLiveSwarmRuns(limit = 20): Promise<SwarmLiveResponse> {
  const response = await apiFetch(`/agents/swarm/live?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Live swarm request failed: ${response.status}`);
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
    throw new Error(`Search request failed: ${response.status}`);
  }
  return response.json();
}

export type DependencyStatus = {
  dependencies: Array<{
    name: string;
    ok: boolean;
    detail: string;
  }>;
};

export async function getDependencyStatus(): Promise<DependencyStatus> {
  const response = await apiFetch("/health/dependencies", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Dependency status request failed: ${response.status}`);
  }
  return response.json();
}

export async function getMyPreferences(): Promise<UserPreferences> {
  const response = await apiFetch("/preferences/me", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Preferences request failed: ${response.status}`);
  }
  return response.json();
}

export async function updateMyPreferences(update: {
  search_mode: string;
  model_class: string;
  model_override: string | null;
}): Promise<UserPreferences> {
  const response = await apiFetch("/preferences/me", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!response.ok) {
    throw new Error(`Preferences update failed: ${response.status}`);
  }
  return response.json();
}
