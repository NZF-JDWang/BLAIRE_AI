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
  return {};
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

export type KnowledgeStatus = {
  drop_folder: string;
  files_detected: number;
  last_scan_at: string | null;
  qdrant_reachable: boolean;
};

export async function getKnowledgeStatus(): Promise<KnowledgeStatus> {
  const response = await apiFetch("/knowledge/status", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Knowledge status request failed: ${response.status}`);
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
};

export async function runResearch(query: string, searchMode?: string): Promise<ResearchResponse> {
  const response = await apiFetch("/agents/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      search_mode: searchMode || null
    })
  });
  if (!response.ok) {
    throw new Error(`Research request failed: ${response.status}`);
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
