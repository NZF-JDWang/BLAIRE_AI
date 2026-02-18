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

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://backend:8000";

export async function getHealth(): Promise<unknown> {
  const response = await fetch(`${apiBaseUrl}/health`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

export async function getRuntimeOptions(): Promise<unknown> {
  const response = await fetch(`${apiBaseUrl}/runtime/options`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Runtime options request failed: ${response.status}`);
  }
  return response.json();
}

export async function getPendingApprovals(limit = 50): Promise<ApprovalRecord[]> {
  const response = await fetch(`${apiBaseUrl}/approvals/pending?limit=${limit}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Pending approvals request failed: ${response.status}`);
  }
  return response.json();
}
