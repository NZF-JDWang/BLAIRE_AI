"use client";

import { useEffect, useState } from "react";

import {
  ApprovalAuditEvent,
  ApprovalRecord,
  approveApproval,
  executeApproval,
  getApprovalAudit,
  getRecentApprovals,
  rejectApproval
} from "@/lib/api";

export default function ApprovalsPage() {
  const [approvals, setApprovals] = useState<ApprovalRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tokens, setTokens] = useState<Record<string, string>>({});
  const [audit, setAudit] = useState<Record<string, ApprovalAuditEvent[]>>({});

  async function load() {
    setLoading(true);
    setError("");
    try {
      const data = await getRecentApprovals(150);
      setApprovals(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load approvals");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function onApprove(id: string) {
    setError("");
    try {
      const result = await approveApproval(id);
      setTokens((prev) => ({ ...prev, [id]: result.execution_token }));
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Approve failed");
    }
  }

  async function onReject(id: string) {
    setError("");
    try {
      await rejectApproval(id, "Rejected from approval queue UI");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reject failed");
    }
  }

  async function onExecute(id: string, expectedPayloadHash: string) {
    setError("");
    const token = tokens[id];
    if (!token) {
      setError("No execution token available for this approval. Approve first.");
      return;
    }
    try {
      await executeApproval(id, token, expectedPayloadHash);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Execute failed");
    }
  }

  async function onLoadAudit(id: string) {
    setError("");
    try {
      const events = await getApprovalAudit(id, 100);
      setAudit((prev) => ({ ...prev, [id]: events }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Audit load failed");
    }
  }

  return (
    <main style={{ maxWidth: "1000px", margin: "48px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "16px" }}>Approval Queue</h1>
      <button onClick={load} disabled={loading} style={{ marginBottom: "12px", padding: "8px 12px" }}>
        {loading ? "Refreshing..." : "Refresh"}
      </button>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {!error && approvals.length === 0 ? <p>No approvals found.</p> : null}
      {!error && approvals.length > 0 ? (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.92rem" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1" }}>
              <th style={{ padding: "8px" }}>Status</th>
              <th style={{ padding: "8px" }}>Class</th>
              <th style={{ padding: "8px" }}>Target</th>
              <th style={{ padding: "8px" }}>Tool</th>
              <th style={{ padding: "8px" }}>Requested By</th>
              <th style={{ padding: "8px" }}>Created</th>
              <th style={{ padding: "8px" }}>Payload Hash</th>
              <th style={{ padding: "8px" }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {approvals.map((approval) => (
              <tr key={approval.id} style={{ borderBottom: "1px solid #e2e8f0" }}>
                <td style={{ padding: "8px", fontFamily: "monospace" }}>{approval.status}</td>
                <td style={{ padding: "8px", fontFamily: "monospace" }}>{approval.action_class}</td>
                <td style={{ padding: "8px" }}>{approval.target_host}</td>
                <td style={{ padding: "8px" }}>{approval.tool_name}</td>
                <td style={{ padding: "8px" }}>{approval.requested_by}</td>
                <td style={{ padding: "8px" }}>{new Date(approval.created_at).toLocaleString()}</td>
                <td style={{ padding: "8px", fontFamily: "monospace" }}>{approval.payload_hash.slice(0, 14)}...</td>
                <td style={{ padding: "8px" }}>
                  <div style={{ display: "flex", gap: "6px" }}>
                    <button onClick={() => onApprove(approval.id)} style={{ padding: "6px 10px" }}>
                      Approve
                    </button>
                    <button
                      onClick={() => onExecute(approval.id, approval.payload_hash)}
                      style={{ padding: "6px 10px" }}
                    >
                      Execute
                    </button>
                    <button onClick={() => onReject(approval.id)} style={{ padding: "6px 10px" }}>
                      Reject
                    </button>
                    <button onClick={() => onLoadAudit(approval.id)} style={{ padding: "6px 10px" }}>
                      Audit
                    </button>
                  </div>
                  {(audit[approval.id] ?? []).length > 0 ? (
                    <div style={{ marginTop: "8px", fontSize: "0.8rem", maxWidth: "380px" }}>
                      {(audit[approval.id] ?? []).slice(0, 5).map((event) => (
                        <div key={event.id} style={{ marginBottom: "4px", fontFamily: "monospace" }}>
                          {event.event_type} by {event.actor}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </main>
  );
}
