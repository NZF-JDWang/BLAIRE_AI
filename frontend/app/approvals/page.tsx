"use client";

import { useEffect, useState } from "react";

import { ApprovalRecord, approveApproval, getPendingApprovals, rejectApproval } from "@/lib/api";

export default function ApprovalsPage() {
  const [approvals, setApprovals] = useState<ApprovalRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastToken, setLastToken] = useState<string>("");

  async function load() {
    setLoading(true);
    setError("");
    try {
      const data = await getPendingApprovals(100);
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
      setLastToken(result.execution_token);
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

  return (
    <main style={{ maxWidth: "1000px", margin: "48px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "16px" }}>Approval Queue</h1>
      <button onClick={load} disabled={loading} style={{ marginBottom: "12px", padding: "8px 12px" }}>
        {loading ? "Refreshing..." : "Refresh"}
      </button>
      {lastToken ? (
        <p style={{ fontFamily: "monospace", background: "#f8fafc", padding: "8px", borderRadius: "6px" }}>
          last execution token: {lastToken}
        </p>
      ) : null}
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {!error && approvals.length === 0 ? <p>No pending approvals.</p> : null}
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
                <td style={{ padding: "8px" }}>
                  <div style={{ display: "flex", gap: "6px" }}>
                    <button onClick={() => onApprove(approval.id)} style={{ padding: "6px 10px" }}>
                      Approve
                    </button>
                    <button onClick={() => onReject(approval.id)} style={{ padding: "6px 10px" }}>
                      Reject
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </main>
  );
}

