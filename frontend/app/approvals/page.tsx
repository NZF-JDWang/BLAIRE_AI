"use client";

import { useEffect, useState } from "react";

import {
  ApprovalAuditEvent,
  ApprovalRecord,
  getApprovalAudit,
  getRecentApprovals,
  rejectApproval,
  submitCliDecision
} from "@/lib/api";

export default function ApprovalsPage() {
  const [approvals, setApprovals] = useState<ApprovalRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [audit, setAudit] = useState<Record<string, ApprovalAuditEvent[]>>({});
  const [result, setResult] = useState<Record<string, string>>({});

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

  async function onReject(id: string) {
    setError("");
    try {
      await rejectApproval(id, "Rejected from approval queue UI");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reject failed");
    }
  }

  async function onCliDecision(id: string, decision: "allow_once" | "allow_always") {
    setError("");
    try {
      const response = await submitCliDecision(id, decision);
      if (response.record) {
        setResult((prev) => ({ ...prev, [id]: JSON.stringify(response.record) }));
      }
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Decision failed");
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
    <main style={{ maxWidth: "1100px", margin: "48px auto", padding: "0 16px" }}>
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
              <th style={{ padding: "8px" }}>Command</th>
              <th style={{ padding: "8px" }}>Requested By</th>
              <th style={{ padding: "8px" }}>Created</th>
              <th style={{ padding: "8px" }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {approvals.map((approval) => {
              const cmd = typeof approval.action_payload?.command === "string" ? approval.action_payload.command : "-";
              const isCli = approval.tool_name === "cli.execute";
              return (
                <tr key={approval.id} style={{ borderBottom: "1px solid #e2e8f0" }}>
                  <td style={{ padding: "8px", fontFamily: "monospace" }}>{approval.status}</td>
                  <td style={{ padding: "8px", fontFamily: "monospace" }}>{approval.action_class}</td>
                  <td style={{ padding: "8px" }}>{approval.target_host}</td>
                  <td style={{ padding: "8px" }}>{approval.tool_name}</td>
                  <td style={{ padding: "8px", fontFamily: "monospace" }}>{cmd}</td>
                  <td style={{ padding: "8px" }}>{approval.requested_by}</td>
                  <td style={{ padding: "8px" }}>{new Date(approval.created_at).toLocaleString()}</td>
                  <td style={{ padding: "8px" }}>
                    <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                      {isCli ? (
                        <>
                          <button onClick={() => onCliDecision(approval.id, "allow_once")} style={{ padding: "6px 10px" }}>
                            Allow Once
                          </button>
                          <button onClick={() => onCliDecision(approval.id, "allow_always")} style={{ padding: "6px 10px" }}>
                            Allow Always
                          </button>
                        </>
                      ) : null}
                      <button onClick={() => onReject(approval.id)} style={{ padding: "6px 10px" }}>
                        Reject
                      </button>
                      <button onClick={() => onLoadAudit(approval.id)} style={{ padding: "6px 10px" }}>
                        Audit
                      </button>
                    </div>
                    {result[approval.id] ? (
                      <pre style={{ marginTop: "8px", whiteSpace: "pre-wrap", maxWidth: "460px", fontSize: "0.75rem" }}>{result[approval.id]}</pre>
                    ) : null}
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
              );
            })}
          </tbody>
        </table>
      ) : null}
    </main>
  );
}
