"use client";

import { useEffect, useState } from "react";

import {
  ApprovalAuditEvent,
  ApprovalRecord,
  approveApproval,
  executeApproval,
  formatApiError,
  getApprovalAudit,
  getRecentApprovals,
  rejectApproval,
} from "@/lib/api";

function statusClass(status: string): string {
  if (status === "approved" || status === "executed") return "pill success";
  if (status === "rejected" || status === "expired") return "pill error";
  return "pill warn";
}

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
      setError(formatApiError(err, "Failed to load approvals"));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  async function onApprove(id: string) {
    setError("");
    try {
      const result = await approveApproval(id);
      setTokens((prev) => ({ ...prev, [id]: result.execution_token }));
      await load();
    } catch (err) {
      setError(formatApiError(err, "Approve failed"));
    }
  }

  async function onReject(id: string) {
    setError("");
    try {
      await rejectApproval(id, "Rejected from approval queue UI");
      await load();
    } catch (err) {
      setError(formatApiError(err, "Reject failed"));
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
      setError(formatApiError(err, "Execute failed"));
    }
  }

  async function onLoadAudit(id: string) {
    setError("");
    try {
      const events = await getApprovalAudit(id, 100);
      setAudit((prev) => ({ ...prev, [id]: events }));
    } catch (err) {
      setError(formatApiError(err, "Audit load failed"));
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Human-In-The-Loop</p>
        <h1 className="page-title">Review and execute approval-gated actions safely.</h1>
        <p className="page-description">
          Approve first to mint an execution token, then execute with payload hash matching and traceable audit events.
        </p>
      </section>

      <section className="surface stack" aria-label="Approval controls">
        <div className="toolbar">
          <button onClick={() => void load()} disabled={loading} className="button button-primary">
            {loading ? "Refreshing queue..." : "Refresh queue"}
          </button>
          <span className="pill">{approvals.length} records</span>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="surface stack" aria-label="Approval table">
        {approvals.length === 0 ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>No approvals found in recent history.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Class</th>
                  <th>Target</th>
                  <th>Tool</th>
                  <th>Requested By</th>
                  <th>Created</th>
                  <th>Payload Hash</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {approvals.map((approval) => (
                  <tr key={approval.id}>
                    <td>
                      <span className={statusClass(approval.status)}>{approval.status}</span>
                    </td>
                    <td className="mono">{approval.action_class}</td>
                    <td>{approval.target_host || "-"}</td>
                    <td>{approval.tool_name}</td>
                    <td>{approval.requested_by}</td>
                    <td>{new Date(approval.created_at).toLocaleString()}</td>
                    <td className="mono">{approval.payload_hash.slice(0, 14)}...</td>
                    <td>
                      <div className="row">
                        <button onClick={() => void onApprove(approval.id)} className="button">
                          Approve
                        </button>
                        <button
                          onClick={() => void onExecute(approval.id, approval.payload_hash)}
                          className="button button-primary"
                        >
                          Execute
                        </button>
                        <button onClick={() => void onReject(approval.id)} className="button button-danger">
                          Reject
                        </button>
                        <button onClick={() => void onLoadAudit(approval.id)} className="button button-muted">
                          Audit
                        </button>
                      </div>
                      {(audit[approval.id] ?? []).length > 0 ? (
                        <div className="stack" style={{ marginTop: "8px" }}>
                          {(audit[approval.id] ?? []).slice(0, 5).map((event) => (
                            <p key={event.id} className="help-text mono" style={{ margin: 0 }}>
                              {event.event_type} by {event.actor}
                            </p>
                          ))}
                        </div>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
