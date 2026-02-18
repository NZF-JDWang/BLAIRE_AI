import { getPendingApprovals } from "@/lib/api";

export default async function ApprovalsPage() {
  let approvals: Awaited<ReturnType<typeof getPendingApprovals>> = [];
  let error = "";
  try {
    approvals = await getPendingApprovals(100);
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load approvals";
  }

  return (
    <main style={{ maxWidth: "1000px", margin: "48px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "16px" }}>Approval Queue</h1>
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
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </main>
  );
}

