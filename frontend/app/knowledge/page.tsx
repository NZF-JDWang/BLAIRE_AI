import { getKnowledgeStatus } from "@/lib/api";

export default async function KnowledgePage() {
  let status:
    | {
        drop_folder: string;
        files_detected: number;
        last_scan_at: string | null;
        qdrant_reachable: boolean;
      }
    | null = null;
  let error = "";

  try {
    status = await getKnowledgeStatus();
  } catch (err) {
    error = err instanceof Error ? err.message : "Failed to load knowledge status";
  }

  return (
    <main style={{ maxWidth: "840px", margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>Knowledge</h1>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {status ? (
        <div style={{ border: "1px solid #cbd5e1", borderRadius: "8px", padding: "12px" }}>
          <p>
            <strong>Drop folder:</strong> <code>{status.drop_folder}</code>
          </p>
          <p>
            <strong>Files detected:</strong> {status.files_detected}
          </p>
          <p>
            <strong>Last scan:</strong>{" "}
            {status.last_scan_at ? new Date(status.last_scan_at).toLocaleString() : "never"}
          </p>
          <p>
            <strong>Qdrant reachable:</strong> {String(status.qdrant_reachable)}
          </p>
        </div>
      ) : null}
    </main>
  );
}

