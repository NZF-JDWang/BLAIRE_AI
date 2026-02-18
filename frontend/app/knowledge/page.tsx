"use client";

import { useEffect, useState } from "react";

import { getKnowledgeStatus } from "@/lib/api";

type KnowledgeStatus = {
  drop_folder: string;
  files_detected: number;
  last_scan_at: string | null;
  qdrant_reachable: boolean;
};

export default function KnowledgePage() {
  const [status, setStatus] = useState<KnowledgeStatus | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);

  async function refresh() {
    setError("");
    try {
      const data = await getKnowledgeStatus();
      setStatus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load knowledge status");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function onUpload(file: File) {
    setError("");
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch("/api/knowledge/upload", { method: "POST", body: form });
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <main style={{ maxWidth: "840px", margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>Knowledge</h1>
      <div style={{ marginBottom: "12px", display: "flex", gap: "8px", alignItems: "center" }}>
        <input
          type="file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void onUpload(file);
          }}
        />
        <button onClick={() => void refresh()} disabled={uploading}>
          Refresh
        </button>
      </div>
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

