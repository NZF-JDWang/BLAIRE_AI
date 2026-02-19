"use client";

import { useEffect, useState } from "react";

import { getKnowledgeStatus, reindexObsidian } from "@/lib/api";

type KnowledgeStatus = {
  drop_folder: string;
  files_detected: number;
  last_scan_at: string | null;
  qdrant_reachable: boolean;
  obsidian_vault_path: string;
  obsidian_files_detected: number;
  obsidian_last_scan_at: string | null;
};

export default function KnowledgePage() {
  const [status, setStatus] = useState<KnowledgeStatus | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [reindexInfo, setReindexInfo] = useState("");

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
    void refresh();
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

  async function onReindex(fullRescan: boolean) {
    setError("");
    try {
      const result = await reindexObsidian(fullRescan);
      setReindexInfo(`indexed=${result.indexed_files}, unchanged=${result.unchanged_files}, chunks=${result.chunks_indexed}`);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reindex failed");
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Knowledge Operations</p>
        <h1 className="page-title">Ingest files and maintain index health.</h1>
        <p className="page-description">
          Upload source files, monitor storage/index status, and run delta or full Obsidian reindex operations.
        </p>
      </section>

      <section className="surface stack" aria-label="Knowledge controls">
        <div className="toolbar">
          <label className="field-label" style={{ minWidth: "280px" }}>
            Upload file
            <input
              className="file-input"
              type="file"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void onUpload(file);
              }}
              disabled={uploading}
            />
          </label>
          <button onClick={() => void refresh()} disabled={uploading} className="button button-muted">
            Refresh status
          </button>
          <button onClick={() => void onReindex(false)} disabled={uploading} className="button">
            Reindex Obsidian (delta)
          </button>
          <button onClick={() => void onReindex(true)} disabled={uploading} className="button button-primary">
            Reindex Obsidian (full)
          </button>
        </div>
        {uploading ? <p className="help-text">Uploading file and refreshing status...</p> : null}
        {reindexInfo ? <p className="help-text mono">{reindexInfo}</p> : null}
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="surface stack" aria-label="Knowledge status">
        <h2>Status</h2>
        {!status ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Status is unavailable. Use Refresh status to retry.</p>
          </div>
        ) : (
          <div className="stats-grid">
            <article className="stat-card">
              <p className="stat-label">Drop folder</p>
              <p className="stat-value mono">{status.drop_folder}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Files detected</p>
              <p className="stat-value">{status.files_detected}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Last scan</p>
              <p className="stat-value">{status.last_scan_at ? new Date(status.last_scan_at).toLocaleString() : "Never"}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Qdrant reachable</p>
              <p className="stat-value">{String(status.qdrant_reachable)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Obsidian vault</p>
              <p className="stat-value mono">{status.obsidian_vault_path}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Obsidian markdown files</p>
              <p className="stat-value">{status.obsidian_files_detected}</p>
            </article>
          </div>
        )}
      </section>
    </main>
  );
}
