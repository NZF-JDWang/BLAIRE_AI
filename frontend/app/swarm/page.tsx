"use client";

import { FormEvent, useState } from "react";

import { formatApiError, getLiveSwarmRuns, ResearchResponse, runResearch, SwarmLiveResponse } from "@/lib/api";

export default function SwarmPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [live, setLive] = useState<SwarmLiveResponse | null>(null);
  const [error, setError] = useState("");

  async function refreshLive() {
    try {
      setLive(await getLiveSwarmRuns(10));
    } catch {
      setLive(null);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await runResearch(query.trim());
      setResult(response);
      await refreshLive();
    } catch (err) {
      setError(formatApiError(err, "Research failed"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Swarm Orchestration</p>
        <h1 className="page-title">Launch multi-agent research and inspect execution traces.</h1>
        <p className="page-description">
          Submit a query to the swarm pipeline, then review supervisor summaries, worker outputs, and live run state.
        </p>
      </section>

      <section className="surface stack" aria-label="Run swarm">
        <form onSubmit={onSubmit} className="stack">
          <label className="field-label">
            Research prompt
            <textarea
              className="textarea"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Compare local LLM serving strategies for incident-response assistants..."
            />
          </label>
          <div className="toolbar">
            <button type="submit" disabled={loading || !query.trim()} className="button button-primary">
              {loading ? "Running swarm..." : "Run swarm"}
            </button>
            <button type="button" onClick={() => void refreshLive()} className="button button-muted">
              Refresh live runs
            </button>
          </div>
        </form>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="surface stack" aria-label="Swarm result">
        <h2>Latest result</h2>
        {!result ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>No completed run yet. Submit a query to populate this panel.</p>
          </div>
        ) : (
          <div className="stack">
            <p>
              <strong>Supervisor summary:</strong> {result.supervisor_summary}
            </p>
            {result.workers.map((worker) => (
              <article key={worker.worker_id} className="surface" style={{ padding: "12px" }}>
                <p style={{ marginBottom: "6px" }}>
                  <strong>{worker.worker_id}</strong>
                </p>
                <p style={{ marginTop: 0 }}>{worker.summary}</p>
                <ul className="list-reset">
                  {worker.sources.map((source) => (
                    <li key={source}>
                      <a href={source} target="_blank" rel="noreferrer">
                        {source}
                      </a>
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="surface stack" aria-label="Live swarm runs">
        <h2>Live runs</h2>
        {!live || live.runs.length === 0 ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>No active or recent live runs available.</p>
          </div>
        ) : (
          <div className="panel-list">
            {live.runs.map((run) => (
              <article key={run.run_id} className="surface" style={{ padding: "12px" }}>
                <p style={{ marginBottom: "6px" }}>
                  <strong>{run.query}</strong>
                </p>
                <p className="help-text">{new Date(run.created_at).toLocaleString()}</p>
                <p>{run.supervisor_summary}</p>
                <div className="mono" style={{ fontSize: "0.82rem" }}>
                  {run.trace.map((step, idx) => (
                    <div key={`${run.run_id}-${idx}`}>
                      {step.status}: {step.step}
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
