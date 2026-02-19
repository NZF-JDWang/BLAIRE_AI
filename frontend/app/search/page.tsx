"use client";

import { FormEvent, useState } from "react";

import { runSearch } from "@/lib/api";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("");
  const [result, setResult] = useState<{
    mode: string;
    providers_used: string[];
    results: Array<{ title: string; url: string; snippet: string; provider: string }>;
  } | null>(null);
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setError("");
    try {
      const data = await runSearch(query.trim(), mode || undefined);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Search Endpoint</p>
        <h1 className="page-title">Run direct web search tests by provider mode.</h1>
      </section>

      <section className="surface stack">
        <form onSubmit={onSubmit} className="stack">
          <label className="field-label">
            Query
            <input
              className="input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search query..."
            />
          </label>
          <label className="field-label">
            Mode
            <select className="select" value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="">(use preference/default)</option>
              <option value="searxng_only">searxng_only</option>
              <option value="brave_only">brave_only</option>
              <option value="auto_fallback">auto_fallback</option>
              <option value="parallel">parallel</option>
            </select>
          </label>
          <div>
            <button className="button button-primary" type="submit">
              Search
            </button>
          </div>
        </form>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="surface stack">
        {!result ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>No search results yet.</p>
          </div>
        ) : (
          <>
            <p className="help-text mono">
              mode={result.mode} providers={result.providers_used.join(",")}
            </p>
            <div className="panel-list">
              {result.results.map((item) => (
                <article key={`${item.url}-${item.provider}`} className="surface" style={{ padding: "12px" }}>
                  <a href={item.url} target="_blank" rel="noreferrer">
                    <strong>{item.title}</strong>
                  </a>
                  <p>{item.snippet}</p>
                  <p className="help-text mono">{item.provider}</p>
                </article>
              ))}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
