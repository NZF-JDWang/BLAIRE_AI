"use client";

import { FormEvent, useEffect, useState } from "react";

import { formatApiError, getRuntimeOptionsTyped, runSearch } from "@/lib/api";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("");
  const [modes, setModes] = useState<string[]>(["searxng_only", "brave_only", "auto_fallback", "parallel"]);
  const [defaultMode, setDefaultMode] = useState<string>("unknown");
  const [result, setResult] = useState<{
    mode: string;
    providers_used: string[];
    results: Array<{ title: string; url: string; snippet: string; provider: string }>;
  } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getRuntimeOptionsTyped()
      .then((runtime) => {
        setModes(runtime.search_modes);
        setDefaultMode(runtime.default_search_mode);
      })
      .catch(() => undefined);
  }, []);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setError("");
    try {
      const data = await runSearch(query.trim(), mode || undefined);
      setResult(data);
    } catch (err) {
      setError(formatApiError(err, "Search failed"));
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Search Endpoint</p>
        <h1 className="page-title">Run direct web search tests by provider mode.</h1>
        <p className="page-description">Current runtime default mode: {defaultMode}</p>
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
              {modes.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
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
