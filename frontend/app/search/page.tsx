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
    <main style={{ maxWidth: "900px", margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>Search</h1>
      <form onSubmit={onSubmit} style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search query..."
          style={{ flex: 1, padding: "10px", border: "1px solid #94a3b8", borderRadius: "6px" }}
        />
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="">(use preference/default)</option>
          <option value="searxng_only">searxng_only</option>
          <option value="brave_only">brave_only</option>
          <option value="auto_fallback">auto_fallback</option>
          <option value="parallel">parallel</option>
        </select>
        <button type="submit">Search</button>
      </form>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {result ? (
        <section>
          <p style={{ fontFamily: "monospace" }}>
            mode={result.mode} providers={result.providers_used.join(",")}
          </p>
          <ul>
            {result.results.map((item) => (
              <li key={`${item.url}-${item.provider}`} style={{ marginBottom: "12px" }}>
                <a href={item.url} target="_blank" rel="noreferrer">
                  {item.title}
                </a>
                <div>{item.snippet}</div>
                <small>{item.provider}</small>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </main>
  );
}

