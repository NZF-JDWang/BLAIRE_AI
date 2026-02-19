"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { getMyPreferences, getRuntimeOptionsTyped, RuntimeOptions } from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type Citation = {
  source_name?: string;
  source_path?: string;
  file_type?: string;
  chunk_index?: number;
  score?: number;
  text?: string;
  ingested_at?: string;
};

const API_BASE = "/api";

async function streamChat(params: {
  messages: Array<{ role: "user" | "assistant" | "system"; content: string }>;
  modelClass: string;
  modelOverride?: string;
  onToken: (token: string) => void;
  onMeta: (meta: { citations?: Citation[]; rag_status?: string; rag_error?: string | null }) => void;
  onDone: () => void;
}): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: params.messages,
      model_class: params.modelClass,
      model_override: params.modelOverride || null,
      stream: true,
    }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const eventChunk of events) {
      const lines = eventChunk.split("\n");
      const event = lines.find((line) => line.startsWith("event:"))?.replace("event:", "").trim();
      const dataLine = lines.find((line) => line.startsWith("data:"))?.replace("data:", "").trim();
      if (!dataLine) continue;
      let payload: {
        text?: string;
        message?: string;
        citations?: Citation[];
        rag_status?: string;
        rag_error?: string | null;
      } = {};
      try {
        payload = JSON.parse(dataLine);
      } catch {
        continue;
      }
      if (event === "meta") {
        params.onMeta(payload);
      }
      if (event === "token" && payload.text) {
        params.onToken(payload.text);
      }
      if (event === "done") {
        params.onDone();
      }
      if (event === "error") {
        throw new Error(payload.message ?? "Stream failed");
      }
    }
  }
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [runtimeOptions, setRuntimeOptions] = useState<RuntimeOptions | null>(null);
  const [modelClass, setModelClass] = useState("general");
  const [modelOverride, setModelOverride] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [ragStatus, setRagStatus] = useState<string>("disabled");
  const [ragError, setRagError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getRuntimeOptionsTyped(), getMyPreferences()])
      .then(([runtime, prefs]) => {
        setRuntimeOptions(runtime);
        setModelClass(prefs.model_class ?? "general");
        setModelOverride(prefs.model_override ?? "");
      })
      .catch(() => setRuntimeOptions(null));
  }, []);

  const availableModels = useMemo(() => {
    if (!runtimeOptions) return [];
    return runtimeOptions.model_allowlist[modelClass] ?? [];
  }, [runtimeOptions, modelClass]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || loading) return;

    setError("");
    setCitations([]);
    setRagStatus("disabled");
    setRagError(null);

    const userMessage: ChatMessage = { role: "user", content: input.trim() };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    let assistantBuffer = "";
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      await streamChat({
        messages: nextMessages.map((message) => ({ role: message.role, content: message.content })),
        modelClass,
        modelOverride: modelOverride || undefined,
        onMeta: (meta) => {
          setCitations(meta.citations ?? []);
          setRagStatus(meta.rag_status ?? "disabled");
          setRagError(meta.rag_error ?? null);
        },
        onToken: (token) => {
          assistantBuffer += token;
          setMessages((prev) => {
            const cloned = [...prev];
            cloned[cloned.length - 1] = { role: "assistant", content: assistantBuffer };
            return cloned;
          });
        },
        onDone: () => undefined,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Chat Workspace</p>
        <h1 className="page-title">Stream responses with model and RAG visibility.</h1>
        <p className="page-description">
          Select a model class, optionally override with a specific runtime model, and inspect citations from retrieval.
        </p>
      </section>

      <section className="surface stack" aria-label="Chat runtime controls">
        <div className="row">
          <label className="field-label" style={{ minWidth: "210px" }}>
            Model class
            <select className="select" value={modelClass} onChange={(e) => setModelClass(e.target.value)}>
              <option value="general">general</option>
              <option value="vision">vision</option>
              <option value="embedding">embedding</option>
              <option value="code">code</option>
            </select>
          </label>
          <label className="field-label" style={{ minWidth: "260px" }}>
            Model override
            <select className="select" value={modelOverride} onChange={(e) => setModelOverride(e.target.value)}>
              <option value="">(class default)</option>
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
          <span className={ragError ? "pill error" : ragStatus === "ready" ? "pill success" : "pill"}>
            RAG {ragStatus}
          </span>
        </div>
        {ragError ? <p className="error-text">{ragError}</p> : null}
      </section>

      <section className="surface stack" aria-label="Chat history">
        <div className="message-feed">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p style={{ margin: 0 }}>No messages yet. Ask a question to start a streamed response.</p>
            </div>
          ) : null}
          {messages.map((message, idx) => (
            <article key={`${message.role}-${idx}`} className={message.role === "user" ? "message user" : "message"}>
              <p className="message-role">{message.role}</p>
              <p className="message-text">{message.content || (loading ? "..." : "(empty response)")}</p>
            </article>
          ))}
        </div>

        <form onSubmit={onSubmit} className="stack">
          <label className="field-label">
            Prompt
            <textarea
              className="textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask BLAIRE to analyze, summarize, or reason through a task..."
              disabled={loading}
            />
          </label>
          <div className="toolbar">
            <button type="submit" className="button button-primary" disabled={loading || !input.trim()}>
              {loading ? "Streaming response..." : "Send message"}
            </button>
            <p className="help-text">Press Enter inside the field to submit.</p>
          </div>
        </form>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="surface stack" aria-label="Citation panel">
        <h2>Citations</h2>
        {citations.length === 0 ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>No citations returned for the latest response.</p>
          </div>
        ) : (
          <div className="panel-list">
            {citations.map((citation, idx) => (
              <article key={`${citation.source_name}-${citation.chunk_index}-${idx}`} className="surface" style={{ padding: "12px" }}>
                <p style={{ marginBottom: "6px" }}>
                  <strong>{citation.source_name ?? "Unknown source"}</strong> {citation.file_type ? `(${citation.file_type})` : ""}
                </p>
                <p className="help-text" style={{ marginBottom: "6px" }}>
                  chunk #{citation.chunk_index ?? "n/a"} | score {typeof citation.score === "number" ? citation.score.toFixed(3) : "n/a"}
                </p>
                {citation.source_path ? (
                  citation.source_path.startsWith("http://") || citation.source_path.startsWith("https://") ? (
                    <a href={citation.source_path} target="_blank" rel="noreferrer" className="mono" style={{ fontSize: "0.82rem" }}>
                      {citation.source_path}
                    </a>
                  ) : (
                    <p className="mono" style={{ margin: "0 0 8px", fontSize: "0.82rem" }}>
                      {citation.source_path}
                    </p>
                  )
                ) : null}
                {citation.ingested_at ? (
                  <p className="help-text" style={{ marginBottom: "8px" }}>
                    Ingested: {new Date(citation.ingested_at).toLocaleString()}
                  </p>
                ) : null}
                <p style={{ margin: 0 }}>{String(citation.text ?? "").slice(0, 240)}</p>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
