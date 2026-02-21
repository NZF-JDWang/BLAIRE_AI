"use client";

import { FormEvent, useEffect, useState } from "react";

import { apiFetch, formatApiError, getMyPreferences } from "@/lib/api";

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

async function streamChat(params: {
  messages: Array<{ role: "user" | "assistant" | "system"; content: string }>;
  temperature: number;
  topP: number;
  maxTokens: number | null;
  useRag: boolean;
  retrievalK: number;
  onToken: (token: string) => void;
  onMeta: (meta: { citations?: Citation[]; rag_status?: string; rag_error?: string | null }) => void;
  onDone: () => void;
}): Promise<void> {
  const response = await apiFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: params.messages,
      stream: true,
      temperature: params.temperature,
      top_p: params.topP,
      max_tokens: params.maxTokens,
      use_rag: params.useRag,
      retrieval_k: params.retrievalK,
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

type ReasoningDepth = "fast" | "balanced" | "deep";

function depthSettings(depth: ReasoningDepth): { temperature: number; topP: number; maxTokens: number | null } {
  if (depth === "fast") {
    return { temperature: 0.3, topP: 0.9, maxTokens: 768 };
  }
  if (depth === "deep") {
    return { temperature: 0.6, topP: 0.95, maxTokens: 2048 };
  }
  return { temperature: 0.7, topP: 1.0, maxTokens: null };
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [retrievalK, setRetrievalK] = useState(4);
  const [reasoningDepth, setReasoningDepth] = useState<ReasoningDepth>("balanced");
  const [toolsEnabled, setToolsEnabled] = useState(true);
  const [approvalMode, setApprovalMode] = useState<"ask" | "auto_safe">("ask");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [ragStatus, setRagStatus] = useState<string>("disabled");
  const [ragError, setRagError] = useState<string | null>(null);

  useEffect(() => {
    getMyPreferences()
      .then((prefs) => {
        setRetrievalK(prefs.retrieval_k ?? 4);
      })
      .catch(() => undefined);
  }, []);

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
      const depth = depthSettings(reasoningDepth);
      await streamChat({
        messages: nextMessages.map((message) => ({ role: message.role, content: message.content })),
        temperature: depth.temperature,
        topP: depth.topP,
        maxTokens: depth.maxTokens,
        useRag: true,
        retrievalK,
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
      setError(formatApiError(err, "Request failed"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Chat</p>
        <h1 className="page-title">Single interface for normal work.</h1>
        <p className="page-description">
          Model selection lives in Settings. RAG and citations are always on.
        </p>
      </section>

      <section className="surface stack" aria-label="Chat runtime controls">
        <div className="row">
          <label className="field-label" style={{ minWidth: "200px" }}>
            Reasoning depth
            <select className="select" value={reasoningDepth} onChange={(e) => setReasoningDepth(e.target.value as ReasoningDepth)}>
              <option value="fast">fast</option>
              <option value="balanced">balanced</option>
              <option value="deep">deep</option>
            </select>
          </label>
          <label className="field-label" style={{ minWidth: "180px" }}>
            Tools
            <select
              className="select"
              value={toolsEnabled ? "enabled" : "disabled"}
              onChange={(e) => setToolsEnabled(e.target.value === "enabled")}
              disabled
            >
              <option value="enabled">enabled</option>
              <option value="disabled">disabled</option>
            </select>
          </label>
          <label className="field-label" style={{ minWidth: "220px" }}>
            Approval mode
            <select
              className="select"
              value={approvalMode}
              onChange={(e) => setApprovalMode(e.target.value as "ask" | "auto_safe")}
              disabled
            >
              <option value="ask">ask before actions</option>
              <option value="auto_safe">auto-safe actions</option>
            </select>
          </label>
          <span className="pill success">RAG always on</span>
          <span className={ragError ? "pill error" : ragStatus === "used" ? "pill success" : "pill"}>
            RAG {ragStatus}
          </span>
        </div>
        <p className="help-text">Tools and approval-mode toggles are visible now, but backend chat tool execution wiring is still pending.</p>
        {ragError ? <p className="error-text">{ragError}</p> : null}
      </section>

      <section className="chat-layout" aria-label="Chat and citations">
        <div className="surface stack" aria-label="Chat history">
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
        </div>

        <aside className="surface stack" aria-label="Citation panel">
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
        </aside>
      </section>
    </main>
  );
}
