"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { getRuntimeOptionsTyped, RuntimeOptions } from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE = "/api";

async function streamChat(params: {
  messages: Array<{ role: "user" | "assistant" | "system"; content: string }>;
  modelClass: string;
  modelOverride?: string;
  onToken: (token: string) => void;
  onMeta: (meta: { citations?: Array<{ source_name?: string; chunk_index?: number; text?: string }> }) => void;
  onDone: () => void;
}): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: params.messages,
      model_class: params.modelClass,
      model_override: params.modelOverride || null,
      stream: true
    })
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
      let payload: { text?: string; message?: string; citations?: Array<{ source_name?: string; chunk_index?: number; text?: string }> } = {};
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
  const [citations, setCitations] = useState<Array<{ source_name?: string; chunk_index?: number; text?: string }>>([]);

  useEffect(() => {
    getRuntimeOptionsTyped()
      .then((data) => setRuntimeOptions(data))
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
        onMeta: (meta) => setCitations(meta.citations ?? []),
        onToken: (token) => {
          assistantBuffer += token;
          setMessages((prev) => {
            const cloned = [...prev];
            cloned[cloned.length - 1] = { role: "assistant", content: assistantBuffer };
            return cloned;
          });
        },
        onDone: () => undefined
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: "960px", margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "8px" }}>Chat</h1>
      <p style={{ marginBottom: "16px" }}>Streaming chat endpoint with runtime-selectable model classes.</p>

      <div style={{ display: "flex", gap: "12px", marginBottom: "16px", flexWrap: "wrap" }}>
        <label>
          Model class
          <select value={modelClass} onChange={(e) => setModelClass(e.target.value)} style={{ marginLeft: "8px" }}>
            <option value="general">general</option>
            <option value="vision">vision</option>
            <option value="embedding">embedding</option>
            <option value="code">code</option>
          </select>
        </label>
        <label>
          Override
          <select
            value={modelOverride}
            onChange={(e) => setModelOverride(e.target.value)}
            style={{ marginLeft: "8px" }}
          >
            <option value="">(class default)</option>
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ border: "1px solid #cbd5e1", borderRadius: "8px", padding: "12px", minHeight: "280px" }}>
        {messages.length === 0 ? <p style={{ opacity: 0.7 }}>No messages yet.</p> : null}
        {messages.map((message, idx) => (
          <div key={`${message.role}-${idx}`} style={{ marginBottom: "12px" }}>
            <strong>{message.role}:</strong> <span>{message.content}</span>
          </div>
        ))}
      </div>

      <form onSubmit={onSubmit} style={{ marginTop: "12px", display: "flex", gap: "8px" }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask BLAIRE..."
          style={{ flex: 1, padding: "10px", border: "1px solid #94a3b8", borderRadius: "6px" }}
        />
        <button type="submit" disabled={loading} style={{ padding: "10px 16px" }}>
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
      {error ? <p style={{ color: "#b91c1c", marginTop: "8px" }}>{error}</p> : null}
      {citations.length > 0 ? (
        <section style={{ marginTop: "16px" }}>
          <h2 style={{ fontSize: "1.1rem", marginBottom: "8px" }}>Citations</h2>
          <ul>
            {citations.map((citation, idx) => (
              <li key={`${citation.source_name}-${citation.chunk_index}-${idx}`}>
                <strong>{citation.source_name}</strong> #{citation.chunk_index}:{" "}
                {String(citation.text ?? "").slice(0, 180)}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </main>
  );
}
