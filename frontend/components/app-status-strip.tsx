"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiRequestError, getDependencyStatus, getRuntimeDiagnostics, RuntimeDiagnostics } from "@/lib/api";

type DependencySummary = {
  required_total: number;
  required_ok: number;
  enabled_failures: number;
};

function summarizeDependencies(deps: Awaited<ReturnType<typeof getDependencyStatus>>): DependencySummary {
  const required = deps.dependencies.filter((item) => item.required && item.enabled);
  const requiredOk = required.filter((item) => item.ok).length;
  const enabledFailures = deps.dependencies.filter((item) => item.enabled && !item.ok).length;
  return {
    required_total: required.length,
    required_ok: requiredOk,
    enabled_failures: enabledFailures,
  };
}

export function AppStatusStrip() {
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics | null>(null);
  const [depSummary, setDepSummary] = useState<DependencySummary | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "auth_required" | "error">("loading");

  const refresh = useCallback(async () => {
    setStatus("loading");
    try {
      const [diag, deps] = await Promise.all([getRuntimeDiagnostics(), getDependencyStatus()]);
      setDiagnostics(diag);
      setDepSummary(summarizeDependencies(deps));
      setStatus("ready");
    } catch (err) {
      if (err instanceof ApiRequestError && (err.status === 401 || err.status === 403)) {
        setStatus("auth_required");
        setDiagnostics(null);
        setDepSummary(null);
        return;
      }
      setStatus("error");
      setDiagnostics(null);
      setDepSummary(null);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refresh();
    }, 0);
    const onKeyChanged = () => void refresh();
    window.addEventListener("blaire-api-key-changed", onKeyChanged);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("blaire-api-key-changed", onKeyChanged);
    };
  }, [refresh]);

  const depsLabel = useMemo(() => {
    if (!depSummary) return "deps unknown";
    return `${depSummary.required_ok}/${depSummary.required_total} required deps healthy`;
  }, [depSummary]);

  if (status === "loading") {
    return (
      <div className="app-status-strip">
        <span className="pill">Checking runtime...</span>
      </div>
    );
  }

  if (status === "auth_required") {
    return (
      <div className="app-status-strip">
        <span className="pill warn">Auth required</span>
        <Link href="/settings" className="button button-muted">
          Open settings
        </Link>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="app-status-strip">
        <span className="pill error">Runtime status unavailable</span>
        <button type="button" className="button button-muted" onClick={() => void refresh()}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="app-status-strip">
      <span className="pill success">role: {diagnostics?.role}</span>
      <span className="pill mono">search: {diagnostics?.effective_search_mode_default}</span>
      <span className={diagnostics?.effective_sensitive_actions_enabled ? "pill warn" : "pill"}>
        sensitive actions: {diagnostics?.effective_sensitive_actions_enabled ? "on" : "off"}
      </span>
      <span className={depSummary && depSummary.enabled_failures > 0 ? "pill error" : "pill success"}>{depsLabel}</span>
      <button type="button" className="button button-muted" onClick={() => void refresh()}>
        Refresh
      </button>
    </div>
  );
}
