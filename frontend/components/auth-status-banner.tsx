"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { ApiRequestError, formatApiError, getBrowserApiKey, getRuntimeOptionsTyped } from "@/lib/api";

type AuthState = "checking" | "ok" | "missing_key" | "forbidden" | "error";

export function AuthStatusBanner() {
  const pathname = usePathname();
  const [state, setState] = useState<AuthState>("checking");
  const [detail, setDetail] = useState("");

  const check = useCallback(async () => {
    const key = getBrowserApiKey().trim();
    if (!key) {
      setState("missing_key");
      setDetail("No API key is set in this browser.");
      return;
    }

    setState("checking");
    setDetail("");
    try {
      await getRuntimeOptionsTyped();
      setState("ok");
      setDetail("");
    } catch (err) {
      if (err instanceof ApiRequestError && err.status === 401) {
        setState("missing_key");
        setDetail("API key missing from requests.");
        return;
      }
      if (err instanceof ApiRequestError && err.status === 403) {
        setState("forbidden");
        setDetail(err.detail ?? "Invalid key or role does not have access.");
        return;
      }
      setState("error");
      setDetail(formatApiError(err, "Unable to verify backend auth"));
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void check();
    }, 0);
    const onKeyChanged = () => void check();
    window.addEventListener("blaire-api-key-changed", onKeyChanged);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("blaire-api-key-changed", onKeyChanged);
    };
  }, [check]);

  if (state === "ok" || state === "checking") {
    return null;
  }

  const settingsLink = (
    <Link href="/settings" className="button button-primary">
      Open Settings
    </Link>
  );

  return (
    <section className="surface stack auth-banner" aria-live="polite">
      <h2>{state === "missing_key" ? "Setup required: add API key" : state === "forbidden" ? "Access denied" : "Connection issue"}</h2>
      <p className="help-text">{detail}</p>
      <div className="toolbar">
        {settingsLink}
        <button type="button" className="button button-muted" onClick={() => void check()}>
          Retry
        </button>
        {pathname !== "/settings" ? <span className="pill warn">Most pages will fail until auth is fixed</span> : null}
      </div>
    </section>
  );
}
