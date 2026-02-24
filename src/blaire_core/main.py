"""Application entrypoint."""

from __future__ import annotations

import argparse
import sys

from blaire_core.config import read_config_snapshot
from blaire_core.interfaces.cli import run_cli
from blaire_core.orchestrator import build_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLAIRE Core CLI")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--heartbeat-interval", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides: dict[str, str] = {}
    if args.data_path:
        overrides["paths.data_root"] = str(args.data_path)
    if args.llm_base_url:
        overrides["llm.base_url"] = str(args.llm_base_url)
    if args.llm_model:
        overrides["llm.model"] = str(args.llm_model)
    if args.heartbeat_interval is not None:
        overrides["heartbeat.interval_seconds"] = str(args.heartbeat_interval)

    snapshot = read_config_snapshot(env=args.env, cli_overrides=overrides)
    if snapshot.effective_config is None:
        print("Config is invalid. CLI will run in diagnostics-only mode.")
        for issue in snapshot.issues:
            print(f"- {issue}")
        # Use fallback by trying dev config again without overrides for context creation.
        fallback_snapshot = read_config_snapshot(env=args.env, cli_overrides={})
        if fallback_snapshot.effective_config is None:
            return 1
        context = build_context(config=fallback_snapshot.effective_config, snapshot=snapshot)
    else:
        context = build_context(config=snapshot.effective_config, snapshot=snapshot)

    run_cli(context, initial_session_id=args.session_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

