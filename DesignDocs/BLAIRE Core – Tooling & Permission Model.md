## Tool Design Principles

- No generic shell access
- Each tool is narrowly scoped
- Structured JSON input/output
- Risk-level tagging

---

## Risk Levels

Level 1 – Read-only safe tools
Level 2 – Proposal-only actions
Level 3 – Dangerous (requires confirmation)

---

## Tool Set Status

### Implemented (Level 1 safe/read-only)

- `local_search`
- `web_search`
- `check_disk_space`
- `check_server_health` (configured endpoint telemetry summary)
- `home_assistant_read` (entity/sensor/light state reads)
- `obsidian_search` (vault search/index reads)
- `calendar_summary` (today/next events reads)
- `email_summary` (unread sender/topic summaries)

### Planned / Stubbed

- `check_cpu_mem`
- `check_gpu_status`
- `check_docker_containers` (currently stubbed)
- `check_endpoint`

---

## Tool Invocation Flow

LLM outputs structured intent:

```json
{
  "tool": "check_disk_space",
  "args": {}
}
```

Orchestrator validates:

- Allowed?
- Rate limited?
- Within autonomy level?

Then executes safely.
