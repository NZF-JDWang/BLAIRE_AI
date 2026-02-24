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

## Initial Tool Set (v1)

- local_search
- web_search
- check_disk_space
- check_cpu_mem
- check_gpu_status
- check_docker_containers
- check_endpoint

---

## Tool Invocation Flow

LLM outputs structured intent:
{
  "tool": "check_disk_space",
  "args": {}
}

Orchestrator validates:
- Allowed?
- Rate limited?
- Within autonomy level?

Then executes safely.