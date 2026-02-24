## Memory Philosophy

BLAIRE stores unlimited memory on disk but feeds the model a curated subset per request.

Memory is divided into:

1. Short-Term Memory (Conversation)
2. Medium-Term Memory (Session summaries)
3. Long-Term Memory (Facts, Projects, Preferences)

---

## Directory Structure

/data/
  profile.json
  preferences.json
  projects.json
  todos.json
  sessions/
  episodic/
  long_term/
    facts.jsonl
    lessons.jsonl

---

## Profile Memory

Contains stable identity and environment facts.

Fields:
- name
- environment_summary
- long_term_goals
- behavioral_constraints

---

## Preferences Memory

Contains:
- response_style
- autonomy_level
- notification_rules
- quiet_hours

---

## Projects Memory

Each project:
- id
- name
- description
- status
- priority
- summary_card

---

## TODO Memory

Each todo:
- id
- project_id
- title
- description
- priority
- status
- created_at
- last_updated

---

## Long-Term Fact Entry

Each line in facts.jsonl:

{
  "id": "mem_xxx",
  "type": "user_fact | system_fact | lesson",
  "text": "...",
  "tags": [],
  "importance": 0-1,
  "created_at": "",
  "last_used": ""
}

---

## Episodic Memory

One markdown file per day summarizing:
- Decisions
- Actions taken
- Notable observations