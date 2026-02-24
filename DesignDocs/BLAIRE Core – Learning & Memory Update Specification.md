## Post-Conversation Learning Routine

After each session or N turns:

Model is prompted to output:

{
  "session_summary": "...",
  "important_facts_about_user": [],
  "lessons_learned": [],
  "new_todos": []
}

---

## Update Process

1. Append summary to session file
2. Append key notes to episodic day file
3. Merge new facts into:
   - profile
   - preferences
   - facts.jsonl
4. Add new todos

---

## Memory Hygiene

- Merge duplicate facts periodically
- Remove low-importance stale memories
- Distill session summaries weekly