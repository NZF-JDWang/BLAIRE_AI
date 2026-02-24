## Context Budget Strategy

Each model call has a fixed token allocation divided across sections.

### Recommended Allocation (Small Models)

- Soul / Rules: 300 tokens
- Profile + Preferences: 300 tokens
- Project Cards: 500 tokens
- Long-Term Facts: 400 tokens
- Session Summary + Recent Turns: 800 tokens
- Tool Results: dynamic
- User Input: remainder

Total target: <= 4k tokens

---

## Retrieval Strategy

1. Extract topics from user input
2. Retrieve relevant:
   - Project cards
   - Fact entries
   - Lessons
3. Rank by:
   - Relevance
   - Importance
   - Recency
4. Hard cap total memory tokens

---

## Two-Pass Model Strategy

1. Planning Pass
   - Identify topics
   - Identify needed tools
   - Identify relevant memory types

1. Response Pass
   - Build curated context
   - Generate answer