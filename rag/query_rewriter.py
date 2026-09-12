import json


def rewrite_query(llm, user_query, conversation_summary="", recent_messages=None):
    recent_messages = recent_messages or []
    recent = "\n".join(f'{m["role"]}: {m["content"]}' for m in recent_messages[-4:])
    prompt = f"""
Rewrite the latest request as a standalone retrieval query.
Resolve references like 'it', 'this paper', or 'those results' from context.
Do not answer. Return ONLY JSON: {{"query":"..."}}

Conversation summary:
{conversation_summary}

Recent conversation:
{recent}

Latest request:
{user_query}
"""
    try:
        raw = llm.chat([{"role":"user","content":prompt}], json_mode=True, temperature=0.0)
        query = str(json.loads(raw).get("query", user_query)).strip()
        return query or user_query
    except Exception:
        return user_query
