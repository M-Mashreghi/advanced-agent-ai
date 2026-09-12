from config import MAX_HISTORY_MESSAGES, RECENT_MESSAGES_AFTER_SUMMARY


def compact_history(llm, messages, existing_summary=""):
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return existing_summary, messages
    old = messages[:-RECENT_MESSAGES_AFTER_SUMMARY]
    recent = messages[-RECENT_MESSAGES_AFTER_SUMMARY:]
    transcript = "\n".join(f'{m["role"]}: {m["content"]}' for m in old)
    prompt = f"""
Summarize this conversation for future context. Preserve user goals, important facts,
decisions, unresolved questions, references to documents, and useful tool results.
Do not add facts.

Existing summary:
{existing_summary}

Conversation:
{transcript}
"""
    try:
        summary = llm.chat([{"role":"user","content":prompt}], temperature=0.1).strip()
        return summary, recent
    except Exception:
        return existing_summary, messages
