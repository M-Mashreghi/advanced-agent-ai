import json

ALLOWED = {"CHAT", "RAG", "WEB", "RAG_WEB", "CALCULATOR", "RAG_CALCULATOR", "GAME20Q"}


def detect_intent(llm, user_text, has_documents):
    prompt = f"""
You are the intent controller of an AI agent.
Choose one intent: CHAT, RAG, WEB, RAG_WEB, CALCULATOR, RAG_CALCULATOR, GAME20Q.

Definitions:
- CHAT: general conversation/knowledge.
- RAG: answer from uploaded PDFs.
- WEB: current or recent online information.
- RAG_WEB: compare/combine uploaded PDFs with current web information.
- CALCULATOR: numerical calculation.
- RAG_CALCULATOR: retrieve numbers from PDFs and then calculate.
- GAME20Q: start/continue/exit the 20 Questions game.

Uploaded documents available: {has_documents}

Return ONLY JSON:
{{"intent":"CHAT","needs_rag":false,"needs_web":false,"needs_calculator":false,"game_action":"NONE","confidence":0.0}}

game_action: NONE, START, CONTINUE, EXIT

User: {user_text}
"""
    fallback = {"intent":"CHAT","needs_rag":False,"needs_web":False,"needs_calculator":False,"game_action":"NONE","confidence":0.0}
    try:
        raw = llm.chat([{"role":"user","content":prompt}], json_mode=True, temperature=0.0)
        data = json.loads(raw)
        intent = str(data.get("intent", "CHAT")).upper()
        if intent not in ALLOWED:
            intent = "CHAT"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        game_action = str(data.get("game_action", "NONE")).upper()
        if game_action not in {"NONE", "START", "CONTINUE", "EXIT"}:
            game_action = "NONE"
        return {
            "intent": intent,
            "needs_rag": intent in {"RAG", "RAG_WEB", "RAG_CALCULATOR"},
            "needs_web": intent in {"WEB", "RAG_WEB"},
            "needs_calculator": intent in {"CALCULATOR", "RAG_CALCULATOR"},
            "game_action": game_action,
            "confidence": confidence,
        }
    except Exception:
        return fallback
