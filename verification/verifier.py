import json


def _retrieval_score(tool_output):
    scores = []
    for item in tool_output.get("rag_results", []):
        scores.append(float(item.get("hybrid_score", 0.0)))
    for item in tool_output.get("web_results", []):
        score = (float(item.get("score", 0.0)) + 1.0) / 2.0
        scores.append(max(0.0, min(1.0, score)))
    if not scores:
        return None
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def verify_and_score(llm, user_question, answer, tool_output):
    evidence = "\n\n".join(x for x in [tool_output.get("document_context", ""), tool_output.get("web_context", "")] if x)
    calc = tool_output.get("calculation")
    if calc:
        evidence += f'\n\nCalculator: {calc["expression"]} = {calc["result"]}'

    if not evidence.strip():
        return {"supported":None,"verification_score":None,"retrieval_score":None,"confidence":None,"unsupported_claims":[],"citation_quality":None}

    prompt = f"""
Verify whether the answer is supported by the supplied evidence.
Check factual grounding, calculations, and whether [Doc N]/[Web N] citations are used appropriately.
Return ONLY JSON:
{{"supported":true,"support_score":0.0,"citation_quality":0.0,"unsupported_claims":[]}}
Scores are between 0 and 1.

Question: {user_question}
Answer: {answer}
Evidence:\n{evidence}
"""
    try:
        raw = llm.chat([{"role":"user","content":prompt}], json_mode=True, temperature=0.0)
        data = json.loads(raw)
        support = max(0.0, min(1.0, float(data.get("support_score", 0.0))))
        citation = max(0.0, min(1.0, float(data.get("citation_quality", 0.0))))
        retrieval = _retrieval_score(tool_output)
        confidence = (0.75 * support + 0.25 * citation) if retrieval is None else (0.45 * support + 0.35 * retrieval + 0.20 * citation)
        return {
            "supported": bool(data.get("supported")),
            "verification_score": support,
            "retrieval_score": retrieval,
            "confidence": max(0.0, min(1.0, confidence)),
            "unsupported_claims": data.get("unsupported_claims", []),
            "citation_quality": citation,
        }
    except Exception:
        return {"supported":None,"verification_score":None,"retrieval_score":_retrieval_score(tool_output),"confidence":None,"unsupported_claims":[],"citation_quality":None}


def repair_answer(llm, user_question, answer, tool_output, verification):
    if verification.get("supported") is not False:
        return answer
    evidence = "\n\n".join(x for x in [tool_output.get("document_context", ""), tool_output.get("web_context", "")] if x)
    calc = tool_output.get("calculation")
    if calc:
        evidence += f'\n\nCalculator: {calc["expression"]} = {calc["result"]}'
    prompt = f"""
Rewrite the answer so every factual claim is supported by the evidence.
Remove/correct unsupported claims and preserve [Doc N]/[Web N] citations.

Question: {user_question}
Original answer: {answer}
Unsupported claims: {verification.get("unsupported_claims", [])}
Evidence:\n{evidence}
"""
    try:
        return llm.chat([{"role":"user","content":prompt}], temperature=0.1)
    except Exception:
        return answer
