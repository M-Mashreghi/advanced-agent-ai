from rag.query_rewriter import rewrite_query
from tools.calculator import expression_from_request, safe_calculate


def execute_tools(llm, decision, user_text, rag_engine=None, web_search=None, conversation_summary="", recent_messages=None):
    recent_messages = recent_messages or []
    rag_results, web_results, calculation, tool_log = [], [], None, []
    document_context = web_context = ""

    if decision["needs_rag"]:
        if rag_engine is None or not rag_engine.ready():
            tool_log.append({"tool":"rag","ok":False,"error":"No PDF index is available."})
        else:
            query = rewrite_query(llm, user_text, conversation_summary, recent_messages)
            rag_results = rag_engine.search(query)
            document_context = rag_engine.build_context(rag_results)
            tool_log.append({"tool":"rag","ok":True,"query":query,"results":len(rag_results)})

    if decision["needs_web"]:
        if web_search is None or not web_search.available():
            tool_log.append({"tool":"web","ok":False,"error":"Exa API key is not configured."})
        else:
            web_results = web_search.search(user_text, num_results=10, top_k=3)
            web_context = web_search.build_context(web_results)
            tool_log.append({"tool":"web","ok":True,"results":len(web_results)})

    evidence = "\n\n".join(x for x in [document_context, web_context] if x)

    if decision["needs_calculator"]:
        try:
            expression = expression_from_request(llm, user_text, evidence=evidence)
            if not expression:
                raise ValueError("Could not construct a calculation from the request/evidence.")
            value = safe_calculate(expression)
            calculation = {"expression":expression,"result":value}
            tool_log.append({"tool":"calculator","ok":True,"expression":expression,"result":value})
        except Exception as exc:
            tool_log.append({"tool":"calculator","ok":False,"error":str(exc)})

    return {
        "rag_results": rag_results,
        "web_results": web_results,
        "document_context": document_context,
        "web_context": web_context,
        "calculation": calculation,
        "tool_log": tool_log,
    }


def generate_final_answer(llm, user_text, decision, tool_output, conversation_summary="", recent_messages=None):
    recent_messages = recent_messages or []
    recent = "\n".join(f'{m["role"]}: {m["content"]}' for m in recent_messages[-6:])
    calc_text = ""
    if tool_output["calculation"]:
        calc_text = f'Expression: {tool_output["calculation"]["expression"]}\nResult: {tool_output["calculation"]["result"]}'

    evidence_based = decision["needs_rag"] or decision["needs_web"] or decision["needs_calculator"]
    if not evidence_based:
        prompt = f"""
You are a helpful local AI assistant.
Conversation summary: {conversation_summary}
Recent conversation:\n{recent}
User: {user_text}
Answer clearly and concisely.
"""
    else:
        prompt = f"""
You are a grounded AI agent that has executed tools.
Rules:
1. Do not invent unsupported facts.
2. Cite document claims with [Doc N].
3. Cite web claims with [Web N].
4. Use calculator output exactly.
5. If evidence/tool output is insufficient, say so.
6. Clearly distinguish uploaded-document evidence from web evidence.

User request:
{user_text}

DOCUMENT EVIDENCE:
{tool_output["document_context"]}

WEB EVIDENCE:
{tool_output["web_context"]}

CALCULATOR OUTPUT:
{calc_text}
"""

    return llm.chat([
        {"role":"system","content":"You are an accurate tool-using AI agent."},
        {"role":"user","content":prompt},
    ], temperature=0.2)
