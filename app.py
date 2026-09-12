from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from config import CHAT_MODEL, EMBED_MODEL, EXA_API_KEY, MAX_GAME_TURNS
from core.ollama_client import OllamaClient
from core.router import detect_intent
from core.memory import compact_history
from rag.pdf_loader import extract_pages
from rag.hybrid_rag import HybridRAG, chunk_pages, corpus_hash
from tools.exa_search import ExaSearch
from tools.executor import execute_tools, generate_final_answer
from verification.verifier import verify_and_score, repair_answer
from game.twenty_questions import TwentyQuestionsAgent, new_game_state

st.set_page_config(page_title="Advanced Agent AI", page_icon="🤖", layout="wide")
st.title("🤖 Advanced Agent AI")
st.caption("Ollama + Hybrid RAG + Exa Web Search + Function Calling + 20 Questions")

@st.cache_resource
def get_llm():
    return OllamaClient()

llm = get_llm()
web = ExaSearch(llm)
game_agent = TwentyQuestionsAgent(llm)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""
if "rag" not in st.session_state:
    st.session_state.rag = HybridRAG(llm)
if "corpus_id" not in st.session_state:
    st.session_state.corpus_id = None
if "game" not in st.session_state:
    st.session_state.game = new_game_state()
    st.session_state.game["active"] = False
    st.session_state.game["finished"] = False

with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        cid = corpus_hash(uploaded_files)
        if cid != st.session_state.corpus_id:
            with st.spinner("Preparing hybrid PDF index..."):
                rag = HybridRAG(llm)
                if not rag.load(cid):
                    pages = []
                    for file in uploaded_files:
                        pages.extend(extract_pages(file))
                    rag.build(chunk_pages(pages), cid)
                st.session_state.rag = rag
                st.session_state.corpus_id = cid

        if st.session_state.rag.ready():
            st.success("RAG ready")
            st.write(f"**Chunks:** {len(st.session_state.rag.chunks)}")
            st.write("**Dense:** FAISS IndexFlatL2")
            st.write("**Sparse:** BM25")
            st.write("**Injected evidence:** Top 3")

    st.divider()
    st.header("⚙️ Status")
    st.write(f"**LLM:** `{CHAT_MODEL}`")
    st.write(f"**Embedding:** `{EMBED_MODEL}`")
    st.write("**Exa:** " + ("✅ configured" if EXA_API_KEY else "❌ add EXA_API_KEY to .env"))

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.rerun()

chat_tab, game_tab = st.tabs(["💬 Agent Chat", "🎯 20 Questions"])

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_text = st.chat_input("Ask the agent something...")

    if user_text:
        st.session_state.messages.append({"role":"user","content":user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.spinner("Detecting intent..."):
            decision = detect_intent(llm, user_text, st.session_state.rag.ready())

        if decision["intent"] == "GAME20Q":
            st.session_state.game = new_game_state()
            answer = "20 Questions mode is ready. Open the **20 Questions** tab and choose a word in your mind."
            tool_output = {"rag_results":[],"web_results":[],"document_context":"","web_context":"","calculation":None,"tool_log":[]}
            verification = {"confidence":decision["confidence"],"supported":None}
        else:
            with st.spinner("Executing tools..."):
                tool_output = execute_tools(
                    llm=llm,
                    decision=decision,
                    user_text=user_text,
                    rag_engine=st.session_state.rag,
                    web_search=web,
                    conversation_summary=st.session_state.conversation_summary,
                    recent_messages=st.session_state.messages,
                )

            with st.spinner("Generating grounded answer..."):
                answer = generate_final_answer(
                    llm=llm,
                    user_text=user_text,
                    decision=decision,
                    tool_output=tool_output,
                    conversation_summary=st.session_state.conversation_summary,
                    recent_messages=st.session_state.messages,
                )

            with st.spinner("Verifying answer..."):
                verification = verify_and_score(llm, user_text, answer, tool_output)
                if verification.get("supported") is False:
                    answer = repair_answer(llm, user_text, answer, tool_output, verification)
                    verification = verify_and_score(llm, user_text, answer, tool_output)

        with st.chat_message("assistant"):
            st.markdown(answer)
            c1, c2, c3 = st.columns(3)
            c1.metric("Intent", decision["intent"])
            c2.metric("Router confidence", f'{decision["confidence"]:.0%}')
            conf = verification.get("confidence")
            c3.metric("Answer confidence", "N/A" if conf is None else f"{conf:.0%}")

            if tool_output["rag_results"]:
                with st.expander("📚 Document evidence"):
                    for i, item in enumerate(tool_output["rag_results"], start=1):
                        st.markdown(f'### Doc {i} — {item["filename"]}, page {item["page"]}')
                        st.write(f'Hybrid score: {item["hybrid_score"]:.3f}')
                        st.text(item["text"][:1200])

            if tool_output["web_results"]:
                with st.expander("🌐 Exa web evidence"):
                    for i, item in enumerate(tool_output["web_results"], start=1):
                        st.markdown(f'### Web {i} — {item["title"]}')
                        st.write(item["summary"])
                        st.write(item["url"])

            if tool_output["calculation"]:
                with st.expander("🧮 Function call"):
                    st.code(f'{tool_output["calculation"]["expression"]} = {tool_output["calculation"]["result"]}')

            with st.expander("🔬 Agent debug"):
                st.write("**Intent decision**")
                st.json(decision)
                st.write("**Tool log**")
                st.json(tool_output["tool_log"])
                st.write("**Verification**")
                st.json(verification)

        st.session_state.messages.append({"role":"assistant","content":answer})
        summary, compacted = compact_history(llm, st.session_state.messages, st.session_state.conversation_summary)
        st.session_state.conversation_summary = summary
        st.session_state.messages = compacted

with game_tab:
    state = st.session_state.game
    st.subheader("20 Questions")
    st.write(f"Think of one word. The model has at most **{MAX_GAME_TURNS} turns** to guess it.")

    if not state["active"] and not state["finished"]:
        if st.button("▶️ Start Game", type="primary"):
            st.session_state.game = new_game_state()
            st.rerun()

    state = st.session_state.game

    if state["active"]:
        st.progress(min(state["turn"] / MAX_GAME_TURNS, 1.0))
        st.write(f"**Turn {state['turn']} / {MAX_GAME_TURNS}**")

        if state["phase"] == "NEED_QUESTION":
            with st.spinner("Choosing a high-information question..."):
                item = game_agent.generate_question(state)
            state["current_question"] = item["question"]
            state["strategy"] = item.get("strategy", {})
            state["phase"] = "WAITING_QUESTION_ANSWER"
            st.session_state.game = state
            st.rerun()

        if state["phase"] == "WAITING_QUESTION_ANSWER":
            st.info(f'**Question:** {state["current_question"]}')
            with st.expander("Strategy state"):
                st.json(state.get("strategy", {}))
            yes_col, no_col, exit_col = st.columns(3)
            if yes_col.button("Yes", key=f'q_yes_{state["turn"]}'):
                state["pending_answer"] = "Yes"; state["phase"] = "NEED_GUESS"; st.session_state.game = state; st.rerun()
            if no_col.button("No", key=f'q_no_{state["turn"]}'):
                state["pending_answer"] = "No"; state["phase"] = "NEED_GUESS"; st.session_state.game = state; st.rerun()
            if exit_col.button("Exit Game", key=f'exit_q_{state["turn"]}'):
                state.update({"active":False,"finished":True,"won":False,"phase":"FINISHED"}); st.session_state.game = state; st.rerun()

        if state["phase"] == "NEED_GUESS":
            with st.spinner("Making a guess..."):
                guess_data = game_agent.generate_guess(state, state["pending_answer"])
            state["current_guess"] = guess_data["guess"]
            state["guess_confidence"] = guess_data["confidence"]
            state["belief_summary"] = guess_data["belief_summary"]
            state["phase"] = "WAITING_GUESS_RESULT"
            st.session_state.game = state
            st.rerun()

        if state["phase"] == "WAITING_GUESS_RESULT":
            st.info(f'Your answer to "{state["current_question"]}" was **{state["pending_answer"]}**.')
            st.success(f'**My guess:** {state["current_guess"]} ({state.get("guess_confidence",0.0):.0%} confidence)')
            correct_col, wrong_col, exit_col = st.columns(3)
            if correct_col.button("Correct", key=f'g_yes_{state["turn"]}'):
                game_agent.record_turn(state, state["pending_answer"], state["current_guess"], True); st.session_state.game = state; st.rerun()
            if wrong_col.button("Wrong", key=f'g_no_{state["turn"]}'):
                game_agent.record_turn(state, state["pending_answer"], state["current_guess"], False); st.session_state.game = state; st.rerun()
            if exit_col.button("Exit Game", key=f'exit_g_{state["turn"]}'):
                state.update({"active":False,"finished":True,"won":False,"phase":"FINISHED"}); st.session_state.game = state; st.rerun()

    if state["finished"]:
        if state["won"]:
            st.success(f'🎉 I guessed the word in {state["turn"]} turn(s)!')
        else:
            st.error("Game finished without a correct guess.")
        if state["history"]:
            st.write("### Game history")
            st.dataframe(state["history"], use_container_width=True)
        if st.button("🔁 New Game"):
            st.session_state.game = new_game_state(); st.rerun()
