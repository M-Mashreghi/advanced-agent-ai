import json
from config import MAX_GAME_TURNS


def new_game_state():
    return {
        "active": True,
        "finished": False,
        "won": False,
        "turn": 1,
        "phase": "NEED_QUESTION",
        "history": [],
        "current_question": "",
        "current_guess": "",
        "strategy": {},
    }


def _history_text(state):
    return "\n".join(
        f'Q{x["turn"]}: {x.get("question","")} -> {x.get("answer","")}; Guess: {x.get("guess","")} -> {x.get("guess_result","")}'
        for x in state["history"]
    )


class TwentyQuestionsAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_question(self, state):
        if state["turn"] > MAX_GAME_TURNS:
            raise RuntimeError("Maximum number of turns reached.")
        previous = [x.get("question", "") for x in state["history"] if x.get("question")]
        prompt = f"""
You are playing 20 Questions and must identify one unknown word.
Ask ONE short yes/no question.
Strategy:
- maximize information gain: prefer questions that split plausible possibilities broadly,
- start with broad categories then narrow to properties,
- never repeat/paraphrase earlier questions,
- do not make the guess inside the question.

History:\n{_history_text(state)}
Previous questions: {json.dumps(previous, ensure_ascii=False)}

Return ONLY JSON:
{{"question":"...?","strategy":{{"focus":"...","estimated_split_quality":0.0,"belief_summary":"..."}}}}
"""
        raw = self.llm.chat([{"role":"user","content":prompt}], json_mode=True, temperature=0.3)
        data = json.loads(raw)
        return {
            "question": str(data.get("question", "Is it a living thing?")).strip(),
            "strategy": data.get("strategy", {}),
        }

    def generate_guess(self, state, question_answer):
        guesses = [x.get("guess", "") for x in state["history"] if x.get("guess")]
        prompt = f"""
You are playing 20 Questions. Make ONE specific word guess using the complete history and latest answer.
Never repeat a previous guess.

History:\n{_history_text(state)}
Current question: {state["current_question"]}
Current answer: {question_answer}
Previous guesses: {json.dumps(guesses, ensure_ascii=False)}

Return ONLY JSON:
{{"guess":"one word","confidence":0.0,"belief_summary":"short category-level belief"}}
"""
        raw = self.llm.chat([{"role":"user","content":prompt}], json_mode=True, temperature=0.25)
        data = json.loads(raw)
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except Exception:
            confidence = 0.0
        return {
            "guess": str(data.get("guess", "object")).strip(),
            "confidence": confidence,
            "belief_summary": str(data.get("belief_summary", "")).strip(),
        }

    def record_turn(self, state, answer, guess, guess_correct):
        state["history"].append({
            "turn": state["turn"],
            "question": state["current_question"],
            "answer": answer,
            "guess": guess,
            "guess_result": "correct" if guess_correct else "wrong",
        })
        if guess_correct:
            state.update({"finished":True,"won":True,"active":False,"phase":"FINISHED"})
            return state
        if state["turn"] >= MAX_GAME_TURNS:
            state.update({"finished":True,"won":False,"active":False,"phase":"FINISHED"})
            return state
        state["turn"] += 1
        state["current_question"] = ""
        state["current_guess"] = ""
        state["phase"] = "NEED_QUESTION"
        return state
