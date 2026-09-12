import argparse
import statistics
from core.ollama_client import OllamaClient
from game.twenty_questions import TwentyQuestionsAgent, new_game_state

try:
    from test import ValidatorModel
except ImportError as exc:
    raise SystemExit("Place the TA-provided test.py next to evaluate_20Q.py.") from exc


def yes_no(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes","y","true","1","correct"}:
        return True
    if text in {"no","n","false","0","wrong"}:
        return False
    return text.startswith("y") or text.startswith("t")


def play_one(agent):
    validator = ValidatorModel()
    state = new_game_state()
    for _ in range(20):
        q = agent.generate_question(state)
        state["current_question"] = q["question"]
        answer = "Yes" if yes_no(validator.validate_question(state["current_question"])) else "No"
        g = agent.generate_guess(state, answer)
        correct = yes_no(validator.validate_guess(g["guess"]))
        agent.record_turn(state, answer, g["guess"], correct)
        if correct:
            return True, state["turn"]
        if state["finished"]:
            break
    return False, 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=100, help="Number of games")
    args = parser.parse_args()
    llm = OllamaClient()
    agent = TwentyQuestionsAgent(llm)
    wins, turns = 0, []
    for i in range(args.N):
        won, used = play_one(agent)
        wins += int(won); turns.append(used)
        print(f'Game {i+1:03d}/{args.N}: {"WIN" if won else "LOSS"} in {used} turns')
    rate = wins / args.N if args.N else 0.0
    print("\n=== 20 Questions Evaluation ===")
    print(f"Games: {args.N}")
    print(f"Wins: {wins}")
    print(f"Win rate: {rate:.2%}")
    print(f"Average turns: {statistics.mean(turns):.2f}")
    print(f"Best game: {min(turns)} turns")
    print(f"Worst game: {max(turns)} turns")


if __name__ == "__main__":
    main()
