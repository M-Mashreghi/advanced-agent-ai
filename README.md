# Advanced Agent AI

An advanced extension of the Deep Learning final **Agent AI** project, built around a local Ollama model.

## Core assignment requirements

- Conversation history and context-window management
- PDF RAG with persistent FAISS indexing
- Web-search trigger and Exa integration
- Function calling
- 20 Questions game
- `evaluate_20Q.py`
- Streamlit GUI

## Added extensions

- Advanced intent routing with confidence
- Multi-PDF, page-aware RAG
- Query rewriting for follow-up questions
- Hybrid retrieval: FAISS + BM25
- LLM reranking
- RAG + Web combined mode
- Safe Calculator tool
- RAG + Calculator multi-tool mode
- Answer verification and automatic repair
- Evidence-based confidence score
- Structured 20 Questions strategy

## Architecture

```text
                           User
                            |
                            v
                    Intent Controller
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
      CHAT                  RAG               20Q GAME
                            |
                  +---------+---------+
                  |                   |
                  v                   v
               FAISS                 BM25
                  |                   |
                  +---------+---------+
                            |
                       Reranking
                            |
                         Top 3
                            |
                 +----------+----------+
                 |                     |
                 v                     v
              Exa Web             Calculator
                 |                     |
                 +----------+----------+
                            |
                            v
                         Gemma 3
                            |
                            v
                     Answer Verifier
                            |
                            v
                 Answer + Sources + Score
```

## Setup

Recommended: Python 3.11.

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Exa key:

```text
EXA_API_KEY=your_key_here
```

Run:

```bash
python -m streamlit run app.py
```

## Example prompts

```text
According to the uploaded paper, what is the proposed methodology?
```

```text
Compare the method in my uploaded paper with recent approaches online.
```

```text
Find the three accuracy values reported in the PDF and calculate their average.
```

```text
What's the latest news about local LLMs?
```

```text
Let's play 20 questions.
```

## 20 Questions evaluation

Place the TA-provided `test.py` in the project root and run:

```bash
python evaluate_20Q.py -N 100
```

The evaluator reports games, wins, win rate, average turns, best game, and worst game.

## Notes

- The default `gemma3:4b` is below the assignment's 7B-parameter limit.
- PDF indexes are persisted under `cache/` and reused.
- RAG uses `IndexFlatL2` plus BM25 and injects exactly three final passages.
- Exa retrieves a larger candidate set, then local embeddings select three web results before summarization.
- Calculator execution uses a restricted AST evaluator instead of Python `eval()`.
