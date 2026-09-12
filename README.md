# Advanced Agent AI

This project started as a course project on building an AI agent, but I wanted to push it a bit further than a basic chatbot.

The main idea is simple: instead of sending every question directly to a language model, the system first decides what kind of request it is. Depending on the query, it can answer normally, search uploaded documents, use the web, run a calculation, or switch into a 20 Questions game mode.

The whole system is built around a local LLM running with **Ollama**, so the core language-model inference stays on the local machine.

## What it can do

- Chat with conversation history
- Upload and query multiple PDF files
- Retrieve relevant document chunks using **FAISS + BM25**
- Rewrite follow-up questions before retrieval
- Rerank retrieved chunks before sending them to the model
- Search the web using **Exa**
- Combine document evidence with web evidence
- Call a calculator tool for numerical tasks
- Use more than one tool for the same request when needed
- Show document sources and page numbers
- Verify generated answers against retrieved evidence
- Provide a simple confidence score
- Play an interactive **20 Questions** game
- Evaluate the 20 Questions agent with the provided validator script

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

## Why I built it this way

A plain LLM chatbot is useful, but it quickly runs into a few problems: it does not know what is inside local documents, it may not have up-to-date information, and it is not always reliable for calculations or source-based answers.

This project tries to handle those cases explicitly.

For example, if a question refers to an uploaded paper, the agent searches the document first. If the question asks about something current, it can use web search. If the request involves arithmetic, the calculation is done by a dedicated Python tool instead of relying on the language model to do the math itself.

## RAG pipeline

The document pipeline is:

```text
PDF
 |
 v
Text extraction
 |
 v
Chunking
 |
 +--------------------+
 |                    |
 v                    v
FAISS                BM25
 |                    |
 +---------+----------+
           |
           v
      Hybrid ranking
           |
           v
      LLM reranking
           |
           v
        Top 3
           |
           v
        Gemma 3
```

Each retrieved chunk keeps its original file name and page number, so answers can point back to the source.

## Web search

For questions that need recent information, the agent can use the **Exa API**.

The web pipeline retrieves a larger candidate set, ranks the results locally, keeps the most relevant ones, summarizes them, and then includes them in the final prompt.

This also makes it possible to ask questions such as:

```text
Compare the method in my uploaded paper with recent approaches online.
```

In that case, the system can use both RAG and web search in the same response.

## Function calling

The project currently includes a calculator tool.

Example:

```text
Calculate (235 * 17 + 42) / 3.
```

The model detects that a calculation is needed, extracts the mathematical expression, and passes it to a restricted Python evaluator.

It can also combine tools. For example:

```text
Find the reported accuracy values in the uploaded paper and calculate their average.
```

In this case, the agent first retrieves the values from the document and then sends the calculation to the calculator tool.

## 20 Questions

The project also includes an interactive 20 Questions mode.

The model:

1. asks a yes/no question,
2. receives the answer,
3. makes a guess,
4. updates its reasoning state,
5. continues until it guesses correctly or reaches 20 turns.

The question generation prompt is designed to prefer broad, informative questions first and avoid repeating previous questions.

The evaluation script works with the `ValidatorModel` supplied for the assignment:

```bash
python evaluate_20Q.py -N 100
```

It reports the number of wins, win rate, average number of turns, and best/worst games.

## Tech stack

- Python
- Ollama
- Gemma 3
- nomic-embed-text
- Streamlit
- PyMuPDF
- FAISS
- BM25
- Exa API
- NumPy

## Project structure

```text
advanced-agent-ai/
├── app.py
├── config.py
├── evaluate_20Q.py
├── core/
│   ├── ollama_client.py
│   ├── router.py
│   └── memory.py
├── rag/
│   ├── pdf_loader.py
│   ├── hybrid_rag.py
│   └── query_rewriter.py
├── tools/
│   ├── calculator.py
│   ├── exa_search.py
│   └── executor.py
├── verification/
│   └── verifier.py
├── game/
│   └── twenty_questions.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Install the local models

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### 2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

Python 3.11 is recommended.

### 3. Configure Exa

Copy `.env.example` to `.env` and add your Exa API key:

```text
EXA_API_KEY=your_exa_api_key_here
```

### 4. Run the app

```bash
python -m streamlit run app.py
```

## Example prompts

### Normal chat

```text
Explain reinforcement learning in simple terms.
```

### Document question

```text
According to the uploaded paper, what is the main contribution?
```

### Document + web

```text
Compare the method in this paper with recent work online.
```

### Calculator

```text
Calculate (235 * 17 + 42) / 3.
```

### Document + calculator

```text
Find the three reported accuracy values in the PDF and calculate their average.
```

### Game

```text
Let's play 20 questions.
```

## Notes

This is still an experimental project. Small local models such as Gemma 3 1B are fast and convenient, but routing, retrieval, and structured outputs are not always perfect.

One of the main reasons for building the project this way was to make the intermediate steps visible. The Streamlit interface shows the selected intent, retrieved document evidence, web results, tool calls, and verification output, which makes it easier to understand where a response came from and where the system can fail.

## Possible next steps

A few directions I may explore next:

- stronger reranking models,
- better long-term memory,
- more tools,
- structured evaluation for RAG quality,
- better confidence calibration,
- more systematic evaluation of the 20 Questions strategy,
- support for additional document types.

---

Built as an exploration of local LLMs, retrieval-augmented generation, tool use, and agent-style workflows.
