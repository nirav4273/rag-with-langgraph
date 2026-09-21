# rag-with-langgraph

Experiments with [LangGraph](https://github.com/langchain-ai/langgraph), progressing from a minimal graph to a RAG (retrieval-augmented generation) Q&A loop over a PDF.

## Scripts

| File | What it shows |
| --- | --- |
| [main.py](main.py) | Minimal `StateGraph` over a `GreetModel`: `START -> prefix -> greet -> postfix -> END`, prints state and a Mermaid diagram. |
| [01_langgraph.py](01_langgraph.py) | Basic chatbot with LangGraph. |
| [02_langgraph.py](02_langgraph.py) | Multi-agent graph with conditional mapping in nodes. |
| [03_multi_agent_langgraph.py](03_multi_agent_langgraph.py) | Splits a single input into multiple questions and fans them out in parallel (Send API) to the right tool/agent, then fans in the answers. |
| [04_rag_langgraph.py](04_rag_langgraph.py) | RAG over a PDF with an interactive question loop and conversation memory. |

### 04: RAG with LangGraph

Loads `files/sample.pdf`, splits it into chunks (1000 chars, 300 overlap), embeds them with OpenAI `text-embedding-3-large` into an in-memory vector store, and answers questions in a loop:

```
START -> ask_question -> (vector_query -> llm_query -> result_node -> ask_question)
                      \-> end_node -> END      (when the user types "exit")
```

- `ask_question` reads a question from the console.
- `route_after_ask` is a conditional edge: `exit` ends the graph, anything else goes to retrieval.
- `vector_query` retrieves the top 5 chunks; `llm_query` answers from that context.
- `result_node` prints the answer and appends the question to `history` in state.
- History is passed to the prompt so you can ask things like "what did I ask earlier?". State is persisted per `thread_id` with an `InMemorySaver` checkpointer (lost when the process exits).

## Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

```bash
uv sync
```

Create a `.env` file:

```
OPENAI_API_KEY=...
OPENAI_MODEL=...        # chat model used by 04
GROQ_API_KEY=...        # used by earlier scripts
GROQ_MODEL=...
TAVILY_API_KEY=...      # web search tool in 03
MODEL_PROVIDER=...
```

Put the PDF to query at `files/sample.pdf`.

## Run

```bash
uv run 04_rag_langgraph.py
```

Type a question at the `Asked question:` prompt, or `exit` to quit.

## Dependencies

`langgraph`, `langchain-community`, `langchain-openai`, `langchain-groq`, `langchain-tavily`, `langchain-text-splitters`, `pypdf`, `pydantic`, `dotenv`
