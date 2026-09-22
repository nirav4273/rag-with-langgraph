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

- `ask_question` calls `interrupt("Asked question: ")`, which pauses the graph and saves state to the checkpoint for the `thread_id`.
- It then returns a `Command(update={"user_question": ...}, goto=...)`: `exit` goes to `end_node`, anything else goes to `vector_query`.
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

### Interrupt usage

The script drives the graph with a human-in-the-loop pattern (requires a checkpointer and a `thread_id`):

```python
config = {"configurable": {"thread_id": "123"}}
result = graph.invoke({...}, config=config)          # runs until the first interrupt()

while "__interrupt__" in result:
    prompt = result["__interrupt__"][0].value        # the value passed to interrupt()
    answer = input(prompt)
    result = graph.invoke(Command(resume=answer), config=config)  # resume from the checkpoint
```

- `interrupt(value)` pauses the graph and surfaces `value` under `__interrupt__` in the result.
- `Command(resume=answer)` reloads the checkpoint and re-runs the interrupted node; `interrupt()` now returns `answer`.
- The node restarts from its beginning on resume, so keep side effects before `interrupt()` idempotent.
- Use the same `thread_id` for every call, otherwise the history and pending interrupt are not found.

## Dependencies

`langgraph`, `langchain-community`, `langchain-openai`, `langchain-groq`, `langchain-tavily`, `langchain-text-splitters`, `pypdf`, `pydantic`, `dotenv`
