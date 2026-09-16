# rag-with-langgraph

Experiments with [LangGraph](https://github.com/langchain-ai/langgraph) — currently a minimal graph demo, with RAG (retrieval-augmented generation) planned as the project grows.

## Current state

[main.py](main.py) builds a small `StateGraph` over a `GreetModel` (a Pydantic model with a single `message` field) that chains three nodes:

```
START -> prefix -> greet -> postfix -> END
```

Running it prefixes/postfixes a message and prints the resulting state, along with a Mermaid diagram of the graph.

## Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

```bash
uv sync
```

## Run

```bash
uv run main.py
```

## Dependencies

- `langgraph`
- `pydantic`
