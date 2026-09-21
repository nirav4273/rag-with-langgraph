from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma, InMemoryVectorStore
import os
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()


def load_file():
    loader = PyPDFLoader('./files/sample.pdf')
    docs = loader.load()
    return docs


memory = InMemorySaver()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

llm = ChatOpenAI(
    api_key=SecretStr(os.getenv('OPENAI_API_KEY', '')),
    model= str(os.getenv('OPENAI_MODEL', ''))
)

def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=300
    )

    splitter = text_splitter.split_documents(documents=docs)
    return splitter


def vector_search(splitter,query):
    vector_store = InMemoryVectorStore.from_documents(
        documents=splitter,
        embedding=embeddings, # Where to save data locally, remove if not necessary
    )

    search_result = vector_store.similarity_search(query, k=5)
    content = ''

    for data in search_result:
        content = content + data.page_content

    return content



def llm_search(query, content, history=None):
    history_text = "\n".join(f"{i}. {q}" for i, q in enumerate(history or [], 1)) or "None"
    result = llm.invoke(f"""
        Provide the result for the asked question from the provided context.
        If the user asks about their earlier questions, answer from the previous questions list.

        Previous questions:
        {history_text}

        Context: {content}
        Aksed question is {query}
    """)

    return str(result.content)

def execute(query):
    docs = load_file()
    splitter = text_split(docs)
    content = vector_search(splitter=splitter, query=query)
    result = llm_search(query=query,content=content)
    return result


#####
class QnAState(BaseModel):
    user_question: str = Field(description="Asked question by the user", default="")
    # docs: list[str] = Field(description="Docs which found using vector search", default=[])
    result: str = Field(description="Ans for the asked question", default="")
    context: str = Field(description="Context for the LLM model with question to get answer", default= "")
    history: list[str] = Field(description="Questions asked so far in this session", default=[])

graph = StateGraph(QnAState)

def ask_question(state: QnAState) -> QnAState:
    query = input("Asked question: ")
    state.user_question = query
    return state;

def route_after_ask(state: QnAState):
    if state.user_question.strip().lower() == 'exit':
        return 'end_node'
    return 'vector_query'


docs = load_file()
splitter = text_split(docs)

def vector_query(state: QnAState) -> QnAState:
    result = vector_search(query=state.user_question, splitter=splitter)
    state.context = result;
    return state;

def llm_query(state: QnAState) -> QnAState:
    result = llm_search(query=state.user_question, content=state.context, history=state.history)
    state.result = result;
    return state;

def result_node(state:QnAState) -> QnAState:
    print(f"Ans = {state.result}")
    return QnAState(history=state.history + [state.user_question])

def end_node(state:QnAState) -> QnAState:
    print("Exit.... Thank you")
    return state

graph.add_node('ask_question', ask_question)
graph.add_node('vector_query', vector_query)
graph.add_node('llm_query', llm_query)
graph.add_node('result_node', result_node)
graph.add_node('end_node', end_node)


graph.add_edge(START, 'ask_question')
graph.add_conditional_edges('ask_question', route_after_ask, ['vector_query', 'end_node'])
graph.add_edge('vector_query', 'llm_query')
graph.add_edge('llm_query', 'result_node')
graph.add_edge('result_node', 'ask_question')
graph.add_edge('end_node', END)

graph = graph.compile(checkpointer=memory);

graph.invoke(QnAState(), config={"recursion_limit": 1000, 'configurable': {
    'thread_id': '123'
}})

    