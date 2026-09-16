from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, SecretStr
from typing import List, Annotated
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = SecretStr(os.getenv('GROQ_API_KEY', ''))
if not groq_api_key:
    raise RuntimeError('Set GROQ_API_KEY first: $env:GROQ_API_KEY="your_api_key"')

groq_model = os.getenv('GROQ_MODEL', '')

class ChatMessage(BaseModel):
    messages: Annotated[list, add_messages]

graph = StateGraph(ChatMessage)

memory = InMemorySaver()
llm = ChatGroq(
    model=groq_model,
    api_key=groq_api_key
)

def ChatBotNode(state: ChatMessage) -> ChatMessage:
    result = llm.invoke(state.messages)
    state.messages = [result]
    return state

graph.add_node('chatBot', ChatBotNode)

graph.add_edge(START, 'chatBot')
graph.add_edge('chatBot', END)

final_graph = graph.compile(checkpointer=memory)

while True:
    query = input("Ask : ")
    if query:
        response = final_graph.invoke(ChatMessage(messages=[{
            'role': 'human',
            'content': query
        }]), {
            'configurable': {
                'thread_id': 'test'
            }
        })

        for message in response['messages']:
            print(">>>>", type(message) ,">>>> ",message.content)