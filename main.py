from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, SecretStr
from typing import Literal, cast
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch


CATEGORY_LITERAL = Literal['weather', 'coding', 'google_search',]

load_dotenv()

class UserQuestionFlow(BaseModel):
    question: str = Field("Question which asked by the user")
    category: CATEGORY_LITERAL = Field(default="google_search", description="Question category will be store here and not match any category set google_search those kind of questions")
    answer: str = Field(default="", description="Asked question result will be store inside this.")

class QuestionCategory(BaseModel):
    category: CATEGORY_LITERAL = Field(default="google_search", description="Question category will be store here and not match any category set google_search those kind of questions")



llm = ChatGroq(
    api_key=SecretStr(os.getenv('GROQ_API_KEY', '')),
    model= str(os.getenv('GROQ_MODEL', ''))
)

def question_category(state: UserQuestionFlow) -> UserQuestionFlow:
    llm_structured_output = llm.with_structured_output(QuestionCategory)

    result = llm_structured_output.invoke(f"""
    I want to know the question category of my question.
    My Question is {state.question}
    """)

    result = cast(QuestionCategory, result);
    state.category = result.category
    return state

result =  question_category(UserQuestionFlow(question="What is 2+2?"))

def route(state: UserQuestionFlow) -> CATEGORY_LITERAL:
    return state.category


tavily_search_tool  = TavilySearch(
    max_results=5,
    # topic="general",
    # include_answer=False,
    include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    search_depth="basic",
    time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

agent = create_agent(
    model=llm,
    tools=[tavily_search_tool],
    system_prompt="You are web search assistant and get user ans for the asked question in short descriptions"
)


def google_search(state: UserQuestionFlow) -> UserQuestionFlow:
    print("Google search Invoke")
    result = agent.invoke({
        'messages': [
            {
                'role': 'user',
                'content': state.question
            }
        ]
    })
    messages = result['messages']
    state.answer = messages[-1].content
    return state


def coding(state: UserQuestionFlow) -> UserQuestionFlow:
    result = llm.invoke(state.question)
    state.answer = str(result.content)
    return state

def weather(state: UserQuestionFlow) -> UserQuestionFlow:
    result = agent.invoke({
        'messages': [
            {
                'role': 'user',
                'content': state.question
            }
        ]
    })
    messages = result['messages']
    state.answer = messages[-1].content
    return state


graph = StateGraph(UserQuestionFlow)
graph.add_node('question_category', question_category)

## ['weather', 'coding', 'google_search',]

## Node name should be same as literal
graph.add_node('coding',coding)
graph.add_node('weather', weather)
graph.add_node('google_search', google_search)

graph.add_edge(START, 'question_category')
graph.add_conditional_edges('question_category', route)
graph.add_edge('coding', END)
graph.add_edge('weather', END)
graph.add_edge('google_search', END)

final_graph = graph.compile()

result = final_graph.invoke(UserQuestionFlow(
    question= 'what is js'
))
print(result['category'])
print(result['answer'])

