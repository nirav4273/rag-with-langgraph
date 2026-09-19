from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field, SecretStr
from typing import Annotated, List, Literal, cast
import operator
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()

class QuestionList(BaseModel):
    questions: List[str] = Field(description="List of the user asked questions", default=[])

class ToolChoice(BaseModel):
    tool: Literal["weather_search", "google_search", "coding_search"] = Field(
        description="Which tool best answers the question"
    )

class UserQnAState(BaseModel):
    user_input: str = Field(description="Question which is asked by the user")
    questions: List[str] = Field(description="List of the user asked questions", default=[])
    sub_question: str = Field(description="One question extracted from user_input, sent to the LLM in a fan-out branch", default="")
    answers: Annotated[List[str], operator.add] = Field(
        description="Collected answers from every fanned-out question", default=[]
    )
    final_answer: str = Field(description="Combined answer built after all branches finish", default="")

llm = ChatOpenAI(
    api_key=SecretStr(os.getenv('OPENAI_API_KEY', '')),
    model= str(os.getenv('OPENAI_MODEL', ''))
)

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


search_agent = create_agent(llm, tools=[tavily_search_tool])


def _run_search_agent(query: str) -> str:
    """
        Let an agent decide how to use the tavily tool and reason over its
        results instead of returning the raw tool output directly
    """
    result = search_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


def split_user_questions(state: UserQnAState):
    """"
        Split user_input into separate questions only if it actually contains
        more than one distinct question; otherwise keep it as a single question.
    """
    llm_structure_output = llm.with_structured_output(QuestionList)

    prompt = (
        "The user asked the following:\n"
        f"\"{state.user_input}\"\n\n"
        "If this contains multiple distinct questions, split it into separate "
        "questions. If it is only a single question (even if phrased as one "
        "sentence with multiple parts about the same topic), return it "
        "unchanged as the only item in the list. Do not invent or split "
        "questions that aren't actually separate questions."
    )

    result = llm_structure_output.invoke(prompt)
    result = cast(QuestionList, result)
    state.questions = result.questions
    return state


def google_search(input: str):
    """
        Search user query using a tavily-backed agent to get a reasoned result
    """
    return _run_search_agent(input)

def weather_search(input: str):
    """
        Search given place weather using a tavily-backed agent for a reasoned summary
    """
    return _run_search_agent(input)

def coding_search(input: str): 
    """
        Perform the coding or maths problem using LLM
    """
    result = llm.invoke(input=input)
    return result


def route_questions(state: UserQnAState):
    """
        Fan out every split question to its own answer_question branch
        so they run concurrently instead of one after another.
    """
    print(state.questions)
    return [Send('answer_question', {'sub_question': q}) for q in state.questions]


def answer_question(state: UserQnAState):
    """
        Pick the right tool for a single question and answer it
    """
    question = state['sub_question'] if isinstance(state, dict) else state.sub_question
    classifier = llm.with_structured_output(ToolChoice)
    choice = cast(ToolChoice, classifier.invoke(question))
    print("choize >>", choice)

    if choice.tool == 'weather_search':
        result = weather_search(question)
    elif choice.tool == 'google_search':
        result = google_search(question)
    else:
        result = coding_search(question)

    return {'answers': [f"Q: {question}\nA: {result}"]}


def collect_answer(state: UserQnAState):
    """
        Fan-in node: runs once all answer_question branches finish and
        merges their answers into a single final answer
    """
    state.final_answer = "\n\n".join(state.answers)
    return state


graph = StateGraph(UserQnAState)

graph.add_node('split_user_question', split_user_questions)
graph.add_node('answer_question', answer_question)
graph.add_node('collect_answer', collect_answer)

graph.add_edge(START, 'split_user_question')
graph.add_conditional_edges('split_user_question', route_questions, ['answer_question'])
graph.add_edge('answer_question', 'collect_answer')
graph.add_edge('collect_answer', END)

graph = graph.compile()

# from IPython.display import Image

# png_bytes = graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_bytes)



question = "who is PM of india, What is the weather in ahemdabad?"
response = graph.invoke(UserQnAState(user_input=question))
print(response['final_answer'])

