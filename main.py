from langgraph.graph import START, StateGraph, END
from pydantic import BaseModel


class GreetModel(BaseModel):
    message: str = ""

graph = StateGraph(GreetModel)

def greet(state: GreetModel):
    state.message = state.message
    return state


def prefix(state: GreetModel):
    state.message = " ###> " + state.message
    return state

def postfix(state: GreetModel):
    state.message = state.message + " <$$$$ "
    return state

## Connect node with function and name
graph.add_node('greet', greet)
graph.add_node('prefix', prefix)
graph.add_node('postfix', postfix)

### Connect with edges with start to end 
graph.add_edge(START, 'prefix')
graph.add_edge('prefix', 'greet')
graph.add_edge('greet', 'postfix')
graph.add_edge('postfix', END)

final_graph = graph.compile()



result = final_graph.invoke(GreetModel(message='Hello I am Nikk'))
print(result)

print(final_graph.get_graph().draw_mermaid())