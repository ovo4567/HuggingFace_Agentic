from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

# The original tool instance
duckduckgo_tool = DuckDuckGoSearchRun()

def robust_search(query: str) -> str:
    """Performs a search using DuckDuckGo with error handling."""
    try:
        result = duckduckgo_tool.invoke(query)
        if not result or len(result.strip()) == 0:
            return "No search results were found for this query."
        return result
    except Exception as e:
        # We return the error as a string so the LLM can understand 
        # what went wrong and potentially try a different approach.
        return f"Error during search: {str(e)}"

# Initialize the robust version of the tool
search_tool = Tool(
    name="duckduckgo_search",
    func=robust_search,
    description="A reliable search engine to find information on the web."
)


from langchain_core.tools import Tool
import random

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)

from langchain_core.tools import Tool
from huggingface_hub import list_models

def get_hub_stats(author: str) -> str:
    """Fetches the most popular model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model and number of downloaded from a specific author on the Hugging Face Hub."
)

# Example usage
print(hub_stats_tool.invoke("facebook")) # Example: Get the most downloaded model by Facebook

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

# 1. Load env variables from .env file
load_dotenv(r"LangGraph\.env")
# 2. Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# 3. Optional: Give your project a name so it's easy to find in the dashboard
os.environ["LANGCHAIN_PROJECT"] = "Alfred-Agent-Debug"


# Generate the chat interface, including the tools
llm = model = ChatOpenAI(
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_key="lm-studio",
    model = "local-model"
)


tools = [search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = llm.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Who is Facebook? what's their most popular model on the Hugging Face Hub? How many times has it been downloaded? What is the weather like in hong kong?")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

from IPython.display import Image, display

# This generates a PNG of your graph structure
try:
    display(Image(alfred.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Could not render image. Ensure dependencies are installed.")
    print(alfred.get_graph().draw_mermaid()) 