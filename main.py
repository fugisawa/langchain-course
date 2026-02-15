"""ReAct Agent with Gemini and Tavily Search."""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """A source reference."""
    name: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")


class AgentResponse(BaseModel):
    """Structured response with answer and sources."""
    answer: str = Field(description="The answer to the query")
    sources: list[Source] = Field(description="Sources used")


# Config
MODEL = "gemini-2.5-flash"
MAX_RESULTS = 5

SYSTEM_PROMPT = """You are a research assistant with web search capability.

Rules:
1. ALWAYS search before answering - never use prior knowledge
2. Make ONE search with a well-crafted query
3. Summarize the search results directly
4. If results are insufficient, say so - don't search again"""

# Setup
tavily = TavilySearch(max_results=MAX_RESULTS)
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.2)

agent = create_agent(
    model=llm.bind_tools([tavily], tool_choice="tavily_search"),
    tools=[tavily],
    system_prompt=SYSTEM_PROMPT,
    response_format=AgentResponse,
)


def run(query: str) -> AgentResponse:
    """Run agent and return structured response."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 10},
    )
    messages = result["messages"]

    # Log searches
    searches = [m for m in messages if isinstance(m, ToolMessage)]
    if searches:
        print(f"[searched {len(searches)}x]\n")

    # Return structured response
    return result["structured_response"]


def print_response(response: AgentResponse) -> None:
    """Print response in readable format."""
    print(response.answer)

    if response.sources:
        print("\n" + "─" * 40)
        print("Sources:")
        for src in response.sources:
            print(f"  • {src.name}")
            print(f"    {src.url}")


if __name__ == "__main__":
    response = run("AI engineer average salary in Brazil 2026")
    print_response(response)
