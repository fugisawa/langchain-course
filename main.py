"""ReAct Agent with Gemini and Tavily Search."""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

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
)


def run(query: str) -> str:
    """Run agent and return response."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 10},
    )
    messages = result["messages"]

    # Log searches
    searches = [m for m in messages if isinstance(m, ToolMessage)]
    if searches:
        print(f"[searched {len(searches)}x]")

    # Extract final response
    content = messages[-1].content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b.get("type") == "text")
    return content


if __name__ == "__main__":
    print(run("AI engineer average salary in Brazil 2026"))
