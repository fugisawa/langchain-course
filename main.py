"""
ReAct Agent with Google Gemini and Tavily Search.

This module implements a ReAct-style agent using LangChain v1 that can search
the web using Tavily and respond with summaries powered by Google Gemini.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# Load environment variables (GOOGLE_API_KEY, TAVILY_API_KEY)
load_dotenv()

# Configuration constants
RECURSION_LIMIT = 50
TAVILY_MAX_RESULTS = 5
GEMINI_MODEL = "gemini-2.5-flash"

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful research assistant.
Use the search tool to find accurate, up-to-date information.
Provide concise summaries based on search results.
Do not loop unnecessarily - search once and summarize."""

# Initialize tools
tavily_search = TavilySearch(max_results=TAVILY_MAX_RESULTS, topic="general")

# Initialize LLM (uses GOOGLE_API_KEY from environment)
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)

# Create agent using LangChain v1 API
agent = create_agent(model=llm, tools=[tavily_search], system_prompt=SYSTEM_PROMPT)


def main() -> None:
    """Run the AI agent with a sample query."""
    query = "Search for three AI engineer opportunities in Brazil on LinkedIn and tell me the salary range."

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )

        # Extract and print the final response
        final_message = result["messages"][-1]
        content = final_message.content

        # Handle multimodal content (Gemini returns list of content blocks)
        if isinstance(content, list):
            text_parts = [
                block["text"] for block in content if block.get("type") == "text"
            ]
            print("\n".join(text_parts))
        else:
            print(content)

    except Exception as e:
        print(f"Error executing agent: {e}")


if __name__ == "__main__":
    main()
