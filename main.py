"""
ReAct Agent with Google Gemini and Tavily Search.

This module implements a ReAct-style agent using LangChain v1 that can search
the web using Tavily and respond with summaries powered by Google Gemini.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# Load environment variables (GOOGLE_API_KEY, TAVILY_API_KEY)
load_dotenv()

# Configuration constants
RECURSION_LIMIT = 50
TAVILY_MAX_RESULTS = 5
GEMINI_MODEL = "gemini-2.5-flash"

# System prompt for the agent - EXPLICIT about tool usage
SYSTEM_PROMPT = """You are a research assistant that MUST use the tavily_search tool.

IMPORTANT: You MUST call the tavily_search tool for EVERY user query.
DO NOT answer from memory. DO NOT make up information.
ALWAYS search first, then summarize the results.

Workflow:
1. Receive user query
2. Call tavily_search with an appropriate search query
3. Analyze the search results
4. Provide a summary based ONLY on the search results"""

# Initialize tools
tavily_search = TavilySearch(max_results=TAVILY_MAX_RESULTS, topic="general")

# Initialize LLM with tool_choice to force tool usage
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)

# Bind tools with tool_choice="any" to force tool usage
llm_with_forced_tools = llm.bind_tools([tavily_search], tool_choice="any")

# Create agent using the model with forced tool choice
agent = create_agent(
    model=llm_with_forced_tools, tools=[tavily_search], system_prompt=SYSTEM_PROMPT
)


def main() -> None:
    """Run the AI agent with a sample query."""
    query = "Search for three AI engineer opportunities in Brazil on LinkedIn and tell me the salary range."

    print(f"Query: {query}\n")
    print("=" * 60)

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )

        # Debug: Show all messages to verify tool was called
        messages = result["messages"]
        tool_calls_made = 0

        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made += len(msg.tool_calls)
                for tc in msg.tool_calls:
                    print(
                        f"[TOOL CALL] {tc['name']}: {tc['args'].get('query', 'N/A')[:80]}..."
                    )

            if isinstance(msg, ToolMessage):
                print(f"[TOOL RESULT] {msg.name}: {str(msg.content)[:100]}...")

        print("=" * 60)
        print(f"Total tool calls: {tool_calls_made}")
        print("=" * 60 + "\n")

        # Extract and print the final response
        final_message = messages[-1]
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
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
