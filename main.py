import os
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI as gemini
from langchain_core.messages import HumanMessage

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Search the web for the given query.
    
    Args:
        query (str): The query to search for.
    
    Returns:
        str: The search results.
    """
    return f"Search result for {query}"


llm = gemini(google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3-flash-preview")
tools = [search]

agent = create_react_agent(llm, tools)


def main():
    result = agent.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
    # The result from langgraph agent is a dict with 'messages' key containing the conversation history
    # The last message should be the answer
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
