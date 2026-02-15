import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import human_message
from langchain_google_genai import ChatGoogleGenerativeAI as gemini
import pydantic
load_dotenv()


llm = gemini(google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3-flash-preview")
tools = [search]

agent = create_agent(
    llm=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)


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


def main():
   result = agent.invoke({"messages": [human_message(content="What is the capital of France?")]})
   print(result)
if __name__ == "__main__":
    main()
