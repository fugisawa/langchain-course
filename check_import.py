try:
    from langchain.agents import create_react_agent
    print("Found in langchain.agents")
except ImportError:
    print("Not found in langchain.agents")

try:
    from langgraph.prebuilt import create_react_agent
    print("Found in langgraph.prebuilt")
except ImportError:
    print("Not found in langgraph.prebuilt")
