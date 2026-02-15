import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

print("Initializing Chat LLM...")
llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash")
print("Invoking Chat LLM...")
try:
    res = llm.invoke("Hello, how are you?")
    print(f"Response: {res.content}")
except Exception as e:
    print(f"Error: {e}")
