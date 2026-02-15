import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found.")
else:
    client = genai.Client(api_key=api_key)
    try:
        print("Listing available models:")
        for m in client.models.list():
            print(m.name)
    except Exception as e:
        print(f"Error listing models: {e}")
