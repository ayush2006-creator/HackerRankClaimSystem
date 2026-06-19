import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
print("Available models containing 'flash':")
for m in client.models.list():
    if 'flash' in m.name:
        print(m.name)
