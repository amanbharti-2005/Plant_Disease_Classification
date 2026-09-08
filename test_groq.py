import os
from dotenv import load_dotenv
load_dotenv(override=True)

from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
print("Key found:", bool(api_key))
print("Key starts with:", api_key[:10] if api_key else "NONE")

client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(response.choices[0].message.content)