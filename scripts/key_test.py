from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

print("KEY present?", bool(os.getenv("OPENAI_API_KEY")))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT"),
    organization=os.getenv("OPENAI_ORG_ID"),
)

try:
    r = client.embeddings.create(model="text-embedding-3-small", input="hello world")
    print("✅ OK, embedding length:", len(r.data[0].embedding))
except Exception as e:
    print("❌ FAILED:", type(e).__name__, e)
