from dotenv import load_dotenv; load_dotenv()
import os, google.generativeai as genai

api = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY present:", bool(api))
genai.configure(api_key=api)

for m in genai.list_models():
    methods = getattr(m, "supported_generation_methods", []) or []
    if "generateContent" in methods:
        print(m.name)
