"""
List available Gemini models
"""
import google.generativeai as genai

genai.configure(api_key="AIzaSyA0TAJXE39tHgiGJshMy8nAcdTH0_auQRU")

print("Available Gemini Models:")
print("="*80)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  {model.name}")
        print(f"    Description: {model.description[:60]}...")
        print(f"    Methods: {model.supported_generation_methods}")
        print()
