"""
Find working free models on OpenRouter by testing systematically
"""
from openai import OpenAI
import time

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Models to test (known free models)
test_models = [
    # Meta Llama models
    "meta-llama/llama-3.2-1b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    
    # Google models
    "google/gemini-flash-1.5:free",
    "google/gemini-2.0-flash-exp:free",
    
    # Microsoft models
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    
    # Qwen models
    "qwen/qwen-2-7b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    
    # Mistral
    "mistralai/mistral-7b-instruct:free",
    
    # Others
    "huggingfaceh4/zephyr-7b-beta:free",
]

print("="*80)
print(" TESTING FREE MODELS ON OPENROUTER")
print("="*80)

working_models = []

for model in test_models:
    print(f"\nTesting: {model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say only: OK"}],
            max_tokens=5,
        )
        answer = resp.choices[0].message.content
        print(f"  ✓ WORKS! Response: {answer}")
        working_models.append(model)
        time.sleep(1)  # Avoid rate limits
    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            print(f"  ✗ RATE LIMITED (might work later)")
        elif "404" in error_str:
            print(f"  ✗ NOT FOUND")
        elif "quota" in error_str.lower():
            print(f"  ✗ QUOTA EXCEEDED")
        else:
            print(f"  ✗ ERROR: {error_str[:80]}")

print("\n" + "="*80)
print(" WORKING MODELS FOUND:")
print("="*80)
if working_models:
    for model in working_models:
        print(f"  ✓ {model}")
else:
    print("  No working models found (all quota exceeded or rate limited)")
print("="*80)
