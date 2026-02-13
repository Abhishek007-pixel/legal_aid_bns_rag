"""
Test the newest free models from OpenRouter (Feb 2026)
Based on search results
"""
from openai import OpenAI
import time

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Latest free models from search results (Feb 2026)
test_models = [
    # Special auto-router
    "openrouter/free",  # Auto-selects free models
    
    # Newer free models from search
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "stepfun/step-3.5-flash:free",
    "upstage/solar-pro-3:free",
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-oss-120b:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    
    # Fallbacks
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
]

print("="*80)
print(" TESTING LATEST FREE MODELS (FEB 2026)")
print("="*80)

working_models = []

for model in test_models:
    print(f"\nTesting: {model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say only: Hello"}],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content
        print(f"  ✓✓✓ WORKS! Response: {answer}")
        working_models.append(model)
        time.sleep(2)  # Wait between tests
    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            print(f"  ⚠ RATE LIMITED (exists, try later)")
            if model not in working_models:
                working_models.append(f"{model} (rate limited)")
        elif "404" in error_str:
            print(f"  ✗ NOT FOUND")
        elif "502" in error_str or "503" in error_str:
            print(f"  ⚠ SERVER ERROR (might work later)")
        else:
            print(f"  ✗ ERROR: {error_str[:100]}")

print("\n" + "="*80)
print(" SUMMARY:")
print("="*80)
if working_models:
    for model in working_models:
        print(f"  ✓ {model}")
    print(f"\nFound {len(working_models)} potentially working model(s)")
else:
    print("  ✗ No models worked")
print("="*80)
