"""
Test which free models actually work on OpenRouter
"""
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Test models that should be free
free_models = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemini-flash-1.5:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

print("Testing Free Models on OpenRouter")
print("="*80)

for model in free_models:
    print(f"\nTesting: {model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'Hello' only."}],
            max_tokens=10,
            extra_body={
                "provider": {
                    "order": ["DeepInfra", "Novita", "Recursal"],
                    "allow_fallbacks": False
                }
            }
        )
        print(f"  [OK] Works! Response: {resp.choices[0].message.content}")
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  [FAIL] {error_msg}")

print("\n" + "="*80)
