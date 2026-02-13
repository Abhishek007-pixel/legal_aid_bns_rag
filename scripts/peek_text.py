import json
from pathlib import Path

META_FILE = Path("data/index/meta.jsonl")

def main():
    if not META_FILE.exists():
        print("❌ Error: meta.jsonl not found. Run ingest.py first.")
        return

    print(f"📖 Reading text from {META_FILE}...\n")
    
    count = 0
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if count >= 5: break  # Stop after 5 chunks
            
            data = json.loads(line)
            print(f"🔹 CHUNK #{count + 1}")
            print(f"📂 Source: {data.get('filename')}")
            print(f"📑 Section: {data.get('title')}")
            print(f"📝 Text:   {data.get('text')[:200]}...") # First 200 chars
            print("-" * 50)
            count += 1

if __name__ == "__main__":
    main()