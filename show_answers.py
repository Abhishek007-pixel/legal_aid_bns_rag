
"""
Script to run queries and save full answers to a text file
"""
import sys
import json
from datetime import datetime
from app.rag import answer

# ANSI colors for console (not used in file)
class C:
    H = '\033[95m'; B = '\033[94m'; G = '\033[92m'; Y = '\033[93m'
    R = '\033[91m'; E = '\033[0m'; BOLD = '\033[1m'

def main():
    queries = [
        "What is theft under BNS?",
        "What are the punishments for murder?",
        "Explain criminal conspiracy"
    ]
    
    output_file = "answers.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== LEGl AID RAG PIPELINE RESULTS ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, query in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)}: {query}")
            f.write(f"QUERY {i}: {query}\n")
            f.write("-" * 80 + "\n")
            
            try:
                result = answer(query)
                
                if result and result.get('answer'):
                    f.write(f"ANSWER:\n{result['answer']}\n\n")
                    
                    if result.get('citations'):
                        f.write("CITATIONS:\n")
                        for cite in result['citations']:
                            f.write(f"- [{cite.get('ref', '')}] {cite.get('title', '')} ({cite.get('where', '')})\n")
                else:
                    f.write("ERROR: No answer generated\n")
            except Exception as e:
                f.write(f"ERROR: {str(e)}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
    print(f"\nDone! Answers saved to {output_file}")

if __name__ == "__main__":
    main()
