import uvicorn
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("="*60)
    print("   LEGAL AID RAG SERVER STARTING")
    print("="*60)
    print("1. Backend API: http://localhost:8000")
    print("2. Frontend UI: http://localhost:8000/")
    print("3. Documentation: http://localhost:8000/docs")
    print("-" * 60)
    
    try:
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        input("Press Enter to exit...")
