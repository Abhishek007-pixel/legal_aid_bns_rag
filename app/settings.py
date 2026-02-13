from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Hugging Face (for LLM) ---
    HF_TOKEN: str
    HF_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # --- OpenRouter (for embeddings only) ---
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OR_SITE_URL: str | None = None
    OR_SITE_NAME: str | None = None

    # --- Retrieval knobs ---
    USE_EMBEDDINGS: bool = True   
    USE_FAISS: bool = True        
    USE_BM25: bool = True
    
    # Embedding Model (Must match what you used in ingest.py)
    EMBED_MODEL: str = "openai/text-embedding-3-small"

    # Reranking knobs
    ENABLE_RERANKING: bool = True # Toggle to save memory if needed
    TOP_K: int = 5            # How many go to LLM
    RRF_K: int = 60           # RRF constant
    RERANK_CANDIDATES: int = 20 # How many to send to Cross-Encoder

    # --- App ---
    JURISDICTION: str = "IN"

    # This line fixes the error by ignoring old variables in your .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()