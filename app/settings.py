from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- OpenRouter (LLM) ---
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OR_MODEL: str = "alibaba/tongyi-deepresearch-30b-a3b:free"
    OR_SITE_URL: str | None = None
    OR_SITE_NAME: str | None = None

    # --- Retrieval knobs ---
    USE_EMBEDDINGS: bool = False   # <— add this
    USE_FAISS: bool = True         # FAISS will only be used if embeddings exist
    USE_BM25: bool = True
    TOP_K: int = 5
    INITIAL_K: int = 25
    MIN_SIM_SCORE: float = 0.28
    BM25_WEIGHT: float = 0.40
    VEC_WEIGHT: float = 0.60
    EXPAND_NEIGHBORS: int = 1

    # (Optional) embeddings config if you later enable them
    # GOOGLE_API_KEY: str | None = None
    # EMBED_MODEL: str = "text-embedding-004"

    # --- App ---
    JURISDICTION: str = "IN"
    SCOPE_TOPICS: str = "contracts"

    class Config:
        env_file = ".env"

settings = Settings()
