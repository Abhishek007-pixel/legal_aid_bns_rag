from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator

class Settings(BaseSettings):
    # --- Sarvam AI (LLM) ---
    SARVAM_API_KEY: str
    SARVAM_API_URL: str = "https://api.sarvam.ai/v1/chat/completions"
    SARVAM_MODEL: str = "sarvam-m"

    # --- Retrieval mode flags ---
    USE_FAISS: bool = False        # True = vector+BM25 hybrid; False = BM25 only
    USE_BM25: bool = True
    USE_EMBEDDINGS: bool = False   # Must match USE_FAISS

    # --- Embedding (local sentence-transformers, no API key needed) ---
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Reranking ---
    ENABLE_RERANKING: bool = True

    # --- Retrieval knobs ---
    TOP_K: int = 5
    RRF_K: int = 60
    RERANK_CANDIDATES: int = 20
    INITIAL_K: int = 30
    EXPAND_NEIGHBORS: int = 2
    MIN_SIM_SCORE: float = 0.15
    BM25_WEIGHT: float = 1.0
    VEC_WEIGHT: float = 0.0

    # --- App ---
    JURISDICTION: str = "IN"
    SCOPE_TOPICS: str = "criminal law, procedure"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_config(self):
        if self.USE_EMBEDDINGS and not self.USE_FAISS:
            raise ValueError(
                "Invalid config: USE_EMBEDDINGS=True requires USE_FAISS=True. "
                "Either set USE_FAISS=true or set USE_EMBEDDINGS=false."
            )
        return self

settings = Settings()