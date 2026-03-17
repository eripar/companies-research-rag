from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    @field_validator("anthropic_api_key", "openai_api_key")
    @classmethod
    def require_api_key(cls, v: str, info) -> str:
        if not v:
            raise ValueError(
                f"{info.field_name} is required. Set it in your .env file."
            )
        return v

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "fintech_docs"

    # Embeddings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # LLM
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # Retrieval
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5

    # SEC EDGAR
    sec_contact_email: str = ""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
