from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    agent_model: str = "llama3.1:8b"
    critic_model: str = "mistral:7b"
    multimodal_model: str = "llava:7b"
    db_path: str = "deltaloop.db"
    mlflow_tracking_uri: str = "http://localhost:5000"
    preference_pair_threshold: int = 50
    lora_r: int = 16
    lora_alpha: int = 32
    training_epochs: int = 1
    benchmark_batch_size: int = 10
    failure_cluster_k: int = 5

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
