from functools import lru_cache

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NumiIT API"
    environment: str = "development"
    database_url: str = "sqlite:///./numiit.db"
    secret_key: str = "REPLACE-ME-IN-ENV-FILE"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Google SSO (OAuth 2.0)
    google_client_id: str = ""
    google_client_secret: str = ""
    sso_redirect_uri: str = "http://localhost:8000/api/v1/auth/sso/google/callback"

    # Default seed user passwords
    admin_seed_password: str = "adminpassword"
    user_seed_password: str = "userpassword"

    @field_validator("database_url", mode="before")
    @classmethod
    def format_database_url(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+psycopg://", 1)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()

