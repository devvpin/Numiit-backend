from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.db.base import Base
from app.db.session import engine
from app.models import User, Scan


REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_WEB_BUILD = REPO_ROOT / "frontend" / "build" / "web"
UPLOADS_DIR = REPO_ROOT / "backend" / "uploads"


def ensure_database_exists() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError
    from app.core.config import get_settings

    settings = get_settings()
    db_url = settings.database_url

    if "postgresql" in db_url:
        try:
            temp_engine = create_engine(db_url)
            with temp_engine.connect() as conn:
                pass
            temp_engine.dispose()
            return
        except OperationalError as e:
            if "does not exist" in str(e):
                try:
                    base_url, db_name = db_url.rsplit("/", 1)
                    # Strip query parameters if any (e.g. ?sslmode=...)
                    clean_db_name = db_name.split("?")[0]
                    postgres_url = f"{base_url}/postgres"

                    sys_engine = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
                    with sys_engine.connect() as conn:
                        from sqlalchemy import text
                        conn.execute(text(f"CREATE DATABASE {clean_db_name}"))
                    sys_engine.dispose()
                    print(f"Database '{clean_db_name}' created successfully on PostgreSQL.")
                except Exception as create_err:
                    print(f"Could not auto-create database: {create_err}")
            else:
                print(f"Database connection error: {e}")


def create_app() -> FastAPI:
    # Auto-create database if not exists
    ensure_database_exists()

    # Auto-create tables on startup
    Base.metadata.create_all(bind=engine)

    # Database is initialized in ensure_database_exists and create_all
    # Ensure uploads directory exists
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="NumiIT API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")

    # Serve uploads
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

    if FRONTEND_WEB_BUILD.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_WEB_BUILD, html=True), name="frontend")
    else:
        @app.get("/")
        def root() -> dict[str, str]:
            return {"name": "NumiIT API", "status": "ready"}

    return app


app = create_app()