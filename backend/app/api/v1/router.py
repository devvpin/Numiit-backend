from fastapi import APIRouter

from app.api.v1.routes import auth, health, scans, uploads, users, analyze_character

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(uploads.router, prefix="/uploads", tags=["uploads"])
api_router.include_router(analyze_character.router, prefix="/analyze-character", tags=["analyze-character"])
api_router.include_router(scans.router, prefix="/scans", tags=["scans"])
