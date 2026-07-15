import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile, HTTPException, status

router = APIRouter()

# Get project root
REPO_ROOT = Path(__file__).resolve().parents[6]
UPLOADS_DIR = REPO_ROOT / "backend" / "uploads"


@router.post("")
def upload_image(
    file: UploadFile = File(...),
    mode: str = Query(default="coin", description="Scan mode: 'coin' or 'character'"),
) -> dict:
    """
    Upload a coin/character image and run ML inference.

    - **mode=coin** (default): runs coin-era recognition model, returns era_scores
      with ALL eras including negative (0%) scores.
    - **mode=character**: runs Brahmi character recognition model, returns
      individual character regions.
    """
    # Ensure directory exists
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files (jpg, jpeg, png, webp) are allowed.",
        )

    # Generate unique filename and save
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    dest_path = UPLOADS_DIR / unique_filename

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded image: {str(e)}",
        )

    image_url = f"/uploads/{unique_filename}"

    # ── Run real ML inference ─────────────────────────────────────────────────
    try:
        from app.services.ml_inference import infer_coin, infer_character

        if mode == "character":
            inference_result = infer_character(str(dest_path))
        else:
            inference_result = infer_coin(str(dest_path))

    except ImportError:
        # ultralytics not installed yet → return informative mock
        inference_result = _fallback_mock(mode)
    except Exception as e:
        # Model inference error → return fallback with error message
        inference_result = _fallback_mock(mode, error=str(e))

    return {
        "image_path": image_url,
        "thumbnail_path": image_url,
        **inference_result,
    }


def _fallback_mock(mode: str, error: str | None = None) -> dict:
    """
    Returns a structured mock result when ultralytics is not yet installed or
    inference fails, so the frontend still gets the correct schema shape.
    """
    note = f" (Error: {error})" if error else " (ultralytics not installed — install with: pip install ultralytics torch)"
    if mode == "character":
        return {
            "scan_mode": "character",
            "characters": [],
            "regions": [
                {
                    "regionIndex": 0,
                    "boundingBox": {"left": 0.1, "top": 0.1, "width": 0.8, "height": 0.8},
                    "scriptName": "Brahmi",
                    "originalText": "𑀫𑀳𑀸",
                    "transliteration": "mahā",
                    "translation": "Great [demo character]",
                    "dynastyContext": f"Brahmi Script – Character Recognition{note}",
                    "confidence": 0.0,
                    "glyphCount": 1,
                }
            ],
        }
    else:
        return {
            "scan_mode": "coin",
            "era_scores": [
                {"era": "Rudrasena II", "class_name": "rudrasena_ii",
                 "confidence": 0.0, "is_primary": True},
                {"era": "Brahmi Kshatrap", "class_name": "brahmi_kshatrap",
                 "confidence": 0.0, "is_primary": False},
                {"era": "Unknown / Other Era", "class_name": "unknown",
                 "confidence": 0.0, "is_primary": False},
            ],
            "primary_era": "Unknown",
            "primary_confidence": 0.0,
            "dynasty_context": "Unidentified",
            "script": "Brahmi",
            "transliteration": "—",
            "translation": f"ML model not available{note}",
            "regions": [
                {
                    "regionIndex": 0,
                    "boundingBox": {"left": 0.05, "top": 0.05, "width": 0.9, "height": 0.9},
                    "scriptName": "Brahmi",
                    "originalText": "",
                    "transliteration": "—",
                    "translation": f"ML model not available{note}",
                    "dynastyContext": f"Install ultralytics to enable real coin recognition",
                    "confidence": 0.0,
                    "glyphCount": 0,
                }
            ],
        }
