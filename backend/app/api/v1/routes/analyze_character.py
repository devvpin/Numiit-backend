"""
analyze_character.py – Dedicated Brahmi character recognition endpoint.

POST /api/v1/analyze-character
Accepts a multipart image and returns individual Brahmi character detections
with Unicode codepoints, IAST transliterations, bounding boxes, and confidence.
"""
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, status

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[5]
UPLOADS_DIR = REPO_ROOT / "backend" / "uploads"


@router.post("")
def analyze_character(file: UploadFile = File(...)) -> dict:
    """
    Recognise individual Brahmi characters (handwritten or distorted) in an image.

    Returns:
    - **characters**: list of detected characters with Unicode, IAST transliteration,
      confidence, and normalised bounding box.
    - **regions**: DetectedRegion-compatible list for the frontend detection screen.
    - **scan_mode**: always "character"
    """
    # Ensure directory exists
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Validate extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files (jpg, jpeg, png, webp) are allowed.",
        )

    unique_filename = f"char_{uuid.uuid4()}{file_ext}"
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

    try:
        from app.services.ml_inference import infer_character
        result = infer_character(str(dest_path))
    except ImportError:
        result = {
            "scan_mode": "character",
            "characters": [],
            "regions": [
                {
                    "regionIndex": 0,
                    "boundingBox": {"left": 0.1, "top": 0.1, "width": 0.8, "height": 0.8},
                    "scriptName": "Brahmi",
                    "originalText": "",
                    "transliteration": "—",
                    "translation": "ultralytics not installed – run: pip install ultralytics torch",
                    "dynastyContext": "Brahmi Script – Handwritten / Distorted Character Recognition",
                    "confidence": 0.0,
                    "glyphCount": 0,
                }
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Character inference failed: {str(e)}",
        )

    return {
        "image_path": image_url,
        "thumbnail_path": image_url,
        **result,
    }
