import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile, HTTPException, status

router = APIRouter()

# Backend uploads directory
BACKEND_DIR = Path(__file__).resolve().parents[4]
UPLOADS_DIR = BACKEND_DIR / "uploads"


def _save_upload_file(upload_file: UploadFile, prefix: str = "") -> tuple[Path, str]:
    """Helper to validate extension and save an uploaded file."""
    file_ext = Path(upload_file.filename).suffix.lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only image files (jpg, jpeg, png, webp) are allowed. Received: {upload_file.filename}",
        )

    unique_filename = f"{uuid.uuid4()}{f'_{prefix}' if prefix else ''}{file_ext}"
    dest_path = UPLOADS_DIR / unique_filename

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded image: {str(e)}",
        )

    return dest_path, unique_filename


@router.post("")
def upload_image(
    file: UploadFile = File(..., description="Primary image (or obverse face if running paired classification)"),
    reverse: UploadFile | None = File(None, description="Optional reverse legend image for two-sided classification"),
    mode: str = Query(default="coin", description="Scan mode: 'coin', 'character', 'classify', or 'paired'"),
) -> dict:
    """
    Upload a coin/character image and run ML inference.

    - **mode=coin** (default): runs YOLO coin-era recognition model.
    - **mode=character**: runs Brahmi character recognition model.
    - **mode=classify**: runs single-side deep metric learning classification.
    - **mode=paired** (or providing 'reverse' image): runs Method 1 two-sided classification using both coin faces.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    dest_path, unique_filename = _save_upload_file(file)
    image_url = f"/uploads/{unique_filename}"

    # Check if paired two-sided mode requested
    if reverse is not None or mode in ("paired", "paired_classification", "final", "method1"):
        if reverse is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Two-sided classification mode requires both 'file' (obverse) and 'reverse' (legend) images.",
            )

        rev_dest_path, rev_filename = _save_upload_file(reverse, prefix="reverse")
        rev_url = f"/uploads/{rev_filename}"

        try:
            from app.services.ml_inference import infer_paired_coins
            inference_result = infer_paired_coins(str(dest_path), str(rev_dest_path))
        except ImportError:
            inference_result = _fallback_mock("paired_classification")
        except Exception as e:
            inference_result = _fallback_mock("paired_classification", error=str(e))

        return {
            "image_path": image_url,
            "thumbnail_path": image_url,
            "reverse_image_path": rev_url,
            **inference_result,
        }


    # Single-side ML inference
    try:
        from app.services.ml_inference import infer_coin, infer_character, infer_coin_classification

        if mode == "character":
            inference_result = infer_character(str(dest_path))
        elif mode in ("classify", "classification"):
            inference_result = infer_coin_classification(str(dest_path), use_tta=True)
        else:
            inference_result = infer_coin(str(dest_path))

    except ImportError:
        inference_result = _fallback_mock(mode)
    except Exception as e:
        inference_result = _fallback_mock(mode, error=str(e))

    return {
        "image_path": image_url,
        "thumbnail_path": image_url,
        **inference_result,
    }


@router.post("/paired")
@router.post("/classify-paired")
def upload_paired_coins(
    obverse: UploadFile = File(..., description="Obverse / Face portrait image of the coin"),
    reverse: UploadFile = File(..., description="Reverse / Legend inscription image of the coin"),
    use_genealogy: bool = Query(default=True, description="Enable domain-informed genealogical disambiguation"),
) -> dict:
    """
    Method 1 Two-Sided Western Kshatrapa Coin Attribution Endpoint.
    
    Accepts both coin sides:
    - **obverse**: Royal portrait face.
    - **reverse**: Brahmi inscription / chaitya hill legend.
    
    Processes the coin pair with:
    1. Multi-scale dual-stage contrast & relief enhancement.
    2. Concentric ring rotational TTA (16 viewpoints per side).
    3. Multi-resolution Joint Siamese Fusion (224x224 + 288x288 + single view models).
    4. Soft k-NN exemplar die matching.
    5. Domain-informed genealogical disambiguation engine.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    obv_dest, obv_filename = _save_upload_file(obverse, prefix="obverse")
    rev_dest, rev_filename = _save_upload_file(reverse, prefix="reverse")

    obv_url = f"/uploads/{obv_filename}"
    rev_url = f"/uploads/{rev_filename}"

    try:
        from app.services.ml_inference import infer_paired_coins
        inference_result = infer_paired_coins(str(obv_dest), str(rev_dest), use_genealogy=use_genealogy)
    except ImportError:
        inference_result = _fallback_mock("paired_classification")
    except Exception as e:
        inference_result = _fallback_mock("paired_classification", error=str(e))

    return {
        "image_path": obv_url,
        "thumbnail_path": obv_url,
        "reverse_image_path": rev_url,
        **inference_result,
    }


def _fallback_mock(mode: str, error: str | None = None) -> dict:
    """
    Returns a structured mock result when dependencies are not yet installed or
    inference fails, so the frontend still gets the correct schema shape.
    """
    note = f" (Error: {error})" if error else " (Dependencies not installed)"
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
    elif mode in ("paired_classification", "paired"):
        return {
            "scan_mode": "paired_classification",
            "predicted_king": "Unknown",
            "confidence": 0.0,
            "top_predictions": [],
            "disambiguation_applied": False,
            "disambiguation_note": f"Fallback mode active{note}",
            "dynasty_context": "Western Kshatrapas",
            "script": "Brahmi",
            "transliteration": "—",
            "translation": f"Inference unavailable{note}",
            "father": "Unknown",
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
                    "dynastyContext": f"Install dependencies to enable real coin recognition",
                    "confidence": 0.0,
                    "glyphCount": 0,
                }
            ],
        }
