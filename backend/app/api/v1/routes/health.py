from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    import sys
    diagnostics = {
        "executable": sys.executable,
        "prefix": sys.prefix,
    }
    
    # Try importing cv2
    try:
        import cv2
        diagnostics["cv2"] = "ok"
    except Exception as e:
        diagnostics["cv2"] = f"error: {e}"
        
    # Try importing torch
    try:
        import torch
        diagnostics["torch"] = "ok"
    except Exception as e:
        diagnostics["torch"] = f"error: {e}"

    # Try importing ultralytics
    try:
        from ultralytics import YOLO
        diagnostics["ultralytics"] = "ok"
    except Exception as e:
        diagnostics["ultralytics"] = f"error: {e}"
        
    # Try importing Levenshtein
    try:
        import Levenshtein
        diagnostics["Levenshtein"] = "ok"
    except Exception as e:
        diagnostics["Levenshtein"] = f"error: {e}"
        
    # Try importing ml_inference
    try:
        from app.services.ml_inference import infer_coin, infer_paired_coins
        diagnostics["ml_inference"] = "ok"
        diagnostics["paired_inference"] = "ok"
    except Exception as e:
        import traceback
        diagnostics["ml_inference"] = f"error: {e}\n{traceback.format_exc()}"
        diagnostics["paired_inference"] = f"error: {e}\n{traceback.format_exc()}"

    return {
        "status": "ok",
        "diagnostics": diagnostics
    }
