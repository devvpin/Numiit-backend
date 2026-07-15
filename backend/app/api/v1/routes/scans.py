import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user_optional
from app.db.session import get_db
from app.models.user import User
from app.models.scan import Scan
from app.schemas.scan import ScanCreate, ScanRead, ScanUpdate

router = APIRouter()


def map_scan_to_schema(scan: Scan) -> ScanRead:
    try:
        regions = json.loads(scan.regions_json)
    except Exception:
        regions = []
    return ScanRead(
        id=scan.id,
        user_email=scan.user_email,
        image_path=scan.image_path,
        thumbnail_path=scan.thumbnail_path,
        primary_script=scan.primary_script,
        primary_confidence=scan.primary_confidence,
        notes=scan.notes,
        is_saved=scan.is_saved,
        is_starred=scan.is_starred,
        regions=regions,
        scanned_at=scan.scanned_at,
    )


@router.post("", response_model=ScanRead, status_code=status.HTTP_201_CREATED)
def create_scan(
    request: ScanCreate,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> ScanRead:
    email = current_user.email if current_user else "guest"

    # Serialize regions list to JSON string
    regions_list = [r.model_dump() for r in request.regions]
    regions_str = json.dumps(regions_list)

    scan = Scan(
        user_email=email,
        image_path=request.image_path,
        thumbnail_path=request.thumbnail_path,
        primary_script=request.primary_script,
        primary_confidence=request.primary_confidence,
        notes=request.notes,
        is_saved=request.is_saved,
        is_starred=request.is_starred,
        regions_json=regions_str,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return map_scan_to_schema(scan)


@router.get("", response_model=list[ScanRead])
def list_scans(
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
    scripts: list[str] | None = Query(None),
    min_confidence: float | None = Query(None),
    query: str | None = Query(None),
    sort: str = Query("newest"),
) -> list[ScanRead]:
    email = current_user.email if current_user else "guest"

    query_db = db.query(Scan)

    # Scans are filtered by user email. For guest users, we show mock scans (IDs 1 to 3) as well.
    if email == "guest":
        query_db = query_db.filter((Scan.user_email == "guest") | (Scan.id <= 3))
    else:
        query_db = query_db.filter((Scan.user_email == email) | (Scan.id <= 3))

    # Apply scripts filter
    if scripts:
        # In SQL, check if primary_script contains any of the scripts
        query_db = query_db.filter(
            Scan.primary_script.in_(scripts)
        )

    # Apply confidence filter
    if min_confidence is not None:
        query_db = query_db.filter(Scan.primary_confidence >= min_confidence)

    # Apply search query filter
    if query:
        search_pattern = f"%{query}%"
        query_db = query_db.filter(
            (Scan.primary_script.ilike(search_pattern))
            | (Scan.notes.ilike(search_pattern))
            | (Scan.regions_json.ilike(search_pattern))
        )

    # Apply sorting
    if sort == "oldest":
        query_db = query_db.order_by(Scan.scanned_at.asc())
    elif sort == "highest_confidence":
        query_db = query_db.order_by(Scan.primary_confidence.desc())
    elif sort == "script_az":
        query_db = query_db.order_by(Scan.primary_script.asc())
    else:  # newest
        query_db = query_db.order_by(Scan.scanned_at.desc())

    scans = query_db.all()
    return [map_scan_to_schema(s) for s in scans]


@router.get("/{id}", response_model=ScanRead)
def get_scan(
    id: int,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> ScanRead:
    email = current_user.email if current_user else "guest"

    scan = db.query(Scan).filter(Scan.id == id).first()
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )

    # Authorization check
    if scan.user_email != email and scan.id > 3:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this scan",
        )

    return map_scan_to_schema(scan)


@router.put("/{id}", response_model=ScanRead)
def update_scan(
    id: int,
    request: ScanUpdate,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
) -> ScanRead:
    email = current_user.email if current_user else "guest"

    scan = db.query(Scan).filter(Scan.id == id).first()
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )

    # Authorization check
    if scan.user_email != email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this scan",
        )

    # Update fields
    if request.notes is not None:
        scan.notes = request.notes
    if request.is_saved is not None:
        scan.is_saved = request.is_saved
    if request.is_starred is not None:
        scan.is_starred = request.is_starred

    db.commit()
    db.refresh(scan)

    return map_scan_to_schema(scan)


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan(
    id: int,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
):
    email = current_user.email if current_user else "guest"

    scan = db.query(Scan).filter(Scan.id == id).first()
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )

    # Authorization check
    if scan.user_email != email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this scan",
        )

    db.delete(scan)
    db.commit()
    return status.HTTP_204_NO_CONTENT


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_scans(
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
):
    email = current_user.email if current_user else "guest"

    # Only delete scans created by this specific user, do not delete mock scans (ID <= 3)
    db.query(Scan).filter(Scan.user_email == email, Scan.id > 3).delete()
    db.commit()
    return status.HTTP_204_NO_CONTENT
