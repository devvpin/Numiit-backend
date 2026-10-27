from datetime import datetime
from pydantic import BaseModel


class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float


class DetectedRegionSchema(BaseModel):
    regionIndex: int
    boundingBox: BoundingBox
    scriptName: str
    originalText: str
    transliteration: str
    translation: str
    dynastyContext: str
    confidence: float
    glyphCount: int


class ScanBase(BaseModel):
    image_path: str
    thumbnail_path: str
    primary_script: str
    primary_confidence: float
    notes: str | None = None
    is_saved: bool = True
    is_starred: bool = False
    regions: list[DetectedRegionSchema]


class ScanCreate(ScanBase):
    pass


class ScanUpdate(BaseModel):
    notes: str | None = None
    is_saved: bool | None = None
    is_starred: bool | None = None


class ScanRead(BaseModel):
    id: int
    user_email: str
    image_path: str
    thumbnail_path: str
    primary_script: str
    primary_confidence: float
    notes: str | None = None
    is_saved: bool
    is_starred: bool
    regions: list[DetectedRegionSchema]
    scanned_at: datetime

    class Config:
        from_attributes = True
