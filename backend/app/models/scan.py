from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Float, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(String(512), nullable=False)
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    primary_script: Mapped[str] = mapped_column(String(100), nullable=False)
    primary_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_saved: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_starred: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    regions_json: Mapped[str] = mapped_column(Text, nullable=False)
