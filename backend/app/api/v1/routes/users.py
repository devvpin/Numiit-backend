from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, get_current_admin_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import UserRead

router = APIRouter()


@router.get("/me", response_model=UserRead)
def read_current_user(current_user: User = Depends(get_current_user)) -> UserRead:
    return current_user


@router.get("", response_model=list[UserRead])
def list_users(
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_current_admin_user),
) -> list[UserRead]:
    return db.query(User).order_by(User.id.asc()).all()


@router.put("/{id}/role", response_model=UserRead)
def update_user_role(
    id: int,
    role: str,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_current_admin_user),
) -> UserRead:
    if role not in ["user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be 'user' or 'admin'."
        )
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    user.role = role
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    id: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_current_admin_user),
):
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if user.id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself"
        )
    db.delete(user)
    db.commit()
    return status.HTTP_204_NO_CONTENT

