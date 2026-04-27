import logging
import re
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.database import get_db
from app.models import User
from app.auth import (
    create_access_token, 
    get_password_hash, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password,
    get_current_user
)
from app.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)
router = APIRouter()

LOGIN_LIMIT_PER_MINUTE = 10
REGISTER_LIMIT_PER_5_MIN = 5


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request, action: str, limit: int, window: int) -> None:
    key = f"{action}:{_client_ip(request)}"
    allowed, retry_after = rate_limiter.allow(key, limit=limit, window_seconds=window)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many {action} attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


def _validate_registration_input(user_in: "UserRegister") -> None:
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", user_in.id):
        raise HTTPException(
            status_code=422,
            detail="User ID must be 3-50 chars and contain only letters, numbers, underscores, or hyphens.",
        )

    pwd = user_in.password
    if len(pwd) < 10 or not re.search(r"[A-Z]", pwd) or not re.search(r"[a-z]", pwd) or not re.search(r"\d", pwd):
        raise HTTPException(
            status_code=422,
            detail="Password must be at least 10 characters and include uppercase, lowercase, and a number.",
        )

class UserRegister(BaseModel):
    id: str = Field(..., min_length=3, max_length=50)
    name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6)
    persona: str = "Consumer"

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/auth/register", response_model=Token)
async def register(user_in: UserRegister, request: Request, db: Session = Depends(get_db)):
    _enforce_rate_limit(request, "register", REGISTER_LIMIT_PER_5_MIN, 300)
    _validate_registration_input(user_in)

    db_user = db.query(User).filter(User.id == user_in.id).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User ID already registered")
    
    hashed_pw = get_password_hash(user_in.password)
    new_user = User(
        id=user_in.id,
        name=user_in.name,
        persona=user_in.persona,
        avatar="default",
        hashed_password=hashed_pw
    )
    db.add(new_user)
    db.commit()
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/auth/login", response_model=Token)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    _enforce_rate_limit(request, "login", LOGIN_LIMIT_PER_MINUTE, 60)

    user = db.query(User).filter(User.id == form_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Real password verification
    if not user.hashed_password or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

class UserResponse(BaseModel):
    id: str
    name: str
    persona: str
    avatar: str

    class Config:
        from_attributes = True

@router.get("/auth/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

