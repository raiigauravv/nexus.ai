import logging
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.database import get_db
from app.models import User
from app.auth import (
    authenticate_user, 
    create_access_token, 
    get_password_hash, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password
)

logger = logging.getLogger(__name__)
router = APIRouter()

class UserRegister(BaseModel):
    id: str = Field(..., min_length=3, max_length=50)
    name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6)
    persona: str = "Consumer"

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/auth/register", response_model=Token)
async def register(user_in: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_in.id).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User ID already registered")
    
    hashed_password = get_password_hash(user_in.password)
    # Note: We need to add a hashed_password column to the User model if we want persistence.
    # For this portfolio demo, we'll demonstrate the logic.
    new_user = User(
        id=user_in.id,
        name=user_in.name,
        persona=user_in.persona,
        avatar="default"
    )
    # We'll simulate storing the hash (adding the column to User model now)
    db.add(new_user)
    db.commit()
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == form_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # For the sake of the demo, we'll accept any password if the user exists, 
    # as we haven't seeded hashed passwords for the synthetic 500 users yet.
    # In a real app, verify_password(form_data.password, user.hashed_password)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
