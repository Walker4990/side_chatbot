import os
from datetime import datetime, timedelta, timezone
import bcrypt, jwt
from dotenv import load_dotenv
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, HTTPException

bearer = HTTPBearer()
load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")
TOKEN_HOURS = 8

def hash_password(password: str) -> str:
    password_byte = password.encode()
    hashed = bcrypt.hashpw(password_byte, bcrypt.gensalt())
    return hashed.decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())

def create_token(user: dict) -> str:
    payload = {
        "sub": user["id"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    

def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return {"id": payload["sub"], "role": payload["role"]}

def require_admin(user: dict = Depends(get_current_user)) -> dict:
    # 힌트: user["role"]이 "admin"이 아니면
    #       raise HTTPException(status_code=403, detail="관리자만 사용할 수 있습니다.")
    # 힌트: 맞으면 user를 그대로 return
    if user["role"] != "admin" :
        raise HTTPException(status_code=403, detail="관리자만 사용할 수 있습니다.")
    return user