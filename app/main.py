"""웹 서버 (4단계)

실행: (side_chatbot 루트에서) uvicorn app.main:app --reload
접속: http://127.0.0.1:8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app import rag, db, auth

STATIC_DIR = Path(__file__).parent.parent / "static"

# 서버가 켜질 때 한 번만 문서를 읽고 임베딩을 준비해 둔다
state = {"chunks": []}




@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    state["chunks"] = rag.build_or_load_embeddings(rag.load_chunks())
    yield


app = FastAPI(title="사내 문서 챗봇", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str

class LoginRequest(BaseModel):
    username: str
    password: str

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    top_score: float
    unanswerable: bool
    reason: str

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest, user: dict = Depends(auth.get_current_user)):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="질문을 입력하세요.")
    result = rag.ask(query, state["chunks"])
    result["top_score"] = float(result["top_score"])

    # 모든 질문은 이력에 저장 (user_id는 5단계 로그인 전까지 None)
    history_id = db.save_history(
        user["id"], query, result["answer"], result["sources"], result["top_score"], result["unanswerable"]
    )
    # 미답변이면 별도로 저장 (어느 이력에서 나왔는지 history_id로 연결)
    if result["unanswerable"]:
        db.save_unanswered(history_id, query, result["top_score"], result["reason"])

    return result


@app.post("/api/admin/users")
def create_user(req: CreateUserRequest, admin: dict = Depends(auth.require_admin)):
    
    if req.role not in ('admin', 'user'):
        raise HTTPException(status_code=403, detail="권한은 user 혹은 admin을 선택해주세요.")
    check_username = db.get_user_by_username(req.username)

    if check_username is not None:
        raise HTTPException(status_code=409, detail="이미 있는 아이디 입니다.")
    
    password_hash = auth.hash_password(req.password)

    new_id =  db.create_user(req.username, password_hash, req.role)

    return {"id": new_id, "username":req.username, "role": req.role}

@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(state["chunks"])}


@app.get("/api/admin/history")
def admin_history(admin: dict = Depends(auth.require_admin)):
    return db.get_history()


@app.get("/api/admin/unanswered")
def admin_unanswered(admin: dict = Depends(auth.require_admin)):
    return db.get_unanswered()

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/login")
def login(req: LoginRequest):
    # 힌트: 1) db.get_user_by_username(req.username)으로 사용자 찾기
    # 힌트: 2) 사용자가 없거나, auth.verify_password(...)가 False면
    #          raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    check_user = db.get_user_by_username(req.username)
    if check_user is None or not auth.verify_password(req.password, check_user["password_hash"]):
        raise HTTPException(status_code = 401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    token = auth.create_token(check_user)
    return {"access_token": token,"token_type": "bearer", "role": check_user["role"]}
    # 힌트: 3) 맞으면 auth.create_token(user)로 토큰 만들기
    # 힌트: 4) return {"access_token": 토큰, "token_type": "bearer", "role": user["role"]}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
