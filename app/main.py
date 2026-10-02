"""웹 서버 (4단계)

실행: (side_chatbot 루트에서) uvicorn app.main:app --reload
접속: http://127.0.0.1:8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import rag, db

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


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    top_score: float
    unanswerable: bool
    reason: str


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="질문을 입력하세요.")
    result = rag.ask(query, state["chunks"])
    result["top_score"] = float(result["top_score"])

    # 모든 질문은 이력에 저장 (user_id는 5단계 로그인 전까지 None)
    history_id = db.save_history(
        None, query, result["answer"], result["sources"], result["top_score"], result["unanswerable"]
    )
    # 미답변이면 별도로 저장 (어느 이력에서 나왔는지 history_id로 연결)
    if result["unanswerable"]:
        db.save_unanswered(history_id, query, result["top_score"], result["reason"])

    return result



@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(state["chunks"])}


# ─── 관리자 조회 (5단계에서 관리자만 접근하도록 막을 예정) ─────
@app.get("/api/admin/history")
def admin_history():
    return db.get_history()


@app.get("/api/admin/unanswered")
def admin_unanswered():
    return db.get_unanswered()

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
