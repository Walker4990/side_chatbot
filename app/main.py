"""웹 서버: 문서 모드(RAG)와 SQL 모드 API, 로그인·관리자 화면

실행: (챗봇 루트에서) uvicorn app.main:app --reload
접속: http://127.0.0.1:8000
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from text_to_sql.config import load_settings
from text_to_sql.errors import ConfigError, MetadataError
from text_to_sql.factory import create_service, start_warm_up

from app import rag, db, auth, sql_mode

# 우리 코드(app, text_to_sql)의 INFO 로그도 서버 로그에 남기기 위한 설정
# (uvicorn 자체 로그는 uvicorn이 따로 설정하므로 겹치지 않는다)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # 요청마다 찍히는 HTTP 로그는 줄인다
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

# 서버가 켜질 때 한 번만 문서를 읽고 임베딩을 준비해 둔다
# sql_service: Text-to-SQL 서비스 (켜질 때 한 번 만들어 계속 쓴다)
state = {"chunks": [], "sql_service": None}




@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    state["chunks"] = rag.build_or_load_embeddings(rag.load_chunks())
    # .env는 위의 rag·db import 때 이미 읽혔다 (load_dotenv). SQL 모드는 로컬 Ollama를 쓴다 (TTS_OLLAMA_*)
    # SQL 서비스 준비가 실패해도 문서 모드는 동작하도록, 오류를 로그로 남기고 SQL 모드만 끈다
    try:
        state["sql_service"] = create_service(load_settings())
    except (ConfigError, MetadataError):
        logger.exception("SQL 모드를 시작하지 못했습니다. /api/sql은 503을 돌려줍니다")
        state["sql_service"] = None
    else:
        # 첫 SQL 질문이 1분 가까이 걸리지 않도록 Ollama를 백그라운드에서 미리 데운다
        start_warm_up(state["sql_service"])
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

class StatusRequest(BaseModel):
    status: str

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest, user: dict = Depends(auth.get_current_user)):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="질문을 입력하세요.")
    result = rag.ask(query, state["chunks"])
    result["top_score"] = float(result["top_score"])

    # 모든 질문은 이력에 저장
    history_id = db.save_history(
        user["id"], query, result["answer"], result["sources"], result["top_score"], result["unanswerable"]
    )
    # 미답변이면 별도로 저장 (어느 이력에서 나왔는지 history_id로 연결)
    if result["unanswerable"]:
        db.save_unanswered(history_id, query, result["top_score"], result["reason"])

    return result


# SQL 모드: 질문 → 상품 DB 조회 SQL (LLM은 의도만 뽑고 SQL은 템플릿으로 조립)
@app.post("/api/sql", response_model=sql_mode.SqlResponse)
def ask_sql(req: sql_mode.SqlRequest, user: dict = Depends(auth.get_current_user)):
    return sql_mode.ask_sql(req.query, user["id"], state["sql_service"], db.save_history)


@app.post("/api/admin/users")
def create_user(req: CreateUserRequest, admin: dict = Depends(auth.require_admin)):
    
    if req.role not in ('admin', 'user'):
        raise HTTPException(status_code=400, detail="권한은 user 혹은 admin을 선택해주세요.")
    check_username = db.get_user_by_username(req.username)

    if check_username is not None:
        raise HTTPException(status_code=409, detail="이미 있는 아이디 입니다.")
    
    password_hash = auth.hash_password(req.password)

    new_id =  db.create_user(req.username, password_hash, req.role)

    return {"id": new_id, "username":req.username, "role": req.role}

# 화면에 다시 불러오는 대화 기간(일). 오래된 기록까지 불러오면 느려진다 (기록 자체는 DB에 남는다)
HISTORY_DAYS = 14


# 로그인한 사용자 본인의 최근 대화. 새로고침해도 대화가 이어지도록 화면이 처음에 불러온다
@app.get("/api/history/me")
def my_history(user: dict = Depends(auth.get_current_user)):
    return db.get_recent_history(user["id"], HISTORY_DAYS)


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(state["chunks"])}


@app.get("/api/admin/history")
def admin_history(admin: dict = Depends(auth.require_admin)):
    return db.get_history()


@app.get("/api/admin/unanswered")
def admin_unanswered(admin: dict = Depends(auth.require_admin)):
    return db.get_unanswered()


# 미답변 처리 상태 변경: 문서를 보완했으면 resolved, 되돌리려면 pending
@app.patch("/api/admin/unanswered/{unanswered_id}")
def update_unanswered(unanswered_id: str, req: StatusRequest, admin: dict = Depends(auth.require_admin)):
    if req.status not in ("pending", "resolved"):
        raise HTTPException(status_code=400, detail="status는 pending 또는 resolved만 가능합니다.")
    if not db.update_unanswered_status(unanswered_id, req.status):
        raise HTTPException(status_code=404, detail="미답변 질문을 찾을 수 없습니다.")
    return {"id": unanswered_id, "status": req.status}

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/api/login")
def login(req: LoginRequest):
    # 아이디가 없거나 비밀번호가 틀리면 401 (어느 쪽이 틀렸는지는 알려 주지 않는다)
    check_user = db.get_user_by_username(req.username)
    if check_user is None or not auth.verify_password(req.password, check_user["password_hash"]):
        raise HTTPException(status_code = 401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    token = auth.create_token(check_user)
    return {"access_token": token,"token_type": "bearer", "role": check_user["role"]}


@app.get("/api/admin/users_list")
def user_list(admin: dict = Depends(auth.require_admin)):
    return db.get_user_list()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
