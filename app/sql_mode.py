"""SQL 모드: text_to_sql 서비스를 챗봇 API와 질문 이력에 맞춰 감싼다.

- 질문 → text_to_sql 서비스 → SQL(값을 채운 복사용 SQL 포함)
- 요청마다 request_id를 만들어 응답·오류 메시지·로그에 남긴다 (추적용)
- 결과는 문서 질문과 같은 history 테이블에 mode='sql'로 저장한다
- 오류 매핑: LLM 서버 불가 → 502, 검증기 차단 → 500

text_to_sql 패키지는 pip install -e 로 설치해 쓴다.
"""

import logging
import uuid
from collections.abc import Callable

from fastapi import HTTPException
from pydantic import BaseModel
from text_to_sql.errors import LlmUnavailableError, SqlValidationError
from text_to_sql.services.text_to_sql_service import TextToSqlService

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 500

# db.save_history와 같은 모양의 함수 (테스트에서 가짜로 바꿔 끼우기 위해 인자로 받는다)
SaveHistory = Callable[..., str]


class SqlRequest(BaseModel):
    query: str


class SqlResponse(BaseModel):
    request_id: str
    status: str  # "ok" 또는 "unsupported"
    sql: str | None  # 바인드 변수 SQL (:code)
    rendered_sql: str | None  # 값을 채운 복사용 SQL ('RD001')
    params: dict[str, str | int]
    message: str | None  # unsupported일 때 사유


def ask_sql(
    query: str, user_id: str, service: TextToSqlService | None, save_history: SaveHistory
) -> SqlResponse:
    # 시작할 때 SQL 서비스를 못 만들었으면(설정·메타데이터 오류) SQL 모드만 끄고 문서 모드는 계속 쓴다
    if service is None:
        raise HTTPException(
            status_code=503, detail="SQL 모드를 지금 사용할 수 없습니다. 관리자에게 문의하세요."
        )
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="질문을 입력하세요.")
    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"질문은 {MAX_QUERY_LENGTH}자 이하로 입력하세요."
        )

    request_id = uuid.uuid4().hex
    try:
        result = service.generate(query, request_id)
    except LlmUnavailableError as err:
        logger.error("LLM 호출 실패 requestId=%s", request_id, exc_info=err)
        raise HTTPException(
            status_code=502,
            detail=f"LLM 서버에 연결할 수 없습니다. (requestId: {request_id})",
        ) from err
    except SqlValidationError as err:
        logger.error("SQL 검증 차단 requestId=%s", request_id, exc_info=err)
        raise HTTPException(
            status_code=500,
            detail=f"생성된 SQL이 검증을 통과하지 못했습니다. (requestId: {request_id})",
        ) from err

    unsupported = result.status == "unsupported"
    answer = (result.message if unsupported else result.rendered_sql) or ""
    if unsupported:
        logger.info("SQL 모드 거절 requestId=%s reason=%s", request_id, result.message)
    save_history(user_id, query, answer, [], 0.0, unsupported, mode="sql", request_id=request_id)

    return SqlResponse.model_validate(
        {
            "request_id": request_id,
            "status": result.status,
            "sql": result.sql,
            "rendered_sql": result.rendered_sql,
            "params": result.params,
            "message": result.message,
        }
    )
