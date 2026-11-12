"""문서 모드(RAG): docs/ 문서를 조각내 임베딩하고, 질문과 비슷한 조각을 근거로 답한다.

임베딩과 답변 생성은 로컬 Ollama를 쓴다 (폐쇄망에서 외부 API 없이 동작).
- 임베딩: POST /api/embed (기본 bge-m3)
- 답변: POST /api/chat (기본 qwen2.5:7b)
"""

import json
import os
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
LLM_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))  # CPU에서는 답변 생성에 수십 초가 걸린다
TOP_K = int(os.getenv("TOP_K", "3"))  # 답변 근거로 쓸 조각 수
# 1차 미답변 판정 점수. 임베딩 모델마다 점수 분포가 달라서 모델을 바꾸면 다시 맞춰야 한다
# (이전 OpenAI 임베딩용 SIMILARITY_THRESHOLD와 섞이지 않도록 이름을 따로 둔다)
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.55"))

EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.json"
DOCS_DIR = Path(__file__).parent.parent / "docs"

UNANSWERED_MARKER = "문서에서 답을 찾을 수 없습니다"
SYSTEM_PROMPT = (
    "참고 문서 내용만 근거로 한국어로 답하세요. "
    f"참고 문서에 답이 없으면 다른 말 없이 '{UNANSWERED_MARKER}'라고만 답하세요."
)

client = httpx.Client(base_url=OLLAMA_URL, timeout=LLM_TIMEOUT)


# ─── 청킹 ─────────────────────────────────────────────────────


def load_chunks() -> list[dict]:
    """docs/ 폴더의 txt 파일을 빈 줄 기준으로 나눠 반환한다.

    반환 형식: [{"source": "파일명.txt", "text": "조각 내용"}, ...]
    """
    chunks = []
    for file in sorted(DOCS_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        for paragraph in text.split("\n\n"):
            if paragraph.strip():
                chunks.append({"source": file.name, "text": paragraph.strip()})
    return chunks


# ─── 임베딩 ───────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> list[list[float]]:
    """여러 문장을 한 번의 요청으로 임베딩한다."""
    response = client.post(
        "/api/embed", json={"model": EMBED_MODEL, "input": texts, "keep_alive": -1}
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def build_or_load_embeddings(chunks: list[dict]) -> list[dict]:
    """캐시(embeddings.json)가 있고 같은 임베딩 모델로 만든 것이면 불러오고, 아니면 새로 만든다.

    다른 모델로 만든 벡터와는 비교할 수 없으므로, 캐시에 모델 이름을 함께 저장해 확인한다.
    반환 형식: chunks에 "embedding" 키가 추가된 리스트
    """
    if EMBEDDINGS_PATH.exists():
        cached = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("model") == EMBED_MODEL:
            return cached["chunks"]

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = vector

    EMBEDDINGS_PATH.write_text(
        json.dumps({"model": EMBED_MODEL, "chunks": chunks}, ensure_ascii=False), encoding="utf-8"
    )
    return chunks


# ─── 유사도 검색 ──────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도."""
    va = np.array(a)
    vb = np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def search(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """질문과 가장 비슷한 조각을 점수 내림차순으로 top_k개 반환한다.

    반환 형식: chunks에 "score" 키가 추가된 리스트
    """
    q_vec = embed_text(query)
    scored = [{**chunk, "score": cosine_similarity(q_vec, chunk["embedding"])} for chunk in chunks]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ─── 답변 ─────────────────────────────────────────────────────


def generate_answer(query: str, top_chunks: list[dict]) -> dict:
    """검색된 조각을 근거로 답변을 만든다.

    반환 형식: {"answer": "...", "sources": ["파일명.txt", ...]}
    """
    context = "\n\n".join(f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks)
    response = client.post(
        "/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[참고 문서]\n{context}\n\n[질문]\n{query}"},
            ],
        },
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"]
    sources = sorted({c["source"] for c in top_chunks})
    return {"answer": answer, "sources": sources}


# ─── 미답변 판정 ──────────────────────────────────────────────


def is_unanswerable(top_chunks: list[dict], answer: str) -> tuple[bool, str]:
    """미답변 여부와 사유를 반환한다.

    1차: 최고 점수가 SIMILARITY_THRESHOLD 미만이면 미답변 (답변 생성 전에 싸게 거른다)
    2차: 답변에 UNANSWERED_MARKER가 있으면 미답변
    """
    if top_chunks[0]["score"] < SIMILARITY_THRESHOLD:
        return True, "유사도 점수 부족"
    if UNANSWERED_MARKER in answer:
        return True, "AI가 문서에서 근거를 찾지 못함"
    return False, ""


# ─── 전체 흐름 ────────────────────────────────────────────────


def ask(query: str, chunks: list[dict]) -> dict:
    top_chunks = search(query, chunks)

    unanswerable, reason = is_unanswerable(top_chunks, "")
    if unanswerable:
        return {
            "query": query,
            "answer": UNANSWERED_MARKER,
            "sources": [],
            "top_score": top_chunks[0]["score"],
            "unanswerable": True,
            "reason": reason,
        }

    result = generate_answer(query, top_chunks)
    unanswerable, reason = is_unanswerable(top_chunks, result["answer"])
    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "top_score": top_chunks[0]["score"],
        "unanswerable": unanswerable,
        "reason": reason,
    }
