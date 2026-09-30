"""
RAG 핵심 로직: 청킹 → 임베딩 → 검색 → 답변 → 미답변 판정
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.json"
DOCS_DIR = Path(__file__).parent.parent / "docs"

UNANSWERED_MARKER = "문서에서 답을 찾을 수 없습니다"


# ─── 1단계: 청킹 ──────────────────────────────────────────────

def load_chunks() -> list[dict]:
    """docs/ 폴더의 txt 파일을 빈 줄 기준으로 청킹해 반환."""
    chunks = []
    for txt_file in DOCS_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n")]
        for para in paragraphs:
            if para:  # 빈 조각 제외
                chunks.append({"source": txt_file.name, "text": para})
    return chunks


# ─── 1단계: 임베딩 ────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def build_or_load_embeddings(chunks: list[dict]) -> list[dict]:
    """캐시가 있으면 불러오고, 없으면 API를 호출해 저장."""
    if EMBEDDINGS_PATH.exists():
        print("[임베딩] 캐시에서 불러옵니다.")
        return json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    print(f"[임베딩] API 호출 중 (조각 {len(chunks)}개)...")
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])

    EMBEDDINGS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[임베딩] 완료. 캐시 저장됨.")
    return chunks


# ─── 1단계: 유사도 검색 ───────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """질문과 가장 비슷한 조각을 점수 내림차순으로 반환."""
    q_vec = embed_text(query)
    scored = [
        {**c, "score": cosine_similarity(q_vec, c["embedding"])}
        for c in chunks
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ─── 2단계: 답변 ──────────────────────────────────────────────

def generate_answer(query: str, top_chunks: list[dict]) -> dict:
    """검색된 조각을 근거로 AI 답변 생성. 출처 목록도 함께 반환."""
    context = "\n\n".join(
        f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 사내 문서 기반 질의응답 도우미입니다.\n"
                    "아래 참고 문서의 내용만 근거로 답하세요.\n"
                    f"문서에 근거가 없으면 반드시 '{UNANSWERED_MARKER}'라고만 답하세요.\n"
                    "답변 마지막에 근거로 쓴 출처 파일명을 '[출처: ...]' 형식으로 표시하세요."
                ),
            },
            {
                "role": "user",
                "content": f"[참고 문서]\n{context}\n\n[질문]\n{query}",
            },
        ],
    )
    answer = response.choices[0].message.content.strip()
    sources = list({c["source"] for c in top_chunks})
    return {"answer": answer, "sources": sources}


# ─── 3단계: 미답변 판정 ───────────────────────────────────────

def is_unanswerable(top_chunks: list[dict], answer: str) -> tuple[bool, str]:
    """(미답변 여부, 사유) 반환."""
    if not top_chunks or top_chunks[0]["score"] < SIMILARITY_THRESHOLD:
        return True, f"유사도 점수 부족 (최고 점수: {top_chunks[0]['score']:.3f})" if top_chunks else "검색 결과 없음"
    if UNANSWERED_MARKER in answer:
        return True, "AI가 문서에서 근거를 찾지 못함"
    return False, ""


# ─── 전체 파이프라인 ──────────────────────────────────────────

def ask(query: str, chunks: list[dict]) -> dict:
    """질문 하나를 받아 검색 → 답변 → 미답변 판정까지 처리."""
    top_chunks = search(query, chunks)

    # 1차 판정: 유사도 점수
    if not top_chunks or top_chunks[0]["score"] < SIMILARITY_THRESHOLD:
        reason = f"유사도 점수 부족 (최고: {top_chunks[0]['score']:.3f})" if top_chunks else "검색 결과 없음"
        return {
            "query": query,
            "answer": UNANSWERED_MARKER,
            "sources": [],
            "top_score": top_chunks[0]["score"] if top_chunks else 0.0,
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
