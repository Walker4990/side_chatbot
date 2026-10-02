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

EMBEDDINGS_PATH = Path(__file__).parent / "embeddings_practice.json"
DOCS_DIR = Path(__file__).parent.parent / "docs"

UNANSWERED_MARKER = "문서에서 답을 찾을 수 없습니다"


# ─── 1단계: 청킹 ──────────────────────────────────────────────
# 힌트: DOCS_DIR에서 *.txt 파일 목록 가져오기
# 힌트: 파일마다 read_text()로 읽고 "\n\n"으로 split
# 힌트: 각 문단을 {"source": 파일명, "text": 문단} 형태로 chunks에 추가
# 힌트: 빈 문단은 para.strip()으로 걸러내기

def load_chunks() -> list[dict]:
    chunks = []
    files = DOCS_DIR.glob("*.txt")
    for file in files:
        text = file.read_text(encoding="utf-8")
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            if not para.strip():
                continue
            chunks.append({"source": file.name, "text": para.strip()})
    return chunks


# ─── 1단계: 임베딩 ────────────────────────────────────────────
# 힌트: client.embeddings.create()로 API 호출
# 힌트: 결과는 response.data[0].embedding

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding




# 힌트: EMBEDDINGS_PATH.exists()로 캐시 확인
# 힌트: 있으면 json.loads()로 불러와서 바로 return
# 힌트: 없으면 각 chunk마다 embed_text() 호출해서 chunk["embedding"]에 저장
# 힌트: json.dumps()로 변환 후 write_text()로 저장

def build_or_loads(chunks: list[dict]) -> list[dict]:
    if EMBEDDINGS_PATH.exists():
        return json.loads(EMBEDDINGS_PATH.read_text(encoding="UTF-8"))
    for chunk in chunks:
        chunk["embedding"]= embed_text(chunk["text"])
    EMBEDDINGS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent =2), encoding="UTF-8")
    return chunks


# ─── 1단계: 유사도 검색 ───────────────────────────────────────
# 힌트: np.array()로 변환 후
# 힌트: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 힌트: 질문을 embed_text()로 벡터로 변환
# 힌트: 모든 chunk와 cosine_similarity() 계산해서 score 추가
# 힌트: score 내림차순 정렬 후 top_k개 반환

def search(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    q_vector = embed_text(query)
    scored = []
    for chunk in chunks:
        score = cosine_similarity(q_vector, chunk["embedding"])
        scored.append({**chunk, "score":score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ─── 2단계: 답변 ──────────────────────────────────────────────
# 힌트: top_chunks를 "\n\n".join()으로 하나의 문서로 합치기 (context)
# 힌트: client.chat.completions.create()로 GPT 호출
# 힌트: system에 "참고 문서만 보고 답해, 없으면 UNANSWERED_MARKER라고만 답해" 지시
# 힌트: user에 f"[참고 문서]\n{context}\n\n[질문]\n{query}" 전달
# 힌트: {"answer": 답변, "sources": 출처목록} 반환

def generate_answer(query: str, top_chunks: list[dict]) -> dict:
    context = "\n\n".join(f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
           { "role": "system", "content": f"참고 문서 내용만 근거로 답하고, 없으면 '{UNANSWERED_MARKER}' 라고 답변해"},
            {"role": "user", "content" : f"[참고문서]\n{context}\n\n[질문]\n{query}"},
        ]
    )
    answer = response.choices[0].message.content
    sources = list({c["source"] for c in top_chunks})
    return {"answer": answer, "sources": sources}


# ─── 3단계: 미답변 판정 ───────────────────────────────────────
# 힌트: 1차 - top_chunks[0]["score"] < SIMILARITY_THRESHOLD 이면 return True, "사유"
# 힌트: 2차 - UNANSWERED_MARKER in answer 이면 return True, "사유"
# 힌트: 둘 다 통과하면 return False, ""

def is_unanswerable(top_chunks: list[dict], answer: str) -> tuple[bool, str]:
    if top_chunks[0]["score"] < SIMILARITY_THRESHOLD:
        return True, "유사도 점수 부족"
    if UNANSWERED_MARKER in answer:
        return True, "AI가 문서에서 근거를 찾지 못함"
    return False, ""


# ─── 전체 파이프라인 ──────────────────────────────────────────
# 힌트: search() → is_unanswerable() 1차 → generate_answer() → is_unanswerable() 2차
# 힌트: 미답변이면 바로 return, 아니면 답변 결과 return

def ask(query: str, chunks: list[dict]) -> dict:
    top_chunks = search(query, chunks)

    unanswerable, reason = is_unanswerable(top_chunks, "")
    if unanswerable:
        return {
            "query": query,
            "answer": UNANSWERED_MARKER,
            "sources": [],
            "top_score":top_chunks[0]["score"],
            "unanswerable": True,
            "reason": reason
        }
    result = generate_answer(query, top_chunks)

    unanswerable, reason = is_unanswerable(top_chunks, result["answer"])

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "top_score":top_chunks[0]["score"],
        "unanswerable": unanswerable,
        "reason": reason
    }
