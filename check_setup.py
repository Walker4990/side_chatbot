"""
0단계 완료 확인 스크립트.
API 키가 올바르면 임베딩 숫자 묶음 길이를 출력합니다.
"""

from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY", "")
if not key or key.startswith("sk-..."):
    print("❌ .env 파일에 OPENAI_API_KEY를 설정하세요 (.env.example 참고)")
    raise SystemExit(1)

from openai import OpenAI
client = OpenAI(api_key=key)

response = client.embeddings.create(
    model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    input="환경 설정 확인용 테스트 문장입니다.",
)
vec = response.data[0].embedding
print(f"✅ 임베딩 성공! 숫자 묶음 길이: {len(vec)}")
print("0단계 완료 — 1단계를 시작할 수 있습니다.")
