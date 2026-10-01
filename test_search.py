"""eval/test_questions.txt 의 질문을 한 번에 돌려 보는 스크립트.

실행: python test_search.py            (app/rag.py 사용)
      python test_search.py practice   (app/rag_practice.py 사용)
"""
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "practice":
    from app import rag_practice as rag
    build = rag.build_or_loads
else:
    from app import rag
    build = rag.build_or_load_embeddings

QUESTIONS_PATH = Path(__file__).parent / "eval" / "test_questions.txt"

chunks = build(rag.load_chunks())

expect_answer = True
for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("#"):
        # "# 답변이 안 나와야 하는 질문" 줄부터는 미답변이 정답
        expect_answer = "안 나와야" not in line
        continue

    result = rag.ask(line, chunks)
    ok = result["unanswerable"] != expect_answer
    print(f"\n[{'OK' if ok else 'XX'}] 질문: {line}")
    print(f"답변: {result['answer']}")
    print(f"점수: {result['top_score']:.3f}  미답변: {result['unanswerable']}  사유: {result['reason']}")
