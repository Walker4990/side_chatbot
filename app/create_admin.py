"""관리자 계정 만들기.

사용법: python -m app.create_admin 아이디 비밀번호
"""

import sys

from app import auth, db

if len(sys.argv) != 3:
    print("사용법: python -m app.create_admin 아이디 비밀번호")
    sys.exit(1)
username, password = sys.argv[1], sys.argv[2]

db.init_db()  # 테이블이 없으면 만든다
if db.get_user_by_username(username):
    print("사용 중인 아이디입니다.")
    sys.exit(1)

db.create_user(username, auth.hash_password(password), "admin")
print("관리자 계정 생성 완료")
