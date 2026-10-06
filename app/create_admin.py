import sys

from app import auth, db


# 힌트: 1) 입력값이 2개가 아니면 사용법을 출력하고 종료
#          if len(sys.argv) != 3:
#              print("사용법: python -m app.create_admin 아이디 비밀번호")
#              sys.exit(1)
if len(sys.argv) != 3:
    print("사용법: python -m app.create_admin 아이디 비밀번호")
    sys.exit(1)
# 힌트: 2) username, password = sys.argv[1], sys.argv[2]
username, password = sys.argv[1], sys.argv[2]
# 힌트: 3) db.init_db()  (DB 파일이 없을 때를 대비)
db.init_db()
# 힌트: 4) db.get_user_by_username()으로 이미 있는 아이디인지 확인 → 있으면 안내하고 종료
if db.get_user_by_username(username):
    print("사용 중인 아이디입니다.") 
    sys.exit(1)
# 힌트: 5) auth의 함수로 비밀번호 암호화
password_hash = auth.hash_password(password)
# 힌트: 6) db.create_user(아이디, 암호화된 비밀번호, "admin")
admin = db.create_user(username, password_hash, "admin")
# 힌트: 7) 완료 메시지 출력
msg = "관리자 계정 생성 완료"
print(msg)

