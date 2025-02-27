# 1) Python 3 베이스 이미지
FROM python:3.11-bullseye

# 2) 컨테이너 내 작업 디렉터리 설정
WORKDIR /flask_server

# 3) requirements만 먼저 복사
COPY flask_server/requirements.txt /flask_server/

# 4) 의존성 설치
RUN pip install --no-cache-dir -r /flask_server/requirements.txt

# 5) 소스코드 복사
COPY flask_server/ /flask_server

# 6) 포트 개방
EXPOSE 8080

# 7) Flask 앱 실행 (working_dir 내에서 실행)
CMD ["python", "app.py"]
