# Python 3.9 기반 이미지 사용
FROM python:3.9

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Flask 애플리케이션 실행 (Gunicorn 사용)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "main:app"]
