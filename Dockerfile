# 1) Python 3.9 베이스 이미지
FROM python:3.9

# 2) 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3) Flask 앱 소스 복사
#    (flask_server 폴더 안에 app.py, requirements.txt 등이 있다고 가정)
COPY flask_server /app

# 4) 의존성 설치 (requirements.txt 사용)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5) Flask 앱에서 사용할 포트 개방
EXPOSE 8080

# 6) 앱 실행
CMD ["python", "app.py"]
