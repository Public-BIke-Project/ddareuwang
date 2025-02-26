# 1) Python 3 베이스 이미지
FROM python:3.11-bullseye

# 2) 컨테이너 내 작업 디렉터리 설정 (compose.yaml에서 설정한 working_dir과 동일)
WORKDIR /flask_server

# 3) Flask 앱 소스 복사 (compose.yaml의 volumes 매핑과 동일한 경로)
COPY flask_server/ /flask_server

# 4) 디버깅: COPY 결과 확인 -> 배포단계에서는 제거 가능
RUN ls -R /flask_server

# 5) 의존성 설치
RUN pip install --no-cache-dir -r /flask_server/requirements.txt

# 6) 포트 개방
EXPOSE 8080

# 7) Flask 앱 실행 (working_dir 내에서 실행)
CMD ["python", "app.py"]
