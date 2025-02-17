# # 기본 Python 이미지 선택
# FROM python:3.9-slim

# # 작업 디렉터리 설정
# WORKDIR /app

# # Poetry 설치 및 가상환경 비활성화 설정
# RUN pip install poetry && poetry config virtualenvs.create true

# # 프로젝트 파일 복사
# COPY ./pyproject.toml ./poetry.lock ./

# # 의존성 설치
# RUN poetry install --no-root --only main

# # 앱 코드 복사
# COPY . .

# # 컨테이너가 사용할 포트 지정
# EXPOSE 8080

# # Flask 실행
# CMD ["poetry", "run", "python", "flask_server/app.py"]

FROM python:3.9-slim
RUN pip install poetry
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-root
COPY . /app
CMD ["poetry", "run", "python", "flask_server/app.py"]

