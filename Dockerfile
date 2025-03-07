FROM pytorch/manylinux-cpu:latest

# 0) 시스템 빌드 도구 설치 (gcc, g++, make, cmake 등)
RUN yum install -y gcc gcc-c++ make cmake

# 0.5) pip, setuptools, wheel 업그레이드
RUN pip3 install --upgrade pip setuptools wheel

# 1) 작업 디렉토리 설정
WORKDIR /flask_server

# 2) requirements 파일 복사
COPY flask_server/requirements.txt /flask_server/requirements.txt

# 3) pip3를 pip로 사용할 수 있도록 심볼릭 링크 생성 (기존 링크가 있으면 덮어씌움)
RUN ln -sf $(which pip3) /usr/local/bin/pip

# python 기본 명령어가 python3를 가리키도록 심볼릭 링크 생성
RUN ln -sf $(which python3) /usr/local/bin/python

# 환경 변수 설정: Flask 앱 정보 및 실행 환경 지정
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# 4) 의존성 설치
RUN pip install --no-cache-dir -r /flask_server/requirements.txt

# 5) 소스코드 복사
COPY flask_server/ /flask_server

# 6) 포트 개방
EXPOSE 8080

# 7) Flask 앱 실행 (working_dir 내에서 실행)
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

