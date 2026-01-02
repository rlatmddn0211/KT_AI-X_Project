# 1. 베이스 이미지를 python:3.10으로 변경 (표준 Debian)
FROM python:3.10 

# **** 이 부분에 시스템 라이브러리 설치를 추가했습니다 (패키지 이름은 동일) ****
# OpenCV (cv2) 실행에 필요한 기본 GL/X11 라이브러리 설치
# 표준 이미지로 변경했으므로 이 설치 단계는 이제 훨씬 더 성공적입니다.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libgirepository-1.0-1 \
    libgthread-2.0-0 && \
    rm -rf /var/lib/apt/lists/*
# ******************************************************

# 2. 환경 변수 설정
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR /app

# 3. Python 의존성 설치
COPY requirements.txt .
# (수정됨) 무거운 패키지 먼저, 타임아웃 시간 늘려서 설치 (1000초)
RUN pip install --no-cache-dir --default-timeout=1000 torch==2.4.1 tensorflow==2.20.0

# 나머지 패키지 설치
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 4. 모델 파일 복사 (나머지 단계는 모두 동일)
COPY saved_model/ /app/saved_model/
COPY app/ /app/app/

# 6. 서버 실행 명령어
EXPOSE 8000
CMD ["gunicorn", "app.main:app", \
      "--workers", "1", \
      "--worker-class", "uvicorn.workers.UvicornWorker", \
      "--bind", "0.0.0.0:8000"]
