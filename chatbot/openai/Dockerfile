FROM python:3.9-slim

WORKDIR /app

COPY . /app

# requirements.txt 설치
RUN pip install --no-cache-dir -r requirements.txt

# PostgreSQL 클라이언트 설치
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# PostgreSQL 준비 기다리는 스크립트 복사
COPY ./wait-for-postgres.sh /wait-for-postgres.sh
RUN chmod +x /wait-for-postgres.sh

EXPOSE 80

CMD ["/wait-for-postgres.sh", "postgres", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
