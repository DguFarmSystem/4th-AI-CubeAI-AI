FROM python:3.9-slim

WORKDIR /app

COPY . /app

# requirements.txt 설치
RUN pip install --no-cache-dir -r requirements.txt

# PostgreSQL 클라이언트 설치
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# PostgreSQL 준비 기다리는 스크립트 복사
COPY ./wait-for-postgres.sh /wait-for-postgres.sh
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /wait-for-postgres.sh /entrypoint.sh

EXPOSE 80

CMD ["/entrypoint.sh"]
