FROM python:3.9-slim-bullseye

WORKDIR /app

# (선택) 의존성 고정 권장
# COPY requirements.txt .
# RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client \
#   && rm -rf /var/lib/apt/lists/*
# RUN pip install --no-cache-dir -r requirements.txt

# 지금 형태를 유지하려면:
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client \
  && rm -rf /var/lib/apt/lists/*

COPY . /app
COPY ./wait-for-postgres.sh /wait-for-postgres.sh
RUN chmod +x /wait-for-postgres.sh

EXPOSE 80
CMD ["/wait-for-postgres.sh","postgres","uvicorn","app:app","--host","0.0.0.0","--port","80"]
