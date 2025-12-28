FROM python:3.11-slim-bookworm

WORKDIR /application
COPY . /application

RUN apt update && apt install -y awscli && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "application.py"]
