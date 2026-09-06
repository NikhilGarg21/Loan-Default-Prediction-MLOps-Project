FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

RUN pip install --upgrade pip && \
    pip install --default-timeout=120 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]