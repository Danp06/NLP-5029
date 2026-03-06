FROM python:3.10-slim

WORKDIR /app

# Copiar metadatos primero para aprovechar la caché de capas
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copiar código fuente
COPY src/ ./src/

CMD ["python", "-m", "src"]
