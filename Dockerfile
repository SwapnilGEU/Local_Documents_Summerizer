FROM python:3.12-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Send Python output directly to Docker logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first
# This allows Docker to cache the dependency layer.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY app/ ./app/
COPY streamlit_app.py .

# Copy the RAG data
COPY data/ ./data/

# FastAPI port
EXPOSE 8000

# Default container command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]