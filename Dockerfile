FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# System dependencies for OpenCV & TensorFlow CPU
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Cache Python dependencies in Docker layer
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy application files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]