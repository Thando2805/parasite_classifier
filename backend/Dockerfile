FROM python:3.10-slim

WORKDIR /code

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files into the container
COPY . .

# Render defaults to port 10000
EXPOSE 10000

# Run FastAPI using uvicorn binding to Render's PORT environment variable
CMD ["sh", "-c", "uvicorn ${MODULE_PATH:-main:app} --host 0.0.0.0 --port ${PORT:-10000}"]