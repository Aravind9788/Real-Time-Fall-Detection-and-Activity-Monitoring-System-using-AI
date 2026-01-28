# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, dlib, and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install dlib (required for face_recognition)
RUN pip install --no-cache-dir dlib

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p registered_faces

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "main.py"]
