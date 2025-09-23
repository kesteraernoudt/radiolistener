# Use official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential libsndfile1 libopenblas-dev libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Flask port
EXPOSE 5000

# Set environment variables (optional, can be overridden)
ENV PYTHONUNBUFFERED=1

# Start the app
CMD ["python", "app.py"]