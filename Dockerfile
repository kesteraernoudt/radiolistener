FROM nvcr.io/nvidia/pytorch:24.03-py3
#FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV TZ=UTC

RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    apt-get install -y --no-install-recommends \
      curl \
      gnupg \
      ca-certificates \
      tzdata \
      ffmpeg \
      pkg-config \
      build-essential \
      libsndfile1 \
      libopenblas-dev \
      libgfortran5 \
      libavformat-dev \
      libavcodec-dev \
      libavdevice-dev \
      libavutil-dev \
      libswresample-dev \
      libswscale-dev \
      libavfilter-dev \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
