FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8
ENV PYTHONPATH=/workspace/src

WORKDIR /workspace

# Install system dependencies required for OpenCV, extraction, and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration and requirement files first to leverage Docker layer cache
COPY pyproject.toml README.md ./
COPY requirements/ ./requirements/

# Upgrade build tools and install the required packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements/base.txt && \
    pip install --no-cache-dir transformers==4.57.1 decord lmdb opencv-python

# Copy source directory to complete project setup
COPY . /workspace

# Install the project in editable mode
RUN pip install --no-cache-dir --no-build-isolation -e ".[dev]"

CMD ["bash"]
