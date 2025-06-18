FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core Python libraries
RUN pip install --upgrade pip setuptools wheel

# ✅ Install required libraries with known compatible versions
RUN pip install \
    numpy==1.23.5 \
    pandas==1.3.5 \
    scikit-learn==0.24.2 \
    protobuf==3.20.3 \
    sagemaker-training==4.4.10

# ✅ Copy training script
COPY train.py /opt/ml/code/train.py

# ✅ Set SageMaker entry point
ENV SAGEMAKER_PROGRAM train.py
