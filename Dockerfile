FROM ubuntu:22.04

# Prevent timezone prompt during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the script files
COPY . .


# Create DeepFace weights directory and copy models (moved from prod to base)
RUN mkdir -p /root/.deepface/weights
COPY ./models/age_model_weights.h5 /root/.deepface/weights/
COPY ./models/gender_model_weights.h5 /root/.deepface/weights/
COPY ./models/race_model_single_batch.h5 /root/.deepface/weights/


# Set the default command
CMD ["python3", "prediction_downloader.py"] 