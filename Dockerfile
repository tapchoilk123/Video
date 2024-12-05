# Use NVIDIA CUDA 12.6.2 with cuDNN as the base image
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install necessary system tools and Python
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Clone the HunyuanVideo repository
RUN git clone https://github.com/Tencent/HunyuanVideo.git

# Set working directory to the cloned repository
WORKDIR /app/HunyuanVideo

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/Dao-AILab/flash-attention.git@v2.5.9.post1
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Expose the port used by the application
EXPOSE 7860

# Run a default shell to allow manual interaction
CMD ["/bin/bash"]
