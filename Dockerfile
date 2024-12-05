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



# Download the HunyuanVideo model into ckpts directory
RUN huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts

# Set working directory to ckpts and download the llava-llama-3-8b-v1_1-transformers model
WORKDIR /app/HunyuanVideo/ckpts
RUN huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers

# Set working directory back to the main repository and preprocess the text encoder tokenizer
WORKDIR /app/HunyuanVideo
RUN python3 hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
    --input_dir ckpts/llava-llama-3-8b-v1_1-transformers \
    --output_dir ckpts/text_encoder

# Set working directory to ckpts and download clip-vit-large-patch14 model
WORKDIR /app/HunyuanVideo/ckpts
RUN huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2

# Return to the main directory
WORKDIR /app/HunyuanVideo

# Expose the port used by the application
EXPOSE 7860

# Run a default shell to allow manual interaction
# Define ENTRYPOINT
ENTRYPOINT ["/bin/bash", "-c", "DATA_DIR=${DATA_DIRECTORY:-/workspace}; mkdir -p $DATA_DIR && cd $DATA_DIR && python sample_video.py"]
