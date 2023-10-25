# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:latest

# Set a working directory inside the container
WORKDIR /project

# Install git
RUN apt-get update && apt-get install -y git

RUN pip3 install fschat==0.2.23

RUN pip3 install pandas seaborn matplotlib numpy datasets evaluate

RUN pip install anthropic google-generativeai openai