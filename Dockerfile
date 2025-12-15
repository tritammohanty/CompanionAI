FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential \
    python3 python3-pip python3-venv python3-dev \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# alias
RUN echo 'alias python="python3" ' >> ~/.bashrc
RUN echo 'alias pip="pip3" ' >> ~/.bashrc

CMD ["/bin/bash"]
