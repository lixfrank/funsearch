FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

WORKDIR /workspace

# Copy build and dependency files first for better layer caching
COPY pyproject.toml README.md install.sh ./
COPY requirements*.txt ./

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# Install dependencies using either uv or pip based on build arg
# Build with: docker build --build-arg USE_UV=true -t funsearch .
ARG USE_UV=false
RUN if [ "$USE_UV" = "true" ]; then \
        pip install --upgrade pip && \
        pip install uv && \
        uv pip install --system -r requirements.txt; \
    else \
        pip install --upgrade pip && \
        pip install -r requirements.txt; \
    fi

# Create necessary subfolders in data directory  
RUN mkdir -p ./data && \
    cd ./data && \
    mkdir -p scores graphs backups && \
    cd ..


RUN ln -s /usr/bin/python3 /usr/bin/python
# Copy application code
# COPY examples ./examples
# COPY funsearch ./funsearch

# Install the application
# RUN if [ "$USE_UV" = "true" ]; then \
#         uv pip install --system --no-deps .; \
#     else \
#         pip install --no-deps .; \
#     fi && \
#     rm -r ./funsearch ./build

CMD ["bash"]
