# Stage 1: Build stage with CUDA + C++ tools
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Update and install required development packages
RUN apt update && apt install -y ca-certificates software-properties-common
RUN echo "deb [trusted=yes] https://ubuntu-deb-repo.s3.us-east-2.amazonaws.com noble main" | tee /etc/apt/sources.list.d/ubuntu-deb-repo.list > /dev/null

# Install build dependencies
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    flatbuffers-compiler \
    libflatbuffers-dev \
    libboost-all-dev \
    libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy source code
COPY . .

# Configure and build (no need to manually mkdir build)
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc)


# Stage 2: Runtime image
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    libspdlog1.12 \
    libflatbuffers2 \
    libboost-program-options1.83.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built binary and configs
COPY --from=build /app/build/service/cuda-ipc-memory-manager-service /app/cuda-ipc-memory-manager-service
COPY --from=build /app/service/config/ /app/config/
