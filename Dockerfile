# Stage 1: Build stage with CUDA + C++ tools
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-program-options1.83-dev \
    libspdlog-dev \
    cppzmq-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    libnvidia-ml-dev \
    libgtest-dev \
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
    libboost-program-options1.83.0 \
    libzmq5 \
    libnvidia-ml-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built binary and configs
COPY --from=build /app/build/service/cuda-ipc-memory-manager-service /app/cuda-ipc-memory-manager-service
COPY --from=build /app/service/config/ /app/config/
COPY --from=build /app/build/example/create-buffer-example /app/example/
