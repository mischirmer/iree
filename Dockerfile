FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    clang \
    lld \
    curl \
    ccache \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/iree
COPY . .
RUN git submodule update --init
RUN mkdir -p /workspace/iree/build
WORKDIR /workspace/iree/build



RUN cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_TARGET_BACKENDS=llvm-cpu \
  -DIREE_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  ..

# limit to 2 jobs to reduce memory usage during build
# RUN ninja -j 2

CMD ["/bin/bash"]

# Run with docker build --platform=linux/amd64 -f Dockerfile -t iree-builder-amd64 .