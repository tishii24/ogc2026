FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    build-essential \
    clang \
    libclang-dev \
    pkg-config \
    python-is-python3 \
    python3.12 \
    python3-pip \
    zip \
    file \
    libc-bin \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal

ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3.12 -m pip install --break-system-packages \
    shapely==2.1.2

WORKDIR /work/ogc2026

CMD ["bash"]
