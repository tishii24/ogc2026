FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    musl-tools \
    pkg-config \
    python-is-python3 \
    python3.12 \
    python3-pip \
    zip \
    file \
    libc-bin \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/musl-gcc /usr/local/bin/x86_64-linux-musl-gcc

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal

ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup target add x86_64-unknown-linux-musl

RUN python3.12 -m pip install --break-system-packages \
    shapely==2.1.2

WORKDIR /work/ogc2026

CMD ["bash"]
