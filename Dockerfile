FROM rust:latest AS builder
WORKDIR /app

RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/acceptor.rs && \
    echo "fn main() {}" > src/bin/healthcheck.rs && \
    echo "fn main() {}" > src/bin/indexer.rs

COPY Cargo.toml Cargo.lock* ./
ARG RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+fma"
ENV RUSTFLAGS=${RUSTFLAGS}

RUN cargo build --release
RUN rm -f target/release/deps/project* target/release/project* \
          target/release/deps/acceptor* target/release/acceptor* \
          target/release/deps/healthcheck* target/release/healthcheck* \
          target/release/deps/indexer* target/release/indexer*

COPY src ./src

RUN cargo build --release
RUN cp -r src/resources resources && ./target/release/indexer

FROM debian:bookworm-slim
WORKDIR /app

COPY --from=builder /app/target/release/project ./project
COPY --from=builder /app/target/release/healthcheck ./healthcheck
COPY --from=builder /app/target/release/acceptor ./acceptor
COPY --from=builder /app/resources ./resources

CMD ["./project"]
