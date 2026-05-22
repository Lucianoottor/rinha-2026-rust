FROM rust:latest AS compiler
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim AS index-builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgcc-s1 && rm -rf /var/lib/apt/lists/*
COPY --from=compiler /app/target/release/indexer ./indexer
COPY src/resources/references.json.gz ./references.json.gz
RUN DATA_PATH=references.json.gz INDEX_PATH=index.bin ./indexer

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgcc-s1 && rm -rf /var/lib/apt/lists/*
COPY --from=compiler /app/target/release/project ./project
COPY --from=index-builder /app/index.bin ./index.bin
EXPOSE 8080
ENV INDEX_PATH=/app/index.bin
CMD ["./project"]
