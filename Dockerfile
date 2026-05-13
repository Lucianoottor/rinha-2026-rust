# --- Stage 1: build ---
FROM rust:1.87-slim AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

RUN cargo build --release

# --- Stage 2: run ---
FROM debian:bookworm-slim

WORKDIR /app
COPY --from=builder /app/target/release/project ./project

EXPOSE 8080
CMD ["./project"]
