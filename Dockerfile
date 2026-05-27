FROM rust:latest AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release
# Generate the index during build — self-contained, no pre-built file needed
RUN cp -r src/resources resources && ./target/release/indexer

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/project ./project
COPY --from=builder /app/resources ./resources
CMD ["./project"]
