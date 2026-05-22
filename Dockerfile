FROM rust:latest AS compiler
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=compiler /app/target/release/project ./project
COPY --from=compiler /app/target/release/indexer ./build
COPY src/resources ./resources
EXPOSE 8080
CMD ["./project"]
