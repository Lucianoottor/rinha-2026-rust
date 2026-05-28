FROM rust:latest AS planner
WORKDIR /app
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM rust:latest AS builder
WORKDIR /app
RUN cargo install cargo-chef
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY src ./src
COPY Cargo.toml Cargo.lock* ./

RUN cargo build --release

RUN cp -r src/resources resources && ./target/release/indexer

FROM debian:bookworm-slim
WORKDIR /app

COPY --from=builder /app/target/release/project ./project
COPY --from=builder /app/resources ./resources

CMD ["./project"]
