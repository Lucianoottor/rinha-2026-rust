mod normalizer;
mod hnsw_static;
mod types;
mod input;
use std::io::Write;
use std::sync::Arc;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::TcpListener;

use crate::normalizer::DataNormalizer;

/// Run synthetic queries before accepting traffic to:
/// - Pre-allocate the 12 MB VisitedTable (ensure_capacity on first predict call)
/// - Fault in the hot mmap pages (upper layers + high-degree nodes) into OS page cache
/// - Prime CPU branch predictor and instruction cache for the HNSW hot path
fn warmup(model: &hnsw_static::StaticHNSW, iterations: usize) {
    let t = std::time::Instant::now();

    // Phase 1: fault in every mmap page sequentially so real queries never block on I/O
    model.prefetch();
    println!("Prefetch done in {:.0}ms", t.elapsed().as_secs_f64() * 1000.0);

    // Phase 2: HNSW queries to size up QBUFS and warm CPU caches
    let mut rng = 0x517cc1b727220a95u64;
    for _ in 0..iterations {
        let mut v = [0.0f32; 16];
        for x in v[..14].iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *x = (rng >> 11) as f32 / (1u64 << 53) as f32;
        }
        let _ = model.predict(v);
    }
    println!("Warmup done in {:.0}ms total", t.elapsed().as_secs_f64() * 1000.0);
}

#[monoio::main]
async fn main() {
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "resources/index.bin".to_string());
    let knn_model = Arc::new(hnsw_static::StaticHNSW::load(&index_path));

    warmup(&knn_model, 500);

    let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind to port 8080");
    println!("Server running on http://0.0.0.0:8080");

    loop {
        let (mut stream, _) = listener.accept().await.unwrap();
        let knn_model = Arc::clone(&knn_model);
        monoio::spawn(async move {
            stream.set_nodelay(true).ok();
            let mut buf = Vec::with_capacity(8192);

            loop {
                buf.clear();
                let (res, b) = stream.read(buf).await;
                buf = b;

                let n = match res {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                let raw = &buf[..n];

                let close = raw.windows(17).any(|w| w.eq_ignore_ascii_case(b"connection: close"));

                if raw.starts_with(b"GET /ready") {
                    let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK".to_vec();
                    if stream.write_all(response).await.0.is_err() { break; }
                } else if raw.starts_with(b"POST /fraud-score") {
                    if let Some(body_start) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        let body = &raw[body_start + 4..];
                        match input::parse_payload(body) {
                            Some(data) => {
                                let q = DataNormalizer.normalize(&data);
                                let fraud_score = knn_model.predict(q);
                                let approved = fraud_score < 0.6;
                                let body_len = if approved { 35usize } else { 36 };
                                let mut response = Vec::with_capacity(128);
                                let _ = write!(
                                    response,
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: {body_len}\r\n\r\n{{\"approved\":{approved},\"fraud_score\":{fraud_score:.1}}}"
                                );
                                if stream.write_all(response).await.0.is_err() { break; }
                            }
                            None => {
                                let response = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n".to_vec();
                                if stream.write_all(response).await.0.is_err() { break; }
                            }
                        }
                    }
                }

                if close { break; }
            }
        });
    }
}
