#[cfg(target_os = "linux")]
mod parse;
mod normalizer;
mod hnsw_static;
mod types;
mod input;
use std::io::Write;
use std::sync::Arc;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::TcpListener;

use crate::parse::DataLoader;
use crate::{
    normalizer::DataNormalizer,
};

#[monoio::main]
async fn main() {
    let train_data_path = std::env::var("DATA_PATH")
        .unwrap_or_else(|_| "src/resources/references.json.gz".to_string());
    let train_data = DataLoader::load_train_data(train_data_path.as_str());
    let knn_model = Arc::new(
        hnsw_static::StaticHNSW::build(16, 40, 20, 5, train_data)
    );
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
                                // body length is fixed: true=35 chars, false=36 chars, fraud_score always X.X (3 chars)
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
