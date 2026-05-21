#[cfg(target_os = "linux")]
mod parse;
mod types;
mod normalizer;
mod knn;
mod input;
use std::sync::Arc;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::TcpListener;

use crate::parse::DataLoader;
use crate::{
    normalizer::DataNormalizer,
    types::Data
};

#[monoio::main]
async fn main() {
    let train_data_path = std::env::var("DATA_PATH")
        .unwrap_or_else(|_| "src/resources/references.json.gz".to_string());
    let train_data = DataLoader::load_train_data(train_data_path.as_str());
    let knn_model = Arc::new(knn::KNNModel::new(5, train_data));
    let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind to port 8080");
    println!("Server running on http://0.0.0.0:8080");

    loop {
        let (mut stream, _) = listener.accept().await.unwrap();
        let knn_model = Arc::clone(&knn_model);
        monoio::spawn(async move {
            let mut buf = Vec::with_capacity(8192);
            let (res, _buf) = stream.read(buf).await;
            buf = _buf;

            if let Ok(n) = res {
                if n == 0 { return; }
                let raw = &buf[..n];

                if raw.starts_with(b"GET /ready") {
                    let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK".to_vec();
                    let _ = stream.write_all(response).await;
                }

                else if raw.starts_with(b"POST /fraud-score") {
                    if let Some(body_start) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        let body = &raw[body_start + 4..];
                        match input::parse_payload(body) {
                            Some(data) => {
                                let normalized = DataNormalizer.normalize(&data);
                                let fraud_score = knn_model.predict(normalized);
                                let approved = fraud_score < 0.6;
                                let response_body = format!(
                                    "{{\"approved\":{},\"fraud_score\":{:.1}}}",
                                    approved, fraud_score
                                );

                                let response = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                    response_body.len(),
                                    response_body
                                ).into_bytes();

                                let _ = stream.write_all(response).await;
                            }
                            None => {
                                let response = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n".to_vec();
                                let _ = stream.write_all(response).await;
                            }
                        }
                    }
                }
            }
        });
    }
}
