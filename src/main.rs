use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::TcpListener;
use serde_json;

mod load_data;
mod traits;
mod types;
mod normalizer;

use crate::{
    normalizer::DataNormalizer, 
    traits::Normalizer, 
    types::Data
};

#[monoio::main]
async fn main() {
    // Bind to port 9999 as requested
    let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind to port 8080");
    println!("Server running on http://0.0.0.0:8080");

    loop {
        let (mut stream, _) = listener.accept().await.unwrap();
        
        monoio::spawn(async move {
            let mut buf = Vec::with_capacity(8192);
            let (res, _buf) = stream.read(buf).await;
            buf = _buf;

            if let Ok(n) = res {
                if n == 0 { return; }
                let request_str = String::from_utf8_lossy(&buf[..n]);
                
                // --- Simple Routing Logic ---

                // 1. GET /ready
                if request_str.starts_with("GET /ready") {
                    let response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
                        .to_string()
                        .into_bytes();
                    let _ = stream.write_all(response).await;
                } 
                
                // 2. POST / (Your Normalizer)
                else if request_str.starts_with("POST /fraud-score") {
                    if let Some(body_start) = request_str.find("\r\n\r\n") {
                        let body = &request_str[body_start + 4..];

                        match serde_json::from_str::<Data>(body) {
                            Ok(incoming_data) => {
                                let normalizer = DataNormalizer::new();
                                let normalized = normalizer.normalize(&incoming_data);
                                let response_body = serde_json::to_string(&normalized).unwrap();
                                
                                let response = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                    response_body.len(),
                                    response_body
                                ).into_bytes();
                                
                                let _ = stream.write_all(response).await;
                            }
                            Err(_) => {
                                let response = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"
                                    .to_string()
                                    .into_bytes();
                                let _ = stream.write_all(response).await;
                            }
                        }
                    }
                } 
                
                // 3. 404 Not Found
                else {
                    let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
                        .to_string()
                        .into_bytes();
                    let _ = stream.write_all(response).await;
                }
            }
        });
    }
}