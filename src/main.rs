use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::UnixListener;
use std::os::unix::net::UnixListener as StdUnixListener;

use project::hnsw::StaticHNSW;
use project::input;
use project::normalizer::DataNormalizer;
use project::server::{ConnBuf, RESP_READY, RESP_400, RESPONSES};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn warmup(model: &StaticHNSW, iterations: usize) {
    let t = std::time::Instant::now();
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

fn main() {
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "resources/index.bin".to_string());
    let hnsw_model = Box::new(StaticHNSW::load(&index_path));
    let hnsw_model: &'static StaticHNSW = Box::leak(hnsw_model);
    warmup(hnsw_model, 500);

    let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .with_entries(1024)
        .build()
        .expect("build monoio runtime");

    rt.block_on(async move {
        let sock_path = std::env::var("SOCK_PATH").unwrap_or_else(|_| "/tmp/api.sock".to_string());
        let _ = std::fs::remove_file(&sock_path);
        let std_listener = StdUnixListener::bind(&sock_path)
            .expect("Failed to bind UDS");
        std_listener.set_nonblocking(true).expect("set_nonblocking");
        let listener = UnixListener::from_std(std_listener).expect("from_std");
        println!("Server running on {}", sock_path);

        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let hnsw_model = hnsw_model;
            monoio::spawn(async move {
                let mut bufs = ConnBuf::acquire();

                loop {
                    bufs.read.clear();

                    let (res, returned) = stream.read(std::mem::take(&mut bufs.read)).await;
                    bufs.read = returned;

                    let n = match res {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    let raw = &bufs.read[..n];

                    macro_rules! write_response {
                        ($src:expr) => {{
                            bufs.write.clear();
                            bufs.write.extend_from_slice($src);
                            let (res, returned) = stream.write_all(std::mem::take(&mut bufs.write)).await;
                            bufs.write = returned;
                            if res.is_err() { break; }
                        }};
                    }

                    if raw.starts_with(b"GET /ready") {
                        write_response!(RESP_READY);
                    } else if raw.starts_with(b"POST /fraud-score") {
                        if let Some(body_start) = memchr::memmem::find(raw, b"\r\n\r\n") {
                            let body = &raw[body_start + 4..];
                            match input::parse_payload(body) {
                                Some(data) => {
                                    let q = DataNormalizer.normalize(&data);
                                    let fraud_count = hnsw_model.predict(q);
                                    write_response!(RESPONSES[fraud_count]);
                                }
                                None => {
                                    write_response!(RESP_400);
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}
