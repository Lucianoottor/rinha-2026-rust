use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

fn main() {
    let sock = std::env::var("SOCK_PATH").unwrap_or_else(|_| "/tmp/api.sock".to_string());

    let Ok(mut conn) = UnixStream::connect(&sock) else {
        std::process::exit(1);
    };

    if conn
        .write_all(b"GET /ready HTTP/1.1\r\nHost: hc\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        std::process::exit(1);
    }

    let mut buf = [0u8; 16];
    match conn.read(&mut buf) {
        // HTTP/1.1 200 ... — status code sits at byte offset 9
        Ok(n) if n >= 12 && &buf[9..12] == b"200" => std::process::exit(0),
        _ => std::process::exit(1),
    }
}
