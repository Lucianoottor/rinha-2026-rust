use std::cell::RefCell;

pub const RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK";
pub const RESP_400: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

pub const RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

pub struct ConnBuf {
    pub read: Vec<u8>,
    pub write: Vec<u8>,
}

thread_local! {
    static BUF_POOL: RefCell<Vec<ConnBuf>> = RefCell::new(
        (0..16).map(|_| ConnBuf {
            read: Vec::with_capacity(8192),
            write: Vec::with_capacity(128),
        }).collect()
    );
}

impl ConnBuf {
    pub fn acquire() -> Self {
        BUF_POOL.with(|p| {
            p.borrow_mut().pop().unwrap_or_else(|| ConnBuf {
                read: Vec::with_capacity(8192),
                write: Vec::with_capacity(128),
            })
        })
    }
}

impl Drop for ConnBuf {
    fn drop(&mut self) {
        self.read.clear();
        self.write.clear();
        let buf = ConnBuf {
            read: std::mem::take(&mut self.read),
            write: std::mem::take(&mut self.write),
        };
        BUF_POOL.with(|p| p.borrow_mut().push(buf));
    }
}
