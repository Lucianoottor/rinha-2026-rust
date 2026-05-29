use std::collections::HashMap;

use libc;

use crate::input;
use crate::normalizer::DataNormalizer;
use crate::server::{parse_content_length, RESP_400, RESP_READY, RESPONSES};
use crate::IVF::StaticIVF;

const BUF_CAP: usize = 8192;
const READ_FLAGS: u32 = (libc::EPOLLIN | libc::EPOLLRDHUP) as u32;
const WRITE_FLAGS: u32 = (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLRDHUP) as u32;

struct Conn {
    rbuf: Vec<u8>,
    filled: usize,
    resp: &'static [u8],
    written: usize,
}

impl Conn {
    fn new() -> Self {
        Self { rbuf: vec![0u8; BUF_CAP], filled: 0, resp: b"", written: 0 }
    }

    fn process(&mut self, model: &StaticIVF) -> bool {
        let raw = &self.rbuf[..self.filled];

        if raw.starts_with(b"GET /ready") {
            self.resp = RESP_READY;
            self.written = 0;
            self.filled = 0;
            return true;
        }

        if raw.starts_with(b"POST /fraud-score") {
            if let Some(hdr_end) = memchr::memmem::find(raw, b"\r\n\r\n") {
                let cl = parse_content_length(&raw[..hdr_end]).unwrap_or(0);
                let body_start = hdr_end + 4;
                if self.filled >= body_start + cl {
                    let body = &self.rbuf[body_start..body_start + cl];
                    self.resp = match input::parse_payload(body) {
                        Some(data) => {
                            let q = DataNormalizer.normalize(&data);
                            RESPONSES[model.predict(q)]
                        }
                        None => RESP_400,
                    };
                    self.written = 0;
                    self.filled = 0;
                    return true;
                }
            }
        }

        false
    }
}

enum ReadResult { Respond, Close, Wait }
enum SendResult { Done, Partial, Error }

fn epoll_add(epfd: i32, fd: i32, events: u32) {
    let mut ev = libc::epoll_event { events, u64: fd as u64 };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev); }
}

fn epoll_mod(epfd: i32, fd: i32, events: u32) {
    let mut ev = libc::epoll_event { events, u64: fd as u64 };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_MOD, fd, &mut ev); }
}

fn drop_conn(epfd: i32, fd: i32, conns: &mut HashMap<i32, Conn>) {
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut()); }
    unsafe { libc::close(fd); }
    conns.remove(&fd);
}

fn do_read(fd: i32, conns: &mut HashMap<i32, Conn>, model: &StaticIVF) -> ReadResult {
    let conn = match conns.get_mut(&fd) {
        Some(c) => c,
        None => return ReadResult::Close,
    };

    loop {
        if conn.filled == conn.rbuf.len() {
            conn.rbuf.resize(conn.rbuf.len() + BUF_CAP, 0);
        }
        let room = conn.rbuf.len() - conn.filled;

        let n = unsafe {
            libc::recv(fd, conn.rbuf[conn.filled..].as_mut_ptr() as *mut _, room, 0)
        };

        if n > 0 {
            conn.filled += n as usize;
            if conn.process(model) { return ReadResult::Respond; }
        } else if n == 0 {
            return ReadResult::Close;
        } else {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::WouldBlock { return ReadResult::Wait; }
            return ReadResult::Close;
        }
    }
}

fn do_send(fd: i32, conn: &mut Conn) -> SendResult {
    while conn.written < conn.resp.len() {
        let rest = &conn.resp[conn.written..];
        let n = unsafe {
            libc::send(fd, rest.as_ptr() as *const _, rest.len(), libc::MSG_NOSIGNAL)
        };
        if n > 0 {
            conn.written += n as usize;
        } else {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::WouldBlock { return SendResult::Partial; }
            return SendResult::Error;
        }
    }
    SendResult::Done
}

pub fn run(sock_path: &str, model: &'static StaticIVF) {
    let lfd = create_listener(sock_path);
    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    assert!(epfd >= 0);
    epoll_add(epfd, lfd, libc::EPOLLIN as u32);

    let mut conns: HashMap<i32, Conn> = HashMap::new();
    let mut events = [libc::epoll_event { events: 0, u64: 0 }; 256];

    loop {
        let n = unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), 256, -1) };
        if n <= 0 { continue; }

        for i in 0..n as usize {
            let fd = events[i].u64 as i32;
            let ev = events[i].events;

            if fd == lfd {
                loop {
                    let cfd = unsafe {
                        libc::accept4(lfd, std::ptr::null_mut(), std::ptr::null_mut(),
                                      libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC)
                    };
                    if cfd < 0 { break; }
                    conns.insert(cfd, Conn::new());
                    epoll_add(epfd, cfd, READ_FLAGS);
                }
                continue;
            }

            if ev & (libc::EPOLLHUP | libc::EPOLLERR | libc::EPOLLRDHUP) as u32 != 0 {
                drop_conn(epfd, fd, &mut conns);
                continue;
            }

            if ev & libc::EPOLLIN as u32 != 0 {
                match do_read(fd, &mut conns, model) {
                    ReadResult::Close => { drop_conn(epfd, fd, &mut conns); continue; }
                    ReadResult::Wait  => {}
                    ReadResult::Respond => {
                        if let Some(c) = conns.get_mut(&fd) {
                            match do_send(fd, c) {
                                SendResult::Done    => {}
                                SendResult::Partial => epoll_mod(epfd, fd, WRITE_FLAGS),
                                SendResult::Error   => { drop_conn(epfd, fd, &mut conns); continue; }
                            }
                        }
                    }
                }
            }

            if ev & libc::EPOLLOUT as u32 != 0 {
                if let Some(c) = conns.get_mut(&fd) {
                    match do_send(fd, c) {
                        SendResult::Done    => epoll_mod(epfd, fd, READ_FLAGS),
                        SendResult::Partial => {}
                        SendResult::Error   => { drop_conn(epfd, fd, &mut conns); }
                    }
                }
            }
        }
    }
}

fn create_listener(path: &str) -> i32 {
    let fd = unsafe {
        libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC, 0)
    };
    assert!(fd >= 0);

    let _ = std::fs::remove_file(path);

    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_bytes();
    assert!(bytes.len() < addr.sun_path.len());
    for (i, &b) in bytes.iter().enumerate() {
        addr.sun_path[i] = b as libc::c_char;
    }

    let r = unsafe {
        libc::bind(fd, &addr as *const _ as *const libc::sockaddr,
                   std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t)
    };
    assert_eq!(r, 0);

    let cpath = std::ffi::CString::new(path).unwrap();
    unsafe { libc::chmod(cpath.as_ptr(), 0o666); }

    assert_eq!(unsafe { libc::listen(fd, 1024) }, 0);

    fd
}
