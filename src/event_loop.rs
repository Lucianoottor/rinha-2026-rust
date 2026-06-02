use libc;
use std::os::unix::io::RawFd;
use std::sync::{Arc, Mutex};

use crate::input;
use crate::normalizer::DataNormalizer;
use crate::server::{parse_content_length, RESP_400, RESP_READY, RESPONSES};
use crate::IVF::StaticIVF;


const MAX_FDS: usize = 4096;
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
    fn with_capacity(cap: usize) -> Self {
        Self {
            rbuf: vec![0u8; cap],
            filled: 0,
            resp: b"",
            written: 0,
        }
    }

    fn reset(&mut self) {
        self.filled = 0;
        self.resp = b"";
        self.written = 0;
    }
    #[inline(always)]
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
                
                // Hard validation: Does the client payload fit our fixed-size buffer?
                if body_start + cl > self.rbuf.len() {
                    self.resp = RESP_400;
                    self.written = 0;
                    self.filled = 0;
                    return true;
                }

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

fn drop_conn(epfd: i32, fd: i32, conns: &mut [Option<Conn>]) {
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut()); }
    unsafe { libc::close(fd); }
    if let Some(slot) = conns.get_mut(fd as usize) {
        if let Some(conn) = slot {
            conn.reset(); // Clear trackers but preserve the allocated vector capacity
        }
        *slot = None; // Return slot occupancy back to None
    }
}

fn do_read(fd: i32, conns: &mut [Option<Conn>], model: &StaticIVF) -> ReadResult {
    // O(1) direct index lookup instead of a hashing function
    let conn = match conns.get_mut(fd as usize).and_then(|c| c.as_mut()) {
        Some(c) => c,
        None => return ReadResult::Close,
    };

    // REMOVED internal loop: Read only once per epoll trigger to avoid stalling other sockets
    let room = conn.rbuf.len() - conn.filled;
    if room == 0 {
        // Buffer full but request still incomplete: payload exceeds BUF_CAP capacity
        return ReadResult::Close;
    }

    let n = unsafe {
        libc::recv(fd, conn.rbuf[conn.filled..].as_mut_ptr() as *mut _, room, 0)
    };

    if n > 0 {
        conn.filled += n as usize;
        if conn.process(model) { return ReadResult::Respond; }
        ReadResult::Wait
    } else if n == 0 {
        ReadResult::Close
    } else {
        let e = std::io::Error::last_os_error();
        if e.kind() == std::io::ErrorKind::WouldBlock { return ReadResult::Wait; }
        ReadResult::Close
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

pub fn run(sock_path: &str, model: &'static StaticIVF, notify_rx: RawFd, fd_queue: Arc<Mutex<Vec<RawFd>>>) {
    let lfd = create_listener(sock_path);
    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    assert!(epfd >= 0);
    epoll_add(epfd, lfd, libc::EPOLLIN as u32);
    epoll_add(epfd, notify_rx, libc::EPOLLIN as u32);

    let mut conns: Vec<Option<Conn>> = (0..MAX_FDS)
        .map(|_| Some(Conn::with_capacity(BUF_CAP)))
        .collect();

    let mut conn_pool: Vec<Conn> = conns.drain(..).filter_map(|c| c).collect();
    let mut active_conns: Vec<Option<Conn>> = (0..MAX_FDS).map(|_| None).collect();

    let mut events = [libc::epoll_event { events: 0, u64: 0 }; 256];

    loop {
        let n = unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), 256, -1) };
        if n <= 0 { continue; }

        for i in 0..n as usize {
            let fd = events[i].u64 as i32;
            let ev = events[i].events;

            if fd == notify_rx {
                let mut tmp = [0u8; 64];
                unsafe { libc::recv(notify_rx, tmp.as_mut_ptr() as *mut _, 64, 0) };
                let injected: Vec<RawFd> = fd_queue.lock().unwrap().drain(..).collect();
                for ifd in injected {
                    if ifd as usize >= MAX_FDS { unsafe { libc::close(ifd) }; continue; }
                    if let Some(mut conn) = conn_pool.pop() {
                        conn.reset();
                        active_conns[ifd as usize] = Some(conn);
                        epoll_add(epfd, ifd, READ_FLAGS);
                    } else {
                        unsafe { libc::close(ifd) };
                    }
                }
                continue;
            }

            if fd == lfd {
                loop {
                    let cfd = unsafe {
                        libc::accept4(lfd, std::ptr::null_mut(), std::ptr::null_mut(),
                                      libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC)
                    };
                    if cfd < 0 { break; }
                    
                    if cfd >= MAX_FDS as i32 {
                        unsafe { libc::close(cfd); }
                        continue;
                    }

                    // Zero Allocation: Reuse an existing structured buffer from our pool
                    if let Some(mut pooled_conn) = conn_pool.pop() {
                        pooled_conn.reset();
                        active_conns[cfd as usize] = Some(pooled_conn);
                        epoll_add(epfd, cfd, READ_FLAGS);
                    } else {
                        // Max concurrent connections reached; drop to protect resources
                        unsafe { libc::close(cfd); }
                    }
                }
                continue;
            }

            if ev & (libc::EPOLLHUP | libc::EPOLLERR | libc::EPOLLRDHUP) as u32 != 0 {
                if let Some(Some(conn)) = active_conns.get_mut(fd as usize) {
                    let mut taken_conn = active_conns[fd as usize].take().unwrap();
                    taken_conn.reset();
                    conn_pool.push(taken_conn);
                }
                drop_conn(epfd, fd, &mut active_conns);
                continue;
            }

            if ev & libc::EPOLLIN as u32 != 0 {
                match do_read(fd, &mut active_conns, model) {
                    ReadResult::Close => { 
                        if let Some(Some(_)) = active_conns.get_mut(fd as usize) {
                            let mut taken_conn = active_conns[fd as usize].take().unwrap();
                            taken_conn.reset();
                            conn_pool.push(taken_conn);
                        }
                        drop_conn(epfd, fd, &mut active_conns); 
                        continue; 
                    }
                    ReadResult::Wait  => {}
                    ReadResult::Respond => {
                        if let Some(c) = active_conns[fd as usize].as_mut() {
                            match do_send(fd, c) {
                                SendResult::Done => {}
                                SendResult::Partial => epoll_mod(epfd, fd, WRITE_FLAGS),
                                SendResult::Error => { 
                                    let mut taken_conn = active_conns[fd as usize].take().unwrap();
                                    taken_conn.reset();
                                    conn_pool.push(taken_conn);
                                    drop_conn(epfd, fd, &mut active_conns); 
                                    continue; 
                                }
                            }
                        }
                    }
                }
            }

            if ev & libc::EPOLLOUT as u32 != 0 {
                if let Some(c) = active_conns[fd as usize].as_mut() {
                    match do_send(fd, c) {
                        SendResult::Done => epoll_mod(epfd, fd, READ_FLAGS),
                        SendResult::Partial => {}
                        SendResult::Error => { 
                            let mut taken_conn = active_conns[fd as usize].take().unwrap();
                            taken_conn.reset();
                            conn_pool.push(taken_conn);
                            drop_conn(epfd, fd, &mut active_conns); 
                        }
                    }
                }
            }
        }
    }
}

// Keep create_listener(path) unchanged...
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
