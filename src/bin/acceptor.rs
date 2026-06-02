use libc::{CMSG_DATA, CMSG_FIRSTHDR, CMSG_SPACE, SCM_RIGHTS, SOL_SOCKET};
use std::net::TcpListener;
use std::os::unix::io::{AsRawFd, IntoRawFd, RawFd};
use std::os::unix::net::UnixStream;

fn connect_ctrl(path: &str) -> Option<UnixStream> {
    for _ in 0..50 {
        if let Ok(s) = UnixStream::connect(path) {
            return Some(s);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    None
}

fn send_fd(ctrl_fd: libc::c_int, fd: RawFd) -> bool {
    let cmsg_space = unsafe { CMSG_SPACE(std::mem::size_of::<libc::c_int>() as u32) } as usize;
    let mut cmsg_buf = vec![0u8; cmsg_space];
    let mut dummy: u8 = 1;
    let mut iov = libc::iovec {
        iov_base: &mut dummy as *mut u8 as *mut libc::c_void,
        iov_len: 1,
    };
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = cmsg_space;
    unsafe {
        let cmsg = CMSG_FIRSTHDR(&msg);
        (*cmsg).cmsg_level = SOL_SOCKET;
        (*cmsg).cmsg_type = SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _;
        *(CMSG_DATA(cmsg) as *mut libc::c_int) = fd;
        libc::sendmsg(ctrl_fd, &msg, 0) > 0
    }
}

fn main() {
    let ctrl1 = std::env::var("CTRL1").unwrap_or_else(|_| "/sockets/api1.sock.ctrl".to_string());
    let ctrl2 = std::env::var("CTRL2").unwrap_or_else(|_| "/sockets/api2.sock.ctrl".to_string());
    let bind  = std::env::var("BIND").unwrap_or_else(|_| "0.0.0.0:9999".to_string());

    let conn1 = connect_ctrl(&ctrl1).unwrap_or_else(|| panic!("cannot connect to {ctrl1}"));
    let conn2 = connect_ctrl(&ctrl2).unwrap_or_else(|| panic!("cannot connect to {ctrl2}"));
    let backends = [conn1.as_raw_fd(), conn2.as_raw_fd()];
    // keep streams alive so fds stay open
    let _keep = (conn1, conn2);

    let listener = TcpListener::bind(&bind).expect("bind 9999");
    println!("Acceptor listening on {bind}");

    let mut rr = 0usize;
    for stream in listener.incoming().flatten() {
        let _ = stream.set_nodelay(true);
        let _ = stream.set_nonblocking(true);
        let fd = stream.into_raw_fd();
        let slot = rr & 1;
        rr = rr.wrapping_add(1);
        if !send_fd(backends[slot], fd) {
            // backend dropped its ctrl connection — try the other
            if !send_fd(backends[slot ^ 1], fd) {
                unsafe { libc::close(fd) };
                continue;
            }
        }
        unsafe { libc::close(fd) };
    }
}
