use project::fd;
use project::tree::TreeIndex;
use project::event_loop;

use std::os::unix::io::{IntoRawFd, RawFd};
use std::os::unix::net::UnixStream as StdUnixStream;
use std::sync::{Arc, Mutex};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "resources/index.kdt".to_string());

    let model = Box::new(TreeIndex::load(&index_path));
    model.pretouch();
    let model: &'static TreeIndex = Box::leak(model);

    let sock_path = std::env::var("SOCK_PATH")
        .unwrap_or_else(|_| "/tmp/api.sock".to_string());

    let (notify_tx_std, notify_rx_std) = StdUnixStream::pair().expect("socketpair");
    notify_rx_std.set_nonblocking(true).unwrap();

    let fd_queue: Arc<Mutex<Vec<RawFd>>> = Arc::new(Mutex::new(Vec::new()));

    let ctrl_path = format!("{}.ctrl", sock_path);
    {
        let fq = fd_queue.clone();
        let nt = Arc::new(notify_tx_std);
        std::thread::spawn(move || fd::control_thread(ctrl_path, fq, nt));
    }

    let notify_rx = notify_rx_std.into_raw_fd();

    println!("Server running on {sock_path}");
    event_loop::run(&sock_path, model, notify_rx, fd_queue);
}
