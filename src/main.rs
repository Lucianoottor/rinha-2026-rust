use project::tree::TreeIndex;
use project::event_loop;

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

    println!("Server running on {sock_path}");
    event_loop::run(&sock_path, model);
}
