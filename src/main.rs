use project::tree::TreeIndex;
use project::event_loop;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn warmup(model: &TreeIndex, iterations: usize) {
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
        .unwrap_or_else(|_| "resources/index.kdt".to_string());

    let model = Box::new(TreeIndex::load(&index_path));
    model.pretouch();
    let model: &'static TreeIndex = Box::leak(model);

    warmup(model, 500);

    let sock_path = std::env::var("SOCK_PATH")
        .unwrap_or_else(|_| "/tmp/api.sock".to_string());

    println!("Server running on {sock_path}");
    event_loop::run(&sock_path, model);
}
