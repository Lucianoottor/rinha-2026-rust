use project::IVF::build_ivf;
use project::loader::DataLoader;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let data_path  = std::env::var("DATA_PATH")
        .unwrap_or_else(|_| "resources/references.json.gz".to_string());
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "resources/index.ivf".to_string());

    // Tunable via environment variables — rebuild index to change.
    let k_centroids  = env_usize("K_CENTROIDS",  1024);
    let nprobe       = env_usize("NPROBE",           4);
    let k            = env_usize("K",                5);  // must match RESPONSES length
    let kmeans_iters = env_usize("KMEANS_ITERS",    10);

    println!("Loading training data from {data_path}...");
    let t0 = std::time::Instant::now();
    let train_data = DataLoader::load_train_data(&data_path);
    println!("Loaded {} records in {:.1}s", train_data.len(), t0.elapsed().as_secs_f64());

    let model = build_ivf(k_centroids, nprobe, k, kmeans_iters, train_data);

    println!("Saving index to {index_path}...");
    model.save(&index_path);

    let meta = std::fs::metadata(&index_path).unwrap();
    println!("Done. Index file: {:.1} MB", meta.len() as f64 / 1_048_576.0);
}
