use project::loader::DataLoader;
use project::hnsw;

fn main() {
    let data_path = std::env::var("DATA_PATH")
        .unwrap_or_else(|_| "resources/references.json.gz".to_string());
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "/data/index.bin".to_string());

    println!("Loading training data from {}...", data_path);
    let train_data = DataLoader::load_train_data(&data_path);
    let n = train_data.len();
    println!("Loaded {} records", n);

    println!("Building HNSW index (m=16, ef_construction=40)...");
    let model = hnsw::build_index(16, 40, 40, 5, train_data);

    println!("Saving index to {}...", index_path);
    model.save(&index_path);

    println!("Done. Index size: {:.1} MB", {
        let meta = std::fs::metadata(&index_path).unwrap();
        meta.len() as f64 / 1_048_576.0
    });
}
