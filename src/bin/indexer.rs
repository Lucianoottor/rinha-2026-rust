use project::tree;
use project::loader::DataLoader;
use project::types::RawData;

fn label_counts(data: &[RawData]) -> (usize, usize) {
    let fraud = data.iter().filter(|d| d.label == "fraud").count();
    (fraud, data.len() - fraud)
}

fn main() {
    let data_path  = std::env::var("DATA_PATH")
        .unwrap_or_else(|_| "resources/references.json.gz".to_string());
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "resources/index.kdt".to_string());

    println!("Loading training data from {data_path}...");
    let t0 = std::time::Instant::now();
    let train_data = DataLoader::load_train_data(&data_path);
    let (fraud, legit) = label_counts(&train_data);
    println!(
        "Loaded {} records in {:.1}s  (fraud={} {:.1}%, legit={} {:.1}%)",
        train_data.len(), t0.elapsed().as_secs_f64(),
        fraud, 100.0 * fraud as f64 / train_data.len() as f64,
        legit, 100.0 * legit as f64 / train_data.len() as f64,
    );

    println!("Building KD-tree index...");
    let t1 = std::time::Instant::now();
    let model = tree::build(train_data);
    println!("Built in {:.1}s", t1.elapsed().as_secs_f64());

    println!("Saving index to {index_path}...");
    model.save(&index_path);

    let meta = std::fs::metadata(&index_path).unwrap();
    println!("Done. Index file: {:.1} MB", meta.len() as f64 / 1_048_576.0);
}
