use crate::types::RawData;

pub struct DataLoader;

impl DataLoader {
    pub fn load_train_data(file_path: &str) -> Vec<RawData> {
        use std::fs::File;
        use std::io::BufReader;
        use flate2::read::GzDecoder;

        let file = File::open(file_path).expect("Failed to open file");
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        serde_json::from_reader(reader).unwrap()
    }
}
