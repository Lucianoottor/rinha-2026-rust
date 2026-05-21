use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufReader, Read};
use crate::types::{Data, RawData};

pub struct DataLoader;

impl DataLoader {
    pub fn load_data(&self, file_path: &str) -> Vec<Data> {
        let file_content = std::fs::read_to_string(file_path).unwrap();
        let data: Vec<Data> = serde_json::from_str(&file_content).unwrap();
        data
    }
    pub fn load_train_data(file_path: &str) -> Vec<RawData> {
        use std::fs::File;
        use std::io::BufReader;
        use flate2::read::GzDecoder;

        let file = File::open(file_path).expect("Failed to open file");
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        let raw_data: Vec<RawData> = serde_json::from_reader(reader).unwrap();
        raw_data
    }
}

