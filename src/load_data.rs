use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::traits::Loader;
use crate::types::Data;

pub struct DataLoader;

impl Loader for DataLoader {
    fn load_data(&self, file_path: &str) -> Result<Vec<Data>, Box<dyn std::error::Error>> {
        let file_content = std::fs::read_to_string(file_path)?;
        let data: Vec<Data> = serde_json::from_str(&file_content)?;
        Ok(data)
    }
}
