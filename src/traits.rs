use crate::types::{Data, Transaction, Customer, Merchant, Terminal, LastTransaction};

pub trait Normalizer {
    fn normalize(&self, data: &Data) -> Vec<f32>;
}

pub trait Loader {
    fn load_data(&self, file_path: &str) -> Result<Vec<Data>, Box<dyn std::error::Error>>;
}
