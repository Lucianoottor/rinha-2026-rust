use crate::types::{Customer, Data, LastTransaction, Merchant, RawData, Terminal, Transaction};

pub trait Normalizer {
    fn normalize(&self, data: &Data) -> Vec<f32>;
}

pub trait Loader {
    fn load_data(&self, file_path: &str) -> Vec<Data>;
    fn load_train_data(file_path: &str) -> Vec<RawData>;
}
