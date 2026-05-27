use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct RawData {
    pub vector: Vec<f32>,
    pub label: String,
}
