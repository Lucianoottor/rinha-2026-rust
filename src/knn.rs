use std::num::NonZero;
use crate::types::RawData;
use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;

pub struct KNNModel {
    pub k: usize,
    pub tree: ImmutableKdTree<f32, 14>,
    pub is_fraud: Vec<bool>,
}

impl KNNModel {
    pub fn new(k: usize, raw_training_data: Vec<RawData>) -> Self {
        let mut entries: Vec<[f32; 14]> = Vec::with_capacity(raw_training_data.len());
        let mut is_fraud = Vec::with_capacity(raw_training_data.len());

        for raw in raw_training_data {
            let point: [f32; 14] = raw.vector.try_into().unwrap();
            entries.push(point);
            is_fraud.push(raw.label == "fraud");
        }
        let tree = ImmutableKdTree::new_from_slice(&entries);
        Self { k, tree, is_fraud }
    }

    pub fn predict(&self, input: Vec<f32>) -> f32 {
        let point: [f32; 14] = input.try_into().expect("input must have 14 elements");
        let neighbors = self.tree.nearest_n::<SquaredEuclidean>(&point, NonZero::new(self.k).unwrap());

        let fraud_count = neighbors.iter()
            .filter(|n| self.is_fraud[n.item as usize])
            .count();

        fraud_count as f32 / self.k as f32
    }
}
