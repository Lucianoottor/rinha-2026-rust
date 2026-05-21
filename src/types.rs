use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Config {
    pub max_amount: f64,
    pub max_installments: u32,
    pub amount_vs_avg_ratio: f64,
    pub max_minutes: i64,
    pub max_km: f64,
    pub max_tx_count_24h: u32,
    pub max_merchant_avg_amount: f64,
}

#[derive(Deserialize, Debug)]
pub struct Data {
    pub id: String,
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Deserialize, Debug)]
pub struct Transaction {
    pub amount: f64,
    pub installments: u32,
    pub requested_at: DateTime<Utc>,
}

#[derive(Deserialize, Debug)]
pub struct Customer {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f64,
}

#[derive(Deserialize, Debug)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Deserialize, Debug)]
pub struct LastTransaction {
    pub timestamp: DateTime<Utc>,
    pub km_from_current: f64,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct RawData {
    pub vector: Vec<f32>,
    pub label: String
}