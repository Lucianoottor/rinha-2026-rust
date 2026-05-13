use chrono::{Datelike, Timelike};
use crate::traits::Normalizer;
use crate::types::*;

#[inline(always)]
fn comp_mcc(mcc: &str) -> f32 {
    match mcc {
    "5411" => 0.15,
    "5812" => 0.30,
    "5912" => 0.20,
    "5944" => 0.45,
    "7801" => 0.80,
    "7802" => 0.75,
    "7995" => 0.85,
    "4511" => 0.35,
    "5311" => 0.25,
    "5999" => 0.50,
    _ => 0.50
    }
}

pub struct DataNormalizer;
impl DataNormalizer {
    pub fn new() -> Self {
        DataNormalizer
    }
}

impl Normalizer for DataNormalizer {
    fn normalize(&self, data: &Data) -> Vec<f32> {
        let max_amount = 10000.0;
        let max_installments = 12.0;
        let amount_vs_avg_ratio = 10.0;
        let max_minutes = 1440.0;
        let max_km = 1000.0;
        let max_tx_count_24h = 20.0;
        let max_merchant_avg_amount = 10000.0;

        let r4 = |val: f64| ( (val * 10000.0).round() / 10000.0 ) as f32;

        let limit = |val: f64, max: f64| -> f32 {
            if max <= 0.0 { return 0.0; }
            r4((val / max).min(1.0).max(0.0))
        };

        let mut v = Vec::with_capacity(14);

        // 0: amount 
        v.push(limit(data.transaction.amount, max_amount));

        // 1: installments
        v.push(limit(data.transaction.installments as f64, max_installments));

        // 2: amount_vs_avg 
        let ratio = if data.customer.avg_amount > 0.0 {
            data.transaction.amount / data.customer.avg_amount
        } else {
            0.0
        };
        v.push(limit(ratio, amount_vs_avg_ratio));

        // 3: hour_of_day - (0-23 / 23)
        v.push(r4(data.transaction.requested_at.hour() as f64 / 23.0));

        // 4: day_of_week - (Mon=0 to Sun=6 / 6)
        v.push(r4(data.transaction.requested_at.weekday().num_days_from_monday() as f64 / 6.0));

        // 5 & 6: Last Transaction logic (minutes and km)
        if let Some(last) = &data.last_transaction {
            let duration = data.transaction.requested_at.signed_duration_since(last.timestamp);
            v.push(limit(duration.num_minutes() as f64, max_minutes)); // Index 5
            v.push(limit(last.km_from_current, max_km));               // Index 6
        } else {
            v.push(-1.0);
            v.push(-1.0);
        }

        // 7: km_from_home 
        v.push(limit(data.terminal.km_from_home, max_km));

        // 8: tx_count_24h 
        v.push(limit(data.customer.tx_count_24h as f64, max_tx_count_24h));

        // 9: is_online
        v.push(if data.terminal.is_online { 1.0 } else { 0.0 });

        // 10: card_present
        v.push(if data.terminal.card_present { 1.0 } else { 0.0 });

        // 11: unknown_merchant
        let is_known = data.customer.known_merchants.contains(&data.merchant.id);
        v.push(if is_known { 0.0 } else { 1.0 });

        // 12: mcc_risk
        let risk = comp_mcc(data.merchant.mcc.as_str());
        v.push(risk);

        // 13: merchant_avg_amount - (merchant.avg_amount / max_merchant_avg_amount)
        v.push(limit(data.merchant.avg_amount, max_merchant_avg_amount));

        v
    }
}