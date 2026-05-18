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

#[inline(always)]
fn clamp(val: f64, max: f64) -> f32 {
    (val / max).min(1.0).max(0.0) as f32
}

pub struct DataNormalizer;

impl Normalizer for DataNormalizer {
    fn normalize(&self, data: &Data) -> Vec<f32> {
        let mut v = Vec::with_capacity(14);

        // 0: amount
        v.push(clamp(data.transaction.amount, 10000.0));

        // 1: installments
        v.push(clamp(data.transaction.installments as f64, 12.0));

        // 2: amount_vs_avg
        let ratio = if data.customer.avg_amount > 0.0 {
            data.transaction.amount / data.customer.avg_amount
        } else {
            0.0
        };
        v.push(clamp(ratio, 10.0));

        // 3: hour_of_day
        v.push((data.transaction.requested_at.hour() as f32) / 23.0);

        // 4: day_of_week
        v.push((data.transaction.requested_at.weekday().num_days_from_monday() as f32) / 6.0);

        // 5 & 6: Last Transaction logic (minutes and km)
        if let Some(last) = &data.last_transaction {
            let duration = data.transaction.requested_at.signed_duration_since(last.timestamp);
            v.push(clamp(duration.num_minutes() as f64, 1440.0));
            v.push(clamp(last.km_from_current, 1000.0));
        } else {
            v.push(-1.0);
            v.push(-1.0);
        }

        // 7: km_from_home
        v.push(clamp(data.terminal.km_from_home, 1000.0));

        // 8: tx_count_24h
        v.push(clamp(data.customer.tx_count_24h as f64, 20.0));

        // 9: is_online
        v.push(if data.terminal.is_online { 1.0 } else { 0.0 });

        // 10: card_present
        v.push(if data.terminal.card_present { 1.0 } else { 0.0 });

        // 11: unknown_merchant
        let is_known = data.customer.known_merchants.contains(&data.merchant.id);
        v.push(if is_known { 0.0 } else { 1.0 });

        // 12: mcc_risk
        v.push(comp_mcc(data.merchant.mcc.as_str()));

        // 13: merchant_avg_amount
        v.push(clamp(data.merchant.avg_amount, 10000.0));

        v
    }
}
