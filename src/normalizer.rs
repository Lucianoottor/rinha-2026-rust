use crate::input::Payload;

pub struct DataNormalizer;

impl DataNormalizer {
    pub fn normalize(&self, data: &Payload<'_>) -> [f32; 16] {
        let mut v = [0.0f32; 16];

        v[0] = clamp(data.amount, 10000.0);
        v[1] = clamp(data.installments as f32, 12.0);

        let ratio = if data.customer_avg_amount > 0.0 {
            data.amount / data.customer_avg_amount
        } else {
            0.0
        };
        v[2] = clamp(ratio, 10.0);

        v[3] = ts_hour(data.requested_at);
        v[4] = ts_weekday(data.requested_at);

        match (data.last_timestamp, data.last_km) {
            (Some(ts), Some(km)) => {
                let minutes = ts_to_minutes(data.requested_at) - ts_to_minutes(ts);
                v[5] = clamp(minutes as f32, 1440.0);
                v[6] = clamp(km, 1000.0);
            }
            _ => {
                v[5] = -1.0;
                v[6] = -1.0;
            }
        }

        v[7] = clamp(data.km_from_home, 1000.0);
        v[8] = clamp(data.tx_count_24h as f32, 20.0);
        v[9] = if data.is_online { 1.0 } else { 0.0 };
        v[10] = if data.card_present { 1.0 } else { 0.0 };

        let id = data.merchant_id;
        let is_known = data.known_merchants
            .windows(id.len() + 2)
            .any(|w| w[0] == b'"' && &w[1..w.len() - 1] == id && w[w.len() - 1] == b'"');
        v[11] = if is_known { 0.0 } else { 1.0 };

        v[12] = comp_mcc(data.merchant_mcc);
        v[13] = clamp(data.merchant_avg_amount, 10000.0);

        v
    }
}

#[inline(always)]
fn comp_mcc(mcc: &[u8]) -> f32 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.50,
    }
}

#[inline(always)]
fn clamp(val: f32, max: f32) -> f32 {
    (val / max).min(1.0).max(0.0)
}

#[inline(always)]
fn b2(b: &[u8]) -> u32 {
    (b[0] - b'0') as u32 * 10 + (b[1] - b'0') as u32
}

#[inline(always)]
fn b4(b: &[u8]) -> u32 {
    (b[0] - b'0') as u32 * 1000
        + (b[1] - b'0') as u32 * 100
        + (b[2] - b'0') as u32 * 10
        + (b[3] - b'0') as u32
}

#[inline]
fn ymd_to_days(y: u32, m: u32, d: u32) -> i64 {
    let a = (14 - m) / 12;
    let yr = (y + 4800 - a) as i64;
    let mo = (m + 12 * a - 3) as i64;
    d as i64 + (153 * mo + 2) / 5 + 365 * yr + yr / 4 - yr / 100 + yr / 400 - 32045
}

#[inline]
fn ts_to_minutes(b: &[u8]) -> i64 {
    ymd_to_days(b4(&b[0..4]), b2(&b[5..7]), b2(&b[8..10])) * 1440
        + b2(&b[11..13]) as i64 * 60
        + b2(&b[14..16]) as i64
}

#[inline(always)]
fn ts_hour(b: &[u8]) -> f32 {
    b2(&b[11..13]) as f32 / 23.0
}

#[inline]
fn ts_weekday(b: &[u8]) -> f32 {
    const T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = b4(&b[0..4]);
    let m = b2(&b[5..7]);
    let d = b2(&b[8..10]);
    let yr = if m < 3 { y - 1 } else { y };
    let dow = (yr + yr / 4 - yr / 100 + yr / 400 + T[(m - 1) as usize] + d) % 7;
    ((dow + 6) % 7) as f32 / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ts_hour() {
        assert_eq!(ts_hour(b"2026-03-11T18:45:53Z"), 18.0 / 23.0);
        assert_eq!(ts_hour(b"2026-03-11T00:00:00Z"), 0.0);
        assert_eq!(ts_hour(b"2026-03-11T23:59:59Z"), 1.0);
    }

    #[test]
    fn test_ts_weekday() {
        assert_eq!(ts_weekday(b"2026-03-11T00:00:00Z"), 2.0 / 6.0);
        assert_eq!(ts_weekday(b"2026-03-09T00:00:00Z"), 0.0 / 6.0);
        assert_eq!(ts_weekday(b"2026-03-15T00:00:00Z"), 6.0 / 6.0);
    }

    #[test]
    fn test_ts_minutes_diff() {
        let a = b"2026-03-11T20:23:35Z";
        let b = b"2026-03-11T14:58:35Z";
        assert_eq!(ts_to_minutes(a) - ts_to_minutes(b), 325);
    }
}
