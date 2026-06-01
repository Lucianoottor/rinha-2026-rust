#![allow(unsafe_op_in_unsafe_fn)]
use std::slice;

pub struct Payload<'a> {
    pub amount: f32,
    pub installments: u32,
    pub requested_at: &'a [u8],
    pub customer_avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: &'a [u8],
    pub merchant_id: &'a [u8],
    pub merchant_mcc: &'a [u8],
    pub merchant_avg_amount: f32,
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
    pub last_timestamp: Option<&'a [u8]>,
    pub last_km: Option<f32>,
}

const K_ID: u64          = u64::from_le_bytes([b'i', b'd',  0,   0,   0,   0,   0,   0  ]);
const K_TRANSACTION: u64 = u64::from_le_bytes([b't', b'r', b'a', b'n', b's', b'a', b'c', b't']);
const K_CUSTOMER: u64    = u64::from_le_bytes([b'c', b'u', b's', b't', b'o', b'm', b'e', b'r']);
const K_MERCHANT: u64    = u64::from_le_bytes([b'm', b'e', b'r', b'c', b'h', b'a', b'n', b't']);
const K_TERMINAL: u64    = u64::from_le_bytes([b't', b'e', b'r', b'm', b'i', b'n', b'a', b'l']);
const K_LAST_TX: u64     = u64::from_le_bytes([b'l', b'a', b's', b't', b'_', b't', b'r', b'a']);
const K_AMOUNT: u64      = u64::from_le_bytes([b'a', b'm', b'o', b'u', b'n', b't',  0,   0  ]);
const K_INSTALLM: u64    = u64::from_le_bytes([b'i', b'n', b's', b't', b'a', b'l', b'l', b'm']);
const K_REQUESTED: u64   = u64::from_le_bytes([b'r', b'e', b'q', b'u', b'e', b's', b't', b'e']);
const K_AVG_AMOUNT: u64  = u64::from_le_bytes([b'a', b'v', b'g', b'_', b'a', b'm', b'o', b'u']);
const K_TX_COUNT: u64    = u64::from_le_bytes([b't', b'x', b'_', b'c', b'o', b'u', b'n', b't']);
const K_KNOWN_MERCH: u64 = u64::from_le_bytes([b'k', b'n', b'o', b'w', b'n', b'_', b'm', b'e']);
const K_MCC: u64         = u64::from_le_bytes([b'm', b'c', b'c',  0,   0,   0,   0,   0  ]);
const K_IS_ONLINE: u64   = u64::from_le_bytes([b'i', b's', b'_', b'o', b'n', b'l', b'i', b'n']);
const K_CARD_PRES: u64   = u64::from_le_bytes([b'c', b'a', b'r', b'd', b'_', b'p', b'r', b'e']);
const K_KM_FROM: u64     = u64::from_le_bytes([b'k', b'm', b'_', b'f', b'r', b'o', b'm', b'_']);
const K_TIMESTAMP: u64   = u64::from_le_bytes([b't', b'i', b'm', b'e', b's', b't', b'a', b'm']);

type Ptr = *const u8;

#[inline(always)]
unsafe fn skip_ws(mut p: Ptr, end: Ptr) -> Ptr {
    while p < end && *p <= b' ' {
        p = p.add(1);
    }
    p
}

/// For len >= 8 uses a single unaligned load; shorter keys use a zeroed buffer.
#[inline(always)]
unsafe fn key_hash(p: Ptr, len: usize) -> u64 {
    if len >= 8 {
        (p as *const u64).read_unaligned()
    } else {
        let mut arr = [0u8; 8];
        std::ptr::copy_nonoverlapping(p, arr.as_mut_ptr(), len);
        u64::from_le_bytes(arr)
    }
}

/// p -> opening `"`.  Returns (hash, key_len, ptr_after_closing_`"`).
#[inline(always)]
unsafe fn scan_key(p: Ptr) -> (u64, usize, Ptr) {
    let start = p.add(1);
    let mut cur = start;
    while *cur != b'"' {
        cur = cur.add(1);
    }
    let len = cur.offset_from(start) as usize;
    (key_hash(start, len), len, cur.add(1))
}

/// p -> opening `"`.  Returns (content_slice, ptr_after_closing_`"`).
#[inline(always)]
unsafe fn scan_str<'a>(p: Ptr) -> (&'a [u8], Ptr) {
    let start = p.add(1);
    let mut cur = start;
    loop {
        let b = *cur;
        if b == b'"' { break; }
        if b == b'\\' { cur = cur.add(1); }
        cur = cur.add(1);
    }
    (slice::from_raw_parts(start, cur.offset_from(start) as usize), cur.add(1))
}

/// p -> first digit.  Returns (value, ptr_after).
#[inline(always)]
unsafe fn parse_u32(mut p: Ptr) -> (u32, Ptr) {
    let mut n = 0u32;
    loop {
        let d = (*p).wrapping_sub(b'0');
        if d > 9 { break; }
        n = n.wrapping_mul(10).wrapping_add(d as u32);
        p = p.add(1);
    }
    (n, p)
}

/// p -> start of float literal.  Returns (value, ptr_after).
#[inline(always)]
unsafe fn parse_f32(p: Ptr, end: Ptr) -> (f32, Ptr) {
    let remaining = slice::from_raw_parts(p, end.offset_from(p) as usize);
    match fast_float::parse_partial::<f32, _>(remaining) {
        Ok((v, n)) => (v, p.add(n)),
        Err(_)     => (0.0, p),
    }
}

/// p -> `t` or `f`.  Returns (value, ptr_after).
#[inline(always)]
unsafe fn parse_bool(p: Ptr) -> (bool, Ptr) {
    if *p == b't' { (true, p.add(4)) } else { (false, p.add(5)) }
}

/// p -> opening `[`.  Returns (slice_including_brackets, ptr_after).
#[inline(always)]
unsafe fn slice_array<'a>(p: Ptr, end: Ptr) -> (&'a [u8], Ptr) {
    let start = p;
    let mut cur = p;
    let mut depth = 0i32;
    while cur < end {
        let b = *cur;
        cur = cur.add(1);
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 { break; }
            }
            b'"' => loop {
                let c = *cur;
                cur = cur.add(1);
                if c == b'"' { break; }
                if c == b'\\' { cur = cur.add(1); }
            },
            _ => {}
        }
    }
    (slice::from_raw_parts(start, cur.offset_from(start) as usize), cur)
}

/// p -> start of any JSON value.  Returns ptr_after without allocating.
#[inline(always)]
unsafe fn skip_value(p: Ptr, end: Ptr) -> Ptr {
    match *p {
        b'"' => {
            let mut cur = p.add(1);
            loop {
                let b = *cur;
                cur = cur.add(1);
                if b == b'"' { return cur; }
                if b == b'\\' { cur = cur.add(1); }
            }
        }
        b'{' => {
            let mut cur = p.add(1);
            let mut depth = 1i32;
            while cur < end {
                let b = *cur;
                cur = cur.add(1);
                match b {
                    b'{' => depth += 1,
                    b'}' => { depth -= 1; if depth == 0 { return cur; } }
                    b'"' => loop {
                        let c = *cur;
                        cur = cur.add(1);
                        if c == b'"' { break; }
                        if c == b'\\' { cur = cur.add(1); }
                    },
                    _ => {}
                }
            }
            cur
        }
        b'[' => {
            let (_, after) = slice_array(p, end);
            after
        }
        b't' => p.add(4),
        b'f' => p.add(5),
        b'n' => p.add(4),
        _ => {
            let mut cur = p;
            while cur < end {
                let b = *cur;
                if b == b',' || b == b'}' || b == b']' || b <= b' ' { break; }
                cur = cur.add(1);
            }
            cur
        }
    }
}
#[inline(always)]
unsafe fn parse_transaction<'a>(
    mut p: Ptr, end: Ptr,
) -> Option<(f32, u32, &'a [u8], Ptr)> {
    p = p.add(1);
    let mut amount       = None::<f32>;
    let mut installments = None::<u32>;
    let mut requested_at = None::<&[u8]>;
    loop {
        p = skip_ws(p, end);
        match *p {
            b'}' => { p = p.add(1); break; }
            b',' => { p = p.add(1); continue; }
            _    => {}
        }
        let (h, klen, ak) = scan_key(p);
        p = skip_ws(ak, end);
        p = p.add(1);
        p = skip_ws(p, end);
        match (h, klen) {
            (K_AMOUNT,    6)  => { let (v, q) = parse_f32(p, end); amount       = Some(v); p = q; }
            (K_INSTALLM,  12) => { let (v, q) = parse_u32(p);      installments = Some(v); p = q; }
            (K_REQUESTED, 12) => { let (v, q) = scan_str(p);       requested_at = Some(v); p = q; }
            _                 => { p = skip_value(p, end); }
        }
    }
    Some((amount?, installments?, requested_at?, p))
}

#[inline(always)]
unsafe fn parse_customer<'a>(
    mut p: Ptr, end: Ptr,
) -> Option<(f32, u32, &'a [u8], Ptr)> {
    p = p.add(1);
    let mut avg_amount      = None::<f32>;
    let mut tx_count        = None::<u32>;
    let mut known_merchants = None::<&[u8]>;
    loop {
        p = skip_ws(p, end);
        match *p {
            b'}' => { p = p.add(1); break; }
            b',' => { p = p.add(1); continue; }
            _    => {}
        }
        let (h, klen, ak) = scan_key(p);
        p = skip_ws(ak, end);
        p = p.add(1);
        p = skip_ws(p, end);
        match (h, klen) {
            (K_AVG_AMOUNT,  10) => { let (v, q) = parse_f32(p, end);   avg_amount      = Some(v); p = q; }
            (K_TX_COUNT,    12) => { let (v, q) = parse_u32(p);        tx_count        = Some(v); p = q; }
            (K_KNOWN_MERCH, 15) => { let (v, q) = slice_array(p, end); known_merchants = Some(v); p = q; }
            _                   => { p = skip_value(p, end); }
        }
    }
    Some((avg_amount?, tx_count?, known_merchants?, p))
}

#[inline(always)]
unsafe fn parse_merchant<'a>(
    mut p: Ptr, end: Ptr,
) -> Option<(&'a [u8], &'a [u8], f32, Ptr)> {
    p = p.add(1);
    let mut id         = None::<&[u8]>;
    let mut mcc        = None::<&[u8]>;
    let mut avg_amount = None::<f32>;
    loop {
        p = skip_ws(p, end);
        match *p {
            b'}' => { p = p.add(1); break; }
            b',' => { p = p.add(1); continue; }
            _    => {}
        }
        let (h, klen, ak) = scan_key(p);
        p = skip_ws(ak, end);
        p = p.add(1);
        p = skip_ws(p, end);
        match (h, klen) {
            (K_ID,         2)  => { let (v, q) = scan_str(p);       id         = Some(v); p = q; }
            (K_MCC,        3)  => { let (v, q) = scan_str(p);       mcc        = Some(v); p = q; }
            (K_AVG_AMOUNT, 10) => { let (v, q) = parse_f32(p, end); avg_amount = Some(v); p = q; }
            _                  => { p = skip_value(p, end); }
        }
    }
    Some((id?, mcc?, avg_amount?, p))
}

#[inline(always)]
unsafe fn parse_terminal(mut p: Ptr, end: Ptr) -> Option<(bool, bool, f32, Ptr)> {
    p = p.add(1);
    let mut is_online    = None::<bool>;
    let mut card_present = None::<bool>;
    let mut km_from_home = None::<f32>;
    loop {
        p = skip_ws(p, end);
        match *p {
            b'}' => { p = p.add(1); break; }
            b',' => { p = p.add(1); continue; }
            _    => {}
        }
        let (h, klen, ak) = scan_key(p);
        p = skip_ws(ak, end);
        p = p.add(1);
        p = skip_ws(p, end);
        match (h, klen) {
            (K_IS_ONLINE, 9)  => { let (v, q) = parse_bool(p);     is_online    = Some(v); p = q; }
            (K_CARD_PRES, 12) => { let (v, q) = parse_bool(p);     card_present = Some(v); p = q; }
            (K_KM_FROM,   12) => { let (v, q) = parse_f32(p, end); km_from_home = Some(v); p = q; }
            _                 => { p = skip_value(p, end); }
        }
    }
    Some((is_online?, card_present?, km_from_home?, p))
}

#[inline(always)]
unsafe fn parse_last_tx<'a>(mut p: Ptr, end: Ptr) -> Option<(&'a [u8], f32, Ptr)> {
    p = p.add(1);
    let mut timestamp = None::<&[u8]>;
    let mut km        = None::<f32>;
    loop {
        p = skip_ws(p, end);
        match *p {
            b'}' => { p = p.add(1); break; }
            b',' => { p = p.add(1); continue; }
            _    => {}
        }
        let (h, klen, ak) = scan_key(p);
        p = skip_ws(ak, end);
        p = p.add(1);
        p = skip_ws(p, end);
        match (h, klen) {
            (K_TIMESTAMP, 9)  => { let (v, q) = scan_str(p);       timestamp = Some(v); p = q; }
            (K_KM_FROM,   15) => { let (v, q) = parse_f32(p, end); km        = Some(v); p = q; }
            _                 => { p = skip_value(p, end); }
        }
    }
    Some((timestamp?, km?, p))
}

pub fn parse_payload(req_body: &[u8]) -> Option<Payload<'_>> {
    if req_body.len() < 2 { return None; }

    unsafe {
        let base = req_body.as_ptr();
        let end  = base.add(req_body.len());

        let mut p = skip_ws(base, end);
        if *p != b'{' { return None; }
        p = p.add(1);

        let mut amount          = None::<f32>;
        let mut installments    = None::<u32>;
        let mut requested_at    = None::<&[u8]>;
        let mut customer_avg    = None::<f32>;
        let mut tx_count_24h    = None::<u32>;
        let mut known_merchants = None::<&[u8]>;
        let mut merchant_id     = None::<&[u8]>;
        let mut merchant_mcc    = None::<&[u8]>;
        let mut merchant_avg    = None::<f32>;
        let mut is_online       = None::<bool>;
        let mut card_present    = None::<bool>;
        let mut km_from_home    = None::<f32>;
        let mut last_timestamp  = None::<&[u8]>;
        let mut last_km         = None::<f32>;

        loop {
            p = skip_ws(p, end);
            match *p {
                b'}' => { p = p.add(1); break; }
                b',' => { p = p.add(1); continue; }
                _    => {}
            }
            let (h, klen, ak) = scan_key(p);
            p = skip_ws(ak, end);
            p = p.add(1);
            p = skip_ws(p, end);

            match (h, klen) {
                (K_ID, 2) => {
                    let (_, q) = scan_str(p);
                    p = q;
                }
                (K_TRANSACTION, 11) => {
                    let (a, inst, rat, q) = parse_transaction(p, end)?;
                    amount       = Some(a);
                    installments = Some(inst);
                    requested_at = Some(rat);
                    p = q;
                }
                (K_CUSTOMER, 8) => {
                    let (avg, cnt, km, q) = parse_customer(p, end)?;
                    customer_avg    = Some(avg);
                    tx_count_24h    = Some(cnt);
                    known_merchants = Some(km);
                    p = q;
                }
                (K_MERCHANT, 8) => {
                    let (mid, mcc, mavg, q) = parse_merchant(p, end)?;
                    merchant_id  = Some(mid);
                    merchant_mcc = Some(mcc);
                    merchant_avg = Some(mavg);
                    p = q;
                }
                (K_TERMINAL, 8) => {
                    let (online, card, km, q) = parse_terminal(p, end)?;
                    is_online    = Some(online);
                    card_present = Some(card);
                    km_from_home = Some(km);
                    p = q;
                }
                (K_LAST_TX, 16) => {
                    if *p == b'n' {
                        p = p.add(4);
                    } else {
                        let (ts, km, q) = parse_last_tx(p, end)?;
                        last_timestamp = Some(ts);
                        last_km        = Some(km);
                        p = q;
                    }
                }
                _ => { p = skip_value(p, end); }
            }
        }

        Some(Payload {
            amount:              amount?,
            installments:        installments?,
            requested_at:        requested_at?,
            customer_avg_amount: customer_avg?,
            tx_count_24h:        tx_count_24h?,
            known_merchants:     known_merchants?,
            merchant_id:         merchant_id?,
            merchant_mcc:        merchant_mcc?,
            merchant_avg_amount: merchant_avg?,
            is_online:           is_online?,
            card_present:        card_present?,
            km_from_home:        km_from_home?,
            last_timestamp,
            last_km,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SAMPLE: &[u8] = br#"{"id":"tx-100","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.2331},"last_transaction":null}"#;

    const SAMPLE_PRETTY: &[u8] = br#"{
    "id": "tx-1329056812",
    "transaction": {
      "amount": 41.12,
      "installments": 2,
      "requested_at": "2026-03-11T18:45:53Z"
    },
    "customer": {
      "avg_amount": 82.24,
      "tx_count_24h": 3,
      "known_merchants": [
        "MERC-003",
        "MERC-016"
      ]
    },
    "merchant": {
      "id": "MERC-016",
      "mcc": "5411",
      "avg_amount": 60.25
    },
    "terminal": {
      "is_online": false,
      "card_present": true,
      "km_from_home": 29.2331036248
    },
    "last_transaction": null
  }"#;

    #[test]
    fn test_scanner() {
        let payload = parse_payload(SAMPLE).unwrap();
        assert_eq!(payload.amount, 41.12_f32);
        assert_eq!(payload.installments, 2);
        assert_eq!(payload.requested_at, b"2026-03-11T18:45:53Z" as &[u8]);
        assert_eq!(payload.customer_avg_amount, 82.24_f32);
        assert_eq!(payload.tx_count_24h, 3);
        assert_eq!(payload.merchant_id, b"MERC-016" as &[u8]);
        assert_eq!(payload.merchant_mcc, b"5411" as &[u8]);
        assert_eq!(payload.merchant_avg_amount, 60.25_f32);
        assert!(!payload.is_online);
        assert!(payload.card_present);
        assert_eq!(payload.km_from_home, 29.2331_f32);
        assert_eq!(payload.last_timestamp, None);
        assert_eq!(payload.last_km, None);
    }

    const SAMPLE_WITH_LAST_TX: &[u8] = br#"{
    "id": "tx-3576980410",
    "transaction": {
      "amount": 384.88,
      "installments": 3,
      "requested_at": "2026-03-11T20:23:35Z"
    },
    "customer": {
      "avg_amount": 769.76,
      "tx_count_24h": 3,
      "known_merchants": ["MERC-009","MERC-009","MERC-001","MERC-001"]
    },
    "merchant": {
      "id": "MERC-001",
      "mcc": "5912",
      "avg_amount": 298.95
    },
    "terminal": {
      "is_online": false,
      "card_present": true,
      "km_from_home": 13.7090520965
    },
    "last_transaction": {
      "timestamp": "2026-03-11T14:58:35Z",
      "km_from_current": 18.8626479774
    }
  }"#;

    #[test]
    fn test_scanner_with_last_tx() {
        let payload = parse_payload(SAMPLE_WITH_LAST_TX).unwrap();
        assert_eq!(payload.amount, 384.88_f32);
        assert_eq!(payload.installments, 3);
        assert_eq!(payload.merchant_id, b"MERC-001" as &[u8]);
        assert_eq!(payload.merchant_mcc, b"5912" as &[u8]);
        assert_eq!(payload.last_timestamp, Some(b"2026-03-11T14:58:35Z" as &[u8]));
        assert!(payload.last_km.is_some());
    }

    #[test]
    fn test_scanner_pretty() {
        let payload = parse_payload(SAMPLE_PRETTY).unwrap();
        assert_eq!(payload.amount, 41.12_f32);
        assert_eq!(payload.installments, 2);
        assert_eq!(payload.requested_at, b"2026-03-11T18:45:53Z" as &[u8]);
        assert_eq!(payload.customer_avg_amount, 82.24_f32);
        assert_eq!(payload.tx_count_24h, 3);
        assert_eq!(payload.merchant_id, b"MERC-016" as &[u8]);
        assert_eq!(payload.merchant_mcc, b"5411" as &[u8]);
        assert!(!payload.is_online);
        assert!(payload.card_present);
        assert_eq!(payload.last_timestamp, None);
        assert_eq!(payload.last_km, None);
    }
}