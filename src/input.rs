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


// IA gerou as const de comparacao - fara o match com o inicio da word para aplicar jump e evitar leitura de chars
const K_ID: u64          = u64::from_le_bytes([b'i', b'd',  0,   0,   0,   0,   0,   0  ]);
const K_TRANSACTION: u64 = u64::from_le_bytes([b't', b'r', b'a', b'n', b's', b'a', b'c', b't']);
const K_CUSTOMER: u64    = u64::from_le_bytes([b'c', b'u', b's', b't', b'o', b'm', b'e', b'r']);
const K_MERCHANT: u64    = u64::from_le_bytes([b'm', b'e', b'r', b'c', b'h', b'a', b'n', b't']);
const K_TERMINAL: u64    = u64::from_le_bytes([b't', b'e', b'r', b'm', b'i', b'n', b'a', b'l']);
// "last_transaction" → first 8 = "last_tra"
const K_LAST_TX: u64     = u64::from_le_bytes([b'l', b'a', b's', b't', b'_', b't', b'r', b'a']);

// transaction object
const K_AMOUNT: u64      = u64::from_le_bytes([b'a', b'm', b'o', b'u', b'n', b't',  0,   0  ]);
// "installments" → first 8 = "installm"
const K_INSTALLM: u64    = u64::from_le_bytes([b'i', b'n', b's', b't', b'a', b'l', b'l', b'm']);
// "requested_at" → first 8 = "requeste"
const K_REQUESTED: u64   = u64::from_le_bytes([b'r', b'e', b'q', b'u', b'e', b's', b't', b'e']);

// customer object
// "avg_amount" → first 8 = "avg_amou"
const K_AVG_AMOUNT: u64  = u64::from_le_bytes([b'a', b'v', b'g', b'_', b'a', b'm', b'o', b'u']);
// "tx_count_24h" → first 8 = "tx_count"
const K_TX_COUNT: u64    = u64::from_le_bytes([b't', b'x', b'_', b'c', b'o', b'u', b'n', b't']);
// "known_merchants" → first 8 = "known_me"
const K_KNOWN_MERCH: u64 = u64::from_le_bytes([b'k', b'n', b'o', b'w', b'n', b'_', b'm', b'e']);

// merchant object  (K_ID and K_AVG_AMOUNT reused)
const K_MCC: u64         = u64::from_le_bytes([b'm', b'c', b'c',  0,   0,   0,   0,   0  ]);

// terminal object
// "is_online" → first 8 = "is_onlin"
const K_IS_ONLINE: u64   = u64::from_le_bytes([b'i', b's', b'_', b'o', b'n', b'l', b'i', b'n']);
// "card_present" → first 8 = "card_pre"
const K_CARD_PRES: u64   = u64::from_le_bytes([b'c', b'a', b'r', b'd', b'_', b'p', b'r', b'e']);
// "km_from_home" / "km_from_current" → first 8 = "km_from_"
const K_KM_FROM: u64     = u64::from_le_bytes([b'k', b'm', b'_', b'f', b'r', b'o', b'm', b'_']);

// last_transaction object
// "timestamp" → first 8 = "timestam"
const K_TIMESTAMP: u64   = u64::from_le_bytes([b't', b'i', b'm', b'e', b's', b't', b'a', b'm']);


// https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
// identifica se o char esta presente no conjunto x
#[inline(always)]
fn has_byte(x: u64, needle: u8) -> bool {
    let v = x ^ (0x0101_0101_0101_0101_u64.wrapping_mul(needle as u64));
    v.wrapping_sub(0x0101_0101_0101_0101_u64) & !v & 0x8080_8080_8080_8080_u64 != 0
}


#[inline(always)]
fn key_hash(b: &[u8]) -> u64 {
    let mut arr = [0u8; 8];
    let n = b.len().min(8);
    arr[..n].copy_from_slice(&b[..n]);
    u64::from_le_bytes(arr)
}

struct Scanner<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Scanner<'a> {
    #[inline] fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }
    #[inline] fn peek(&self) -> Option<u8> { self.buf.get(self.pos).copied() }

    #[inline]
    fn skip_ws(&mut self) {
        let buf = self.buf;
        let mut i = self.pos;
        while i < buf.len() {
            let b = buf[i];
            if b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' { break; }
            i += 1;
        }
        self.pos = i;
    }

    #[inline]
    fn expect(&mut self, byte: u8) -> Option<()> {
        if self.buf.get(self.pos).copied() == Some(byte) {
            self.pos += 1;
            Some(())
        } else {
            None
        }
    }

    #[inline]
    fn scan_str_content(&mut self) -> Option<&'a [u8]> {
        let buf = self.buf;
        let start = self.pos;
        let mut i = start;

        while i + 8 <= buf.len() {
            let chunk = unsafe { (buf.as_ptr().add(i) as *const u64).read_unaligned() };
            if has_byte(chunk, b'"') || has_byte(chunk, b'\\') {
                break;
            }
            i += 8;
        }

        loop {
            let b = *buf.get(i)?;
            if b == b'"' {
                self.pos = i + 1;
                return Some(&buf[start..i]);
            }
            if b == b'\\' { i += 2; } else { i += 1; }
        }
    }

    #[inline]
    fn read_str(&mut self) -> Option<&'a [u8]> {
        self.expect(b'"')?;
        self.scan_str_content()
    }
    #[inline]
    fn read_key(&mut self) -> Option<(u64, usize)> {
        self.expect(b'"')?;
        let content = self.scan_str_content()?;
        Some((key_hash(content), content.len()))
    }

    #[inline]
    fn parse_u32(&mut self) -> Option<u32> {
        let mut n = 0u32;
        let mut hit = false;
        while let Some(b @ b'0'..=b'9') = self.peek() {
            n = n.wrapping_mul(10).wrapping_add((b - b'0') as u32);
            self.pos += 1;
            hit = true;
        }
        hit.then_some(n)
    }

    #[inline]
    fn parse_f32(&mut self) -> Option<f32> {
        let (val, consumed) = fast_float::parse_partial::<f32, _>(&self.buf[self.pos..]).ok()?;
        self.pos += consumed;
        Some(val)
    }

    #[inline]
    fn parse_bool(&mut self) -> Option<bool> {
        match self.peek()? {
            b't' => { self.pos += 4; Some(true)  }
            b'f' => { self.pos += 5; Some(false) }
            _    => None,
        }
    }

    #[inline]
    fn try_null(&mut self) -> bool {
        if self.peek() == Some(b'n') { self.pos += 4; true } else { false }
    }

    fn slice_array(&mut self) -> Option<&'a [u8]> {
        let start = self.pos;
        self.skip_array()?;
        Some(&self.buf[start..self.pos])
    }

    fn skip_array(&mut self) -> Option<()> {
        self.expect(b'[')?;
        self.skip_ws();
        if self.peek() == Some(b']') { self.pos += 1; return Some(()); }
        loop {
            self.skip_ws();
            self.skip_value()?;
            self.skip_ws();
            match self.peek()? {
                b']' => { self.pos += 1; return Some(()); }
                b',' => { self.pos += 1; }
                _    => return None,
            }
        }
    }

    fn skip_object(&mut self) -> Option<()> {
        self.expect(b'{')?;
        self.skip_ws();
        if self.peek() == Some(b'}') { self.pos += 1; return Some(()); }
        loop {
            self.skip_ws();
            self.expect(b'"')?;
            self.scan_str_content()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            self.skip_value()?;
            self.skip_ws();
            match self.peek()? {
                b'}' => { self.pos += 1; return Some(()); }
                b',' => { self.pos += 1; }
                _    => return None,
            }
        }
    }

    fn skip_value(&mut self) -> Option<()> {
        self.skip_ws();
        match self.peek()? {
            b'"' => { self.pos += 1; self.scan_str_content()?; }
            b'{' => { self.skip_object()?; }
            b'[' => { self.skip_array()?; }
            b't' => { self.pos += 4; } // true
            b'f' => { self.pos += 5; } // false
            b'n' => { self.pos += 4; } // null
            _    => {
                // number
                while matches!(self.peek(), Some(b'-' | b'+' | b'0'..=b'9' | b'.' | b'e' | b'E')) {
                    self.pos += 1;
                }
            }
        }
        Some(())
    }

 
    fn parse_transaction(&mut self) -> Option<(f32, u32, &'a [u8])> {
        self.expect(b'{')?;
        let mut amount = None::<f32>;
        let mut installments = None::<u32>;
        let mut requested_at = None::<&[u8]>;
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => { self.pos += 1; break; }
                b',' => { self.pos += 1; continue; }
                b'"' => {}
                _    => return None,
            }
            let (h, klen) = self.read_key()?;
            self.skip_ws(); self.expect(b':')?; self.skip_ws();

            // compara com as const no inicio do codigo - se encontrar, pula os chars que ja sabemos quais sao e nao precisam ser lidos
            match (h, klen) {
                (K_AMOUNT,     6)  => amount       = Some(self.parse_f32()?),
                (K_INSTALLM,   12) => installments = Some(self.parse_u32()?),
                (K_REQUESTED,  12) => requested_at = Some(self.read_str()?),
                _                  => { self.skip_value()?; }
            }
        }
        Some((amount?, installments?, requested_at?))
    }

    fn parse_customer(&mut self) -> Option<(f32, u32, &'a [u8])> {
        self.expect(b'{')?;
        let mut avg_amount = None::<f32>;
        let mut tx_count = None::<u32>;
        let mut known_merchants = None::<&[u8]>;
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => { self.pos += 1; break; }
                b',' => { self.pos += 1; continue; }
                b'"' => {}
                _    => return None,
            }
            let (h, klen) = self.read_key()?;
            self.skip_ws(); self.expect(b':')?; self.skip_ws();
            match (h, klen) {
                (K_AVG_AMOUNT,  10) => avg_amount      = Some(self.parse_f32()?),
                (K_TX_COUNT,    12) => tx_count         = Some(self.parse_u32()?),
                (K_KNOWN_MERCH, 15) => known_merchants  = Some(self.slice_array()?),
                _                   => { self.skip_value()?; }
            }
        }
        Some((avg_amount?, tx_count?, known_merchants?))
    }

    fn parse_merchant(&mut self) -> Option<(&'a [u8], &'a [u8], f32)> {
        self.expect(b'{')?;
        let mut id = None::<&[u8]>;
        let mut mcc = None::<&[u8]>;
        let mut avg_amount = None::<f32>;
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => { self.pos += 1; break; }
                b',' => { self.pos += 1; continue; }
                b'"' => {}
                _    => return None,
            }
            let (h, klen) = self.read_key()?;
            self.skip_ws(); self.expect(b':')?; self.skip_ws();
            match (h, klen) {
                (K_ID,         2)  => id         = Some(self.read_str()?),
                (K_MCC,        3)  => mcc        = Some(self.read_str()?),
                (K_AVG_AMOUNT, 10) => avg_amount = Some(self.parse_f32()?),
                _                  => { self.skip_value()?; }
            }
        }
        Some((id?, mcc?, avg_amount?))
    }

    fn parse_terminal(&mut self) -> Option<(bool, bool, f32)> {
        self.expect(b'{')?;
        let mut is_online = None::<bool>;
        let mut card_present = None::<bool>;
        let mut km_from_home = None::<f32>;
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => { self.pos += 1; break; }
                b',' => { self.pos += 1; continue; }
                b'"' => {}
                _    => return None,
            }
            let (h, klen) = self.read_key()?;
            self.skip_ws(); self.expect(b':')?; self.skip_ws();
            match (h, klen) {
                (K_IS_ONLINE,  9)  => is_online    = Some(self.parse_bool()?),
                (K_CARD_PRES,  12) => card_present = Some(self.parse_bool()?),
                (K_KM_FROM,    12) => km_from_home = Some(self.parse_f32()?),
                _                  => { self.skip_value()?; }
            }
        }
        Some((is_online?, card_present?, km_from_home?))
    }

    fn parse_last_tx(&mut self) -> Option<(&'a [u8], f32)> {
        self.expect(b'{')?;
        let mut timestamp = None::<&[u8]>;
        let mut km = None::<f32>;
        loop {
            self.skip_ws();
            match self.peek()? {
                b'}' => { self.pos += 1; break; }
                b',' => { self.pos += 1; continue; }
                b'"' => {}
                _    => return None,
            }
            let (h, klen) = self.read_key()?;
            self.skip_ws(); self.expect(b':')?; self.skip_ws();
            match (h, klen) {
                (K_TIMESTAMP, 9)  => timestamp = Some(self.read_str()?),
                (K_KM_FROM,   15) => km        = Some(self.parse_f32()?),
                _                 => { self.skip_value()?; }
            }
        }
        Some((timestamp?, km?))
    }
}


pub fn parse_payload(req_body: &[u8]) -> Option<Payload<'_>> {
    let mut s = Scanner::new(req_body);
    s.skip_ws();
    s.expect(b'{')?;

    let mut amount           = None::<f32>;
    let mut installments     = None::<u32>;
    let mut requested_at     = None::<&[u8]>;
    let mut customer_avg     = None::<f32>;
    let mut tx_count_24h     = None::<u32>;
    let mut known_merchants  = None::<&[u8]>;
    let mut merchant_id      = None::<&[u8]>;
    let mut merchant_mcc     = None::<&[u8]>;
    let mut merchant_avg     = None::<f32>;
    let mut is_online        = None::<bool>;
    let mut card_present     = None::<bool>;
    let mut km_from_home     = None::<f32>;
    let mut last_timestamp   = None::<&[u8]>;
    let mut last_km          = None::<f32>;

    loop {
        s.skip_ws();
        match s.peek()? {
            b'}' => { s.pos += 1; break; }
            b',' => { s.pos += 1; continue; }
            b'"' => {}
            _    => return None,
        }
        let (h, klen) = s.read_key()?;
        s.skip_ws(); s.expect(b':')?; s.skip_ws();

        match (h, klen) {
            (K_ID,          2)  => { s.read_str()?; }
            (K_TRANSACTION, 11) => {
                let (a, inst, rat) = s.parse_transaction()?;
                amount       = Some(a);
                installments = Some(inst);
                requested_at = Some(rat);
            }
            (K_CUSTOMER, 8) => {
                let (avg, cnt, km) = s.parse_customer()?;
                customer_avg    = Some(avg);
                tx_count_24h    = Some(cnt);
                known_merchants = Some(km);
            }
            (K_MERCHANT, 8) => {
                let (mid, mcc, mavg) = s.parse_merchant()?;
                merchant_id  = Some(mid);
                merchant_mcc = Some(mcc);
                merchant_avg = Some(mavg);
            }
            (K_TERMINAL, 8) => {
                let (online, card, km) = s.parse_terminal()?;
                is_online    = Some(online);
                card_present = Some(card);
                km_from_home = Some(km);
            }
            (K_LAST_TX, 16) => {
                if !s.try_null() {
                    let (ts, km) = s.parse_last_tx()?;
                    last_timestamp = Some(ts);
                    last_km        = Some(km);
                }
            }
            _ => { s.skip_value()?; }
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
