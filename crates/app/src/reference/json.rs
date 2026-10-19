//! A minimal JSON value, writer and parser for the reference helper's request and answer (the
//! workspace has no serde; the helper's JSON is small and written by `json.dumps`). Numbers are f64;
//! strings handle the escapes `json.dumps` writes (`\uXXXX` included, surrogate pairs joined).
use std::fmt::Write;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `None` for a missing key and for `null`.
    pub fn field(&self, key: &str) -> Option<&Json> {
        self.get(key).filter(|v| **v != Json::Null)
    }

    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(x) => Some(*x),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// `[x, y, z]`.
    pub fn vec3(&self) -> Option<[f64; 3]> {
        match self.arr()? {
            [x, y, z] => Some([x.num()?, y.num()?, z.num()?]),
            _ => None,
        }
    }

    /// `[[x, y, z], …]`.
    pub fn vec3s(&self) -> Option<Vec<[f64; 3]>> {
        self.arr()?.iter().map(Json::vec3).collect()
    }

    /// Compact JSON text. Non-finite numbers are written as `null` (JSON has none).
    pub fn write(&self) -> String {
        let mut s = String::new();
        self.write_into(&mut s);
        s
    }

    fn write_into(&self, s: &mut String) {
        match self {
            Json::Null => s.push_str("null"),
            Json::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            // `{:?}` of an f64 round-trips exactly (shortest representation)
            Json::Num(x) if x.is_finite() => write!(s, "{x:?}").unwrap(),
            Json::Num(_) => s.push_str("null"),
            Json::Str(t) => quote(t, s),
            Json::Arr(a) => {
                s.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    v.write_into(s);
                }
                s.push(']');
            }
            Json::Obj(kv) => {
                s.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    quote(k, s);
                    s.push(':');
                    v.write_into(s);
                }
                s.push('}');
            }
        }
    }

    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("JSON: trailing text at byte {}", p.i));
        }
        Ok(v)
    }
}

fn quote(t: &str, s: &mut String) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(s, "\\u{:04x}", c as u32).unwrap(),
            c => s.push(c),
        }
    }
    s.push('"');
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("JSON: {what} at byte {}", self.i))
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            self.err("unexpected word")
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'n') => self.lit("null", Json::Null),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            // json.dumps writes NaN / Infinity for non-finite floats (allow_nan defaults to true)
            Some(b'N') => self.lit("NaN", Json::Num(f64::NAN)),
            Some(b'I') => self.lit("Infinity", Json::Num(f64::INFINITY)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return self.err("expected , or ]"),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return self.err("expected a key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return self.err("expected :");
                    }
                    self.i += 1;
                    kv.push((k, self.value()?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(kv));
                        }
                        _ => return self.err("expected , or }"),
                    }
                }
            }
            Some(b'-') if self.b[self.i..].starts_with(b"-Infinity") => {
                self.i += 9;
                Ok(Json::Num(f64::NEG_INFINITY))
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.b[start..self.i]).unwrap();
                t.parse().map(Json::Num).map_err(|_| format!("JSON: bad number '{t}' at byte {start}"))
            }
            Some(_) => self.err("unexpected character"),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u32::from_str_radix(h, 16).ok());
        match h {
            Some(v) => {
                self.i += 4;
                Ok(v)
            }
            None => self.err("bad \\u escape"),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "JSON: invalid UTF-8".to_string())?);
            match self.b.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => {
                    self.i += 1;
                    let Some(&e) = self.b.get(self.i) else { return self.err("unterminated escape") };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) && self.b[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return self.err("bad escape"),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        let v = Json::Obj(vec![
            ("z".into(), Json::Arr(vec![Json::Num(8.0), Json::Num(1.0)])),
            ("x".into(), Json::Num(-1.234567890123456e-7)),
            ("s".into(), Json::Str("a \"q\" \\ \n é ⁺".into())),
            ("n".into(), Json::Null),
            ("b".into(), Json::Bool(false)),
            ("e".into(), Json::Obj(vec![])),
            ("a".into(), Json::Arr(vec![])),
        ]);
        assert_eq!(Json::parse(&v.write()).unwrap(), v);
    }

    #[test]
    fn python_json_dumps_text_parses() {
        // what json.dumps writes: spaces after separators, \uXXXX for non-ASCII, a surrogate pair
        let t = r#"{"ok": true, "energy": -1.1372838344, "forces": [[0.0, 0.0, -0.0123], [0.0, 0.0, 1.23e-02]], "notes": ["café 😀"], "dipole": null, "w": NaN}"#;
        let v = Json::parse(t).unwrap();
        assert_eq!(v.get("ok").and_then(Json::bool), Some(true));
        assert_eq!(v.get("energy").and_then(Json::num), Some(-1.1372838344));
        assert_eq!(v.get("forces").and_then(Json::vec3s), Some(vec![[0.0, 0.0, -0.0123], [0.0, 0.0, 0.0123]]));
        assert_eq!(v.get("notes").and_then(Json::arr).unwrap()[0].str(), Some("café 😀"));
        assert!(v.field("dipole").is_none() && v.get("dipole").is_some());
        assert!(v.get("w").and_then(Json::num).unwrap().is_nan());
    }

    #[test]
    fn bad_text_is_an_error() {
        for t in ["", "{", "[1,]", "{\"a\" 1}", "tru", "\"open", "1 2"] {
            assert!(Json::parse(t).is_err(), "{t}");
        }
    }
}
