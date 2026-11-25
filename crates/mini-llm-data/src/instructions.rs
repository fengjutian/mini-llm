//! Minimal JSONL instruction data: one `{"prompt": "...", "response": "..."}`
//! object per line.
//!
//! Only the small subset needed for instruction tuning is supported: flat
//! objects with string values. Unknown keys with string values are ignored;
//! other value types are rejected so malformed data fails loudly.

use crate::DataError;

/// One prompt/response pair from a JSONL file.
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionPair {
    pub prompt: String,
    pub response: String,
}

/// Parses every non-empty line of `text` as an instruction object.
pub fn parse_instructions(text: &str) -> Result<Vec<InstructionPair>, DataError> {
    let mut pairs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        pairs.push(parse_line(line, line_number)?);
    }
    if pairs.is_empty() {
        return Err(DataError::EmptyCorpus);
    }
    Ok(pairs)
}

fn parse_line(line: &str, line_number: usize) -> Result<InstructionPair, DataError> {
    let bytes = line.as_bytes();
    let mut position = 0usize;

    skip_whitespace(bytes, &mut position);
    expect(bytes, &mut position, b'{', line_number)?;

    let mut prompt = None;
    let mut response = None;
    loop {
        skip_whitespace(bytes, &mut position);
        if peek(bytes, position) == Some(b'}') {
            position += 1;
            break;
        }
        let key = parse_string(bytes, &mut position, line_number)?;
        skip_whitespace(bytes, &mut position);
        expect(bytes, &mut position, b':', line_number)?;
        skip_whitespace(bytes, &mut position);

        match key.as_str() {
            "prompt" => prompt = Some(parse_string(bytes, &mut position, line_number)?),
            "response" => response = Some(parse_string(bytes, &mut position, line_number)?),
            _ => skip_scalar(bytes, &mut position, line_number)?,
        }

        skip_whitespace(bytes, &mut position);
        match peek(bytes, position) {
            Some(b',') => position += 1,
            Some(b'}') => {
                position += 1;
                break;
            }
            _ => return Err(invalid(line_number, "expected ',' or '}'")),
        }
    }
    skip_whitespace(bytes, &mut position);
    if position != bytes.len() {
        return Err(invalid(line_number, "trailing data after the object"));
    }

    let prompt = prompt.ok_or_else(|| invalid(line_number, "missing `prompt`"))?;
    let response = response.ok_or_else(|| invalid(line_number, "missing `response`"))?;
    if response.is_empty() {
        return Err(invalid(line_number, "`response` must not be empty"));
    }
    Ok(InstructionPair { prompt, response })
}

fn invalid(line: usize, reason: &'static str) -> DataError {
    DataError::InvalidInstruction { line, reason }
}

fn peek(bytes: &[u8], position: usize) -> Option<u8> {
    bytes.get(position).copied()
}

fn skip_whitespace(bytes: &[u8], position: &mut usize) {
    while matches!(peek(bytes, *position), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        *position += 1;
    }
}

fn expect(bytes: &[u8], position: &mut usize, expected: u8, line: usize) -> Result<(), DataError> {
    if peek(bytes, *position) == Some(expected) {
        *position += 1;
        Ok(())
    } else {
        Err(invalid(line, "unexpected token"))
    }
}

fn parse_string(bytes: &[u8], position: &mut usize, line: usize) -> Result<String, DataError> {
    expect(bytes, position, b'"', line)?;
    let mut output = String::new();
    loop {
        match peek(bytes, *position) {
            None => return Err(invalid(line, "unterminated string")),
            Some(b'"') => {
                *position += 1;
                return Ok(output);
            }
            Some(b'\\') => {
                *position += 1;
                let escape = peek(bytes, *position).ok_or_else(|| invalid(line, "bad escape"))?;
                *position += 1;
                match escape {
                    b'"' => output.push('"'),
                    b'\\' => output.push('\\'),
                    b'/' => output.push('/'),
                    b'b' => output.push('\u{0008}'),
                    b'f' => output.push('\u{000C}'),
                    b'n' => output.push('\n'),
                    b'r' => output.push('\r'),
                    b't' => output.push('\t'),
                    b'u' => output.push(parse_unicode_escape(bytes, position, line)?),
                    _ => return Err(invalid(line, "unsupported escape")),
                }
            }
            Some(byte) if byte < 0x20 => return Err(invalid(line, "control character in string")),
            Some(_) => {
                // Copy one UTF-8 character.
                let rest = std::str::from_utf8(&bytes[*position..])
                    .map_err(|_| invalid(line, "invalid UTF-8"))?;
                let character = rest
                    .chars()
                    .next()
                    .ok_or_else(|| invalid(line, "invalid UTF-8"))?;
                output.push(character);
                *position += character.len_utf8();
            }
        }
    }
}

fn parse_unicode_escape(
    bytes: &[u8],
    position: &mut usize,
    line: usize,
) -> Result<char, DataError> {
    if *position + 4 > bytes.len() {
        return Err(invalid(line, "short \\u escape"));
    }
    let hex = std::str::from_utf8(&bytes[*position..*position + 4])
        .map_err(|_| invalid(line, "invalid \\u escape"))?;
    let code = u32::from_str_radix(hex, 16).map_err(|_| invalid(line, "invalid \\u escape"))?;
    *position += 4;
    char::from_u32(code).ok_or_else(|| invalid(line, "invalid code point"))
}

/// Skips a non-string JSON value (number, boolean or null).
fn skip_scalar(bytes: &[u8], position: &mut usize, line: usize) -> Result<(), DataError> {
    let start = *position;
    while let Some(byte) = peek(bytes, *position) {
        if matches!(byte, b',' | b'}' | b' ' | b'\t' | b'\r' | b'\n') {
            break;
        }
        *position += 1;
    }
    if *position == start {
        return Err(invalid(line, "unsupported value"));
    }
    Ok(())
}
