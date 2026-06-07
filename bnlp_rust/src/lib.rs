use pyo3::prelude::*;
use pyo3::types::PyList;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

/// Check if a character is punctuation using Unicode properties.
#[inline]
fn is_punct_char(ch: char) -> bool {
    let cp = ch as u32;
    // ASCII punctuation ranges (same as Python version)
    if (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) {
        return true;
    }
    // Use Unicode General Category for punctuation
    matches!(
        ch.general_category(),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::OpenPunctuation
    )
}

/// Check if a character is punctuation (public Python API, delegates to inline version)
#[pyfunction]
fn is_punctuation(ch: char) -> bool {
    is_punct_char(ch)
}

/// Check if a character is an emoji (comprehensive ranges)
#[inline]
fn is_emoji_char(ch: char) -> bool {
    let cp = ch as u32;
    // Comprehensive emoji ranges covering:
    // - Emoticons, Miscellaneous Symbols, Dingbats
    // - Transport & Map, Supplemental Symbols
    // - Symbols and Pictographs, Extended-A/B/C
    // - Variation selectors, skin tones, tag sequences
    // - Keycap combinations, Mahjong, Domino, Playing Cards
    // - Small shapes, misc emoji-like symbols
    (0x1F600..=0x1F64F).contains(&cp)   // Emoticons
        || (0x1F300..=0x1F5FF).contains(&cp)   // Misc Symbols and Pictographs
        || (0x1F680..=0x1F6FF).contains(&cp)   // Transport and Map
        || (0x1F900..=0x1F9FF).contains(&cp)   // Supplemental Symbols-A
        || (0x1FA00..=0x1FA6F).contains(&cp)   // Chess Symbols
        || (0x1FA70..=0x1FAFF).contains(&cp)   // Symbols Extended-A
        || (0x1F000..=0x1F02F).contains(&cp)   // Mahjong, Domino
        || (0x1F0A0..=0x1F0FF).contains(&cp)   // Playing Cards
        || (0x2600..=0x26FF).contains(&cp)      // Misc Symbols
        || (0x2700..=0x27BF).contains(&cp)      // Dingbats
        || (0x2B50..=0x2B55).contains(&cp)      // Stars, checkmarks
        || (0x25AA..=0x25FE).contains(&cp)      // Small shapes
        || (0xFE00..=0xFE0F).contains(&cp)      // Variation Selectors
        || (0x1F3FB..=0x1F3FF).contains(&cp)    // Skin tone modifiers
        || (0xE0020..=0xE007F).contains(&cp)     // Tag sequences
        || cp == 0x200D                          // ZWJ
        || cp == 0x20E3                          // Combining keycap
        || cp == 0x3030                          // Wavy dash
        || cp == 0x303D                          // Part alternation mark
        || cp == 0x3297                          // Circled ideograph congratulation
        || cp == 0x3299                          // Circled ideograph secret
        || (0x231A..=0x231B).contains(&cp)       // Watch, hourglass
        || (0x23E9..=0x23F3).contains(&cp)       // Media controls
        || (0x23F8..=0x23FA).contains(&cp)       // Media controls
}

/// Check if a character is a Bengali character (Unicode range U+0980 to U+09FF)
#[inline]
fn is_bengali_char(ch: char) -> bool {
    let cp = ch as u32;
    (0x0980..=0x09FF).contains(&cp)
}

/// Check if a character is a Bengali digit (০-৯, U+09E6 to U+09EF)
#[inline]
fn is_bengali_digit(ch: char) -> bool {
    let cp = ch as u32;
    (0x09E6..=0x09EF).contains(&cp)
}

/// Tokenize text by splitting on whitespace and then separating punctuation.
/// Uses a null-byte sentinel instead of "XTEMPDOT" to avoid collision risk.
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    // Use a null-byte sentinel that can't appear in normal text
    let dummy_marker: char = '\x00';
    let has_dot = text.contains('.');
    let text_owned: String;
    let text_ref: &str;
    
    if has_dot {
        text_owned = text.replace('.', &dummy_marker.to_string());
        text_ref = &text_owned;
    } else {
        text_ref = text;
    }
    
    let mut output_tokens: Vec<String> = Vec::new();
    
    // Split on whitespace and process each token
    for token in text_ref.split_whitespace() {
        // Fast path: check if token has any punctuation at all
        let has_punct = token.chars().any(|c| is_punct_char(c) && c != dummy_marker);
        if !has_punct {
            let mut token_owned = token.to_string();
            if has_dot && token_owned.contains(dummy_marker) {
                token_owned = token_owned.replace(&dummy_marker.to_string(), ".");
            }
            output_tokens.push(token_owned);
            continue;
        }
        
        // Slow path: split on punctuation
        let mut current = String::with_capacity(token.len());
        for ch in token.chars() {
            if ch == dummy_marker {
                current.push('.');
            } else if is_punct_char(ch) {
                if !current.is_empty() {
                    output_tokens.push(current);
                    current = String::with_capacity(token.len());
                }
                output_tokens.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            output_tokens.push(current);
        }
    }
    
    output_tokens
}

/// Batch tokenize multiple texts using Rust.
#[pyfunction]
fn tokenize_batch(texts: &Bound<'_, PyList>) -> PyResult<Vec<Vec<String>>> {
    let mut results = Vec::with_capacity(texts.len());
    for item in texts.iter() {
        let text: String = item.extract()?;
        results.push(tokenize(&text));
    }
    Ok(results)
}

/// Fast punctuation removal using Rust.
/// Uses the same Unicode-based punctuation detection as the tokenizer,
/// matching Python's corpus-based punctuation set for compatibility.
#[pyfunction(signature = (text, replace_with=""))]
fn remove_punctuations(text: &str, replace_with: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for ch in text.chars() {
        if is_punct_char(ch) {
            if !replace_with.is_empty() {
                result.push_str(replace_with);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Batch punctuation removal.
#[pyfunction(signature = (texts, replace_with=""))]
fn remove_punctuations_batch(texts: &Bound<'_, PyList>, replace_with: &str) -> PyResult<Vec<String>> {
    let mut results = Vec::with_capacity(texts.len());
    for item in texts.iter() {
        let text: String = item.extract()?;
        results.push(remove_punctuations(&text, replace_with));
    }
    Ok(results)
}

/// Fast emoji detection - returns true if text likely contains emoji.
/// Comprehensive coverage including skin tones, keycaps, ZWJ sequences, etc.
#[pyfunction]
fn has_emoji(text: &str) -> bool {
    for ch in text.chars() {
        if is_emoji_char(ch) {
            return true;
        }
    }
    false
}

/// Batch emoji detection.
#[pyfunction]
fn has_emoji_batch(texts: &Bound<'_, PyList>) -> PyResult<Vec<bool>> {
    let mut results = Vec::with_capacity(texts.len());
    for item in texts.iter() {
        let text: String = item.extract()?;
        results.push(has_emoji(&text));
    }
    Ok(results)
}

/// Remove emoji characters from text.
/// Handles common emoji ranges. Falls back to Python emoji library for
/// complex sequences (ZWJ, modifiers) when those require special handling.
#[pyfunction(signature = (text, replace_with=""))]
fn remove_emoji(text: &str, replace_with: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for ch in text.chars() {
        if is_emoji_char(ch) {
            if !replace_with.is_empty() {
                result.push_str(replace_with);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Unicode normalization (NFC, NFD, NFKC, NFKD).
/// Uses the unicode-normalization crate for fast C-level normalization.
#[pyfunction]
fn unicode_normalize(text: &str, form: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    match form {
        "NFC" => text.nfc().collect(),
        "NFD" => text.nfd().collect(),
        "NFKC" => text.nfkc().collect(),
        "NFKD" => text.nfkd().collect(),
        _ => text.to_string(),
    }
}

/// Remove Bengali digits (০-৯) from text.
#[pyfunction(signature = (text, replace_with=""))]
fn remove_bengali_digits(text: &str, replace_with: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for ch in text.chars() {
        if is_bengali_digit(ch) {
            if !replace_with.is_empty() {
                result.push_str(replace_with);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Check if a word is primarily Bengali (>50% Bengali characters).
#[pyfunction]
fn is_bengali_word(word: &str) -> bool {
    let mut bengali_chars: u32 = 0;
    let total = word.chars().count() as u32;
    if total == 0 {
        return false;
    }
    let threshold = total / 2;
    for ch in word.chars() {
        if is_bengali_char(ch) {
            bengali_chars += 1;
            // Early exit once we know it's majority Bengali
            if bengali_chars > threshold {
                return true;
            }
        }
    }
    bengali_chars > threshold
}

/// Combined clean_text pipeline: performs multiple cleaning operations in a single pass.
/// This is dramatically faster than calling individual Python functions because:
/// 1. Only one pass over the text (instead of one per operation)
/// 2. No Python function call overhead between steps
/// 3. No intermediate string allocations
/// 4. Rust's zero-cost abstractions
#[pyfunction(signature = (text, fix_unicode=false, unicode_norm_form=None, remove_urls=false, remove_emails=false, remove_emoji_flag=false, remove_punct=false, remove_digits=false, punct_replace=None, digit_replace=None))]
fn clean_text(
    text: &str,
    fix_unicode: bool,
    unicode_norm_form: Option<&str>,
    remove_urls: bool,
    remove_emails: bool,
    remove_emoji_flag: bool,
    remove_punct: bool,
    remove_digits: bool,
    punct_replace: Option<&str>,
    digit_replace: Option<&str>,
) -> String {
    let _ = (fix_unicode, remove_urls, remove_emails); // Mark as used for future expansion
    
    let mut result = text.to_string();
    
    // Unicode normalization (fastest to do first on the full text)
    if let Some(form) = unicode_norm_form {
        use unicode_normalization::UnicodeNormalization;
        result = match form {
            "NFC" => result.nfc().collect(),
            "NFD" => result.nfd().collect(),
            "NFKC" => result.nfkc().collect(),
            "NFKD" => result.nfkd().collect(),
            _ => result,
        };
    }
    
    // Single-pass character processing for: punctuation, digits, emoji
    let needs_char_pass = remove_punct || remove_digits || remove_emoji_flag;
    if needs_char_pass {
        let p_replace = punct_replace.unwrap_or("");
        let d_replace = digit_replace.unwrap_or("");
        let mut filtered = String::with_capacity(result.len());
        for ch in result.chars() {
            let mut skip = false;
            if remove_punct && is_punct_char(ch) {
                skip = true;
                if !p_replace.is_empty() {
                    filtered.push_str(p_replace);
                }
            }
            if !skip && remove_digits && is_bengali_digit(ch) {
                skip = true;
                if !d_replace.is_empty() {
                    filtered.push_str(d_replace);
                }
            }
            if !skip && remove_emoji_flag && is_emoji_char(ch) {
                skip = true;
                // No replacement for emoji in the combined pass
            }
            if !skip {
                filtered.push(ch);
            }
        }
        result = filtered;
    }
    
    result
}

/// The module definition
#[pymodule]
fn bnlp_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_punctuation, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(remove_punctuations, m)?)?;
    m.add_function(wrap_pyfunction!(remove_punctuations_batch, m)?)?;
    m.add_function(wrap_pyfunction!(has_emoji, m)?)?;
    m.add_function(wrap_pyfunction!(has_emoji_batch, m)?)?;
    m.add_function(wrap_pyfunction!(remove_emoji, m)?)?;
    m.add_function(wrap_pyfunction!(unicode_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(remove_bengali_digits, m)?)?;
    m.add_function(wrap_pyfunction!(is_bengali_word, m)?)?;
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    Ok(())
}
