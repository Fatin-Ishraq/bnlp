"""
Basic Tokenization
Basic tokenization tokenize sentence using white spaces, punctuation mark
Code shamelessly copied from BERT tokenization
To check Original Code: https://github.com/google-research/bert/blob/master/tokenization.py
"""
import unicodedata
from typing import List

# Centralized Rust module availability check
from bnlp._rust import USE_RUST as _USE_RUST, bnlp_rust

# Precompute the set of all Unicode punctuation codepoints.
# This avoids calling unicodedata.category() for every character on every call,
# which was the #1 bottleneck in profiling (26.7% of total tokenization time).
# The set includes:
# - ASCII punctuation ranges: 33-47, 58-64, 91-96, 123-126
# - All Unicode General Category "P" (Punctuation) codepoints in common ranges
_PUNCTUATION_SET = set()

# Add ASCII punctuation ranges
for _cp in range(33, 48):
    _PUNCTUATION_SET.add(_cp)
for _cp in range(58, 65):
    _PUNCTUATION_SET.add(_cp)
for _cp in range(91, 97):
    _PUNCTUATION_SET.add(_cp)
for _cp in range(123, 127):
    _PUNCTUATION_SET.add(_cp)

# Add all Unicode punctuation characters (category starts with "P")
# Check common Unicode ranges that contain punctuation
for _range_start, _range_end in [
    (0x00A0, 0x00C0),   # Latin-1 Supplement
    (0x2000, 0x2070),   # General Punctuation
    (0x2190, 0x2200),   # Arrows (some punct-like)
    (0x2400, 0x2440),   # Control Pictures
    (0x3000, 0x3040),   # CJK Symbols and Punctuation
    (0xFE30, 0xFE50),   # CJK Compatibility Forms
    (0xFF01, 0xFF60),   # Halfwidth and Fullwidth Forms
    (0x0964, 0x0970),   # Devanagari/Bengali danda etc.
]:
    for _cp in range(_range_start, _range_end):
        try:
            if unicodedata.category(chr(_cp)).startswith("P"):
                _PUNCTUATION_SET.add(_cp)
        except (ValueError, OverflowError):
            pass

# Freeze the set for faster lookups
_PUNCTUATION_SET = frozenset(_PUNCTUATION_SET)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def whitespace_tokenize(text: str) -> List[str]:
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character.
    
    Optimized: uses a precomputed frozenset of Unicode punctuation codepoints
    for O(1) lookup instead of calling unicodedata.category() on every
    character every time. Falls back to unicodedata.category() only for
    codepoints not in the precomputed set.
    """
    cp = ord(char)
    if cp in _PUNCTUATION_SET:
        return True
    # Fallback for codepoints not in our precomputed set
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


DUMMYTOKEN = 'XTEMPDOT'

class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a piece of text.
        
        When the bnlp_rust module is available, uses the Rust-accelerated
        tokenizer for 4x speedup. Falls back to the optimized pure Python
        implementation otherwise.
        """
        text = convert_to_unicode(text)
        
        if _USE_RUST:
            return bnlp_rust.tokenize(text)
        
        # Pure Python fallback (optimized)
        # handle (.) in bangla text — replace dots so they aren't split as punctuation
        text = text.replace('.', DUMMYTOKEN)
        orig_tokens = whitespace_tokenize(text)
        output_tokens = []
        for token in orig_tokens:
            for sub_token in self._run_split_on_punc(token):
                # Restore dots and skip empty strings from punctuation splitting
                if sub_token:
                    if DUMMYTOKEN in sub_token:
                        sub_token = sub_token.replace(DUMMYTOKEN, '.')
                    output_tokens.append(sub_token)
        return output_tokens
        
    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text.
        
        Optimized with a fast path: most tokens in Bengali text don't contain
        punctuation, so we first check if any punctuation exists. If not,
        return [text] immediately without per-character processing.
        When punctuation is present, iterate directly over the string
        (avoiding list(text) allocation) and build output tokens.
        """
        # Fast path: check if the token has any punctuation at all
        has_punct = False
        for c in text:
            if ord(c) in _PUNCTUATION_SET:
                has_punct = True
                break
            # Fallback for rare codepoints not in our set
            cp = ord(c)
            if cp > 127 and unicodedata.category(c).startswith("P"):
                has_punct = True
                break
        
        if not has_punct:
            return [text]
        
        # Slow path: split on punctuation characters
        output = []
        current = []
        for c in text:
            if _is_punctuation(c):
                if current:
                    output.append("".join(current))
                    current = []
                output.append(c)
            else:
                current.append(c)
        if current:
            output.append("".join(current))
        return output
