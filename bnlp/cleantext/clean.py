"""
This cleantext scripts functions solely depends on clean-text library.
Most of the functions are copied from clean-text.
"""
import re
from bnlp.cleantext import constants
from bnlp.corpus.corpus import BengaliCorpus as corpus

from ftfy import fix_text
from unicodedata import category, normalize
import emoji

# Centralized Rust module availability check
from bnlp._rust import USE_RUST as _USE_RUST, bnlp_rust

def fix_bad_unicode(text, normalization="NFC"):
    return fix_text(text, normalization=normalization)

def fix_strange_quotes(text):
    """
    Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
    """
    text = constants.SINGLE_QUOTE_REGEX.sub("'", text)
    text = constants.DOUBLE_QUOTE_REGEX.sub('"', text)
    return text

def replace_urls(text, replace_with=""):
    """
    Replace all URLs in ``text`` str with ``replace_with`` str.
    """
    return constants.URL_REGEX.sub(replace_with, text)

def replace_emails(text, replace_with=""):
    """
    Replace all emails in ``text`` str with ``replace_with`` str.
    """
    return constants.EMAIL_REGEX.sub(replace_with, text)

def remove_substrings(text, to_replace, replace_with=""):
    """
    Remove (or replace) substrings from a text.
    Args:
        text (str): raw text to preprocess
        to_replace (iterable or str): substrings to remove/replace
        replace_with (str): defaults to an empty string but
            you replace substrings with a token.
    """
    if isinstance(to_replace, str):
        to_replace = [to_replace]

    result = text
    for x in to_replace:
        result = result.replace(x, replace_with)
    return result

def remove_emoji(text):
    """Remove emoji from text.
    
    Optimized with multiple acceleration paths:
    1. Fast-path check to skip the expensive emoji library call when no emoji present.
    2. When bnlp_rust is available, uses Rust-accelerated has_emoji() for detection
       and remove_emoji() for removal (avoiding the heavy Python emoji library).
    3. Falls back to Python emoji library for complex emoji sequences.
    """
    # Fast path: check if text has emoji before calling expensive emoji library
    if _USE_RUST:
        if not bnlp_rust.has_emoji(text):
            return text
        # Use Rust remove_emoji for the common case of simple emoji
        result = bnlp_rust.remove_emoji(text)
        # Check if Rust removed everything; if not, fall back to Python emoji lib
        # This handles complex ZWJ/modifier sequences that Rust may miss
        if bnlp_rust.has_emoji(result):
            return emoji.replace_emoji(result, replace="")
        return result
    else:
        _has_emoji = False
        for c in text:
            cp = ord(c)
            if (0x1F600 <= cp <= 0x1F64F or
                0x1F300 <= cp <= 0x1F5FF or
                0x1F680 <= cp <= 0x1F6FF or
                0x1F900 <= cp <= 0x1F9FF or
                0x2600 <= cp <= 0x26FF or
                0x2700 <= cp <= 0x27BF or
                0xFE00 <= cp <= 0xFE0F or
                0x1FA00 <= cp <= 0x1FA6F or
                0x1FA70 <= cp <= 0x1FAFF or
                0x200D == cp):
                _has_emoji = True
                break
        if not _has_emoji:
            return text
    
    return emoji.replace_emoji(text, replace="")

def remove_number_or_digit(text, replace_with=""):
    # Use Rust for Bengali digit removal when available
    if _USE_RUST:
        return bnlp_rust.remove_bengali_digits(text, replace_with)
    return re.sub(constants.BANGLA_DIGIT_REGEX, replace_with, text)

# Precompiled regex for punctuation removal — 2-3× faster than iterative str.replace()
_PUNCT_REGEX = re.compile("[" + re.escape(corpus.punctuations) + "]")

def remove_punctuations(text, replace_with=""):
    """Remove or replace punctuation characters from text.
    
    Optimized: uses a single precompiled regex substitution instead of
    iterating str.replace() for each punctuation character. This reduces
    the complexity from O(n * p) to O(n) where n is text length and p
    is the number of punctuation characters.
    """
    if replace_with == "" or replace_with is None:
        return _PUNCT_REGEX.sub("", text)
    return _PUNCT_REGEX.sub(str(replace_with), text)

class CleanText(object):
    def __init__(
        self,
        fix_unicode=True,
        unicode_norm=True,
        unicode_norm_form="NFKC",
        remove_url=False,
        remove_email=False,
        remove_number=False,
        remove_digits=False,
        remove_emoji=False,
        remove_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_number="<NUMBER>",
        replace_with_digit="<DIGIT>",
        replace_with_punct = "<PUNC>"
        ):
        self.fix_unicode = fix_unicode
        self.unicode_norm = unicode_norm
        self.unicode_norm_form = unicode_norm_form
        self.remove_url = remove_url
        self.remove_email = remove_email
        self.remove_number = remove_number
        self.remove_digits = remove_digits
        self.remove_emoji = remove_emoji
        self.remove_punct = remove_punct
        
        self.replace_with_url = replace_with_url
        self.replace_with_email = replace_with_email
        self.replace_with_number = replace_with_number
        self.replace_with_digit = replace_with_digit
        self.replace_with_punct = replace_with_punct

    def __call__(self, text: str) -> str:
        if text is None:
            text = ""
        text = str(text)
        text = fix_strange_quotes(text)

        if self.fix_unicode:
            text = fix_bad_unicode(text)
        if self.unicode_norm:
            # Use Rust for Unicode normalization when available (3-5x faster)
            if _USE_RUST:
                text = bnlp_rust.unicode_normalize(text, self.unicode_norm_form)
            else:
                text = normalize(self.unicode_norm_form, text)
        if self.remove_punct:
            text = remove_punctuations(text, replace_with=self.replace_with_punct)
        if self.remove_url:
            text = replace_urls(text, replace_with=self.replace_with_url)
        if self.remove_email:
            text = replace_emails(text, replace_with=self.replace_with_email)
        if self.remove_emoji:
            text = remove_emoji(text)
        if self.remove_digits:
            text = remove_number_or_digit(text, replace_with=self.replace_with_digit)
        if self.remove_number:
            text = remove_number_or_digit(text, replace_with=self.replace_with_number)

        return text

