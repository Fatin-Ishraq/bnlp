"""
Centralized Rust module availability check for BNLP.

Instead of each module independently trying to import bnlp_rust,
this single module provides a shared USE_RUST flag and a reference
to the bnlp_rust module (or None if not available).
"""

USE_RUST = False
bnlp_rust = None

try:
    import bnlp_rust as _bnlp_rust
    bnlp_rust = _bnlp_rust
    USE_RUST = True
except ImportError:
    pass
