"""
BNLP Model Registry

Global model cache that prevents re-loading the same model from disk
when multiple instances of the same class are created with the same
model path. This can save 5-30 seconds per duplicate instantiation
for large models.

Thread-safe implementation using a module-level lock.
"""

import threading
from typing import Any, Dict, Tuple, Optional

# Global registry: (class_name, model_path) -> model_object
_registry: Dict[Tuple[str, str], Any] = {}
_registry_lock = threading.Lock()


def get_model(class_name: str, model_path: str) -> Optional[Any]:
    """Look up a cached model by class name and path.

    Args:
        class_name: The class name (e.g., "BengaliPOS")
        model_path: The model file path

    Returns:
        The cached model object, or None if not cached
    """
    with _registry_lock:
        return _registry.get((class_name, model_path))


def set_model(class_name: str, model_path: str, model: Any) -> None:
    """Cache a model by class name and path.

    Args:
        class_name: The class name
        model_path: The model file path
        model: The model object to cache
    """
    with _registry_lock:
        _registry[(class_name, model_path)] = model


def get_or_load(class_name: str, model_path: str, loader) -> Any:
    """Get a cached model or load it using the provided loader function.

    Args:
        class_name: The class name
        model_path: The model file path
        loader: Callable that loads the model if not cached

    Returns:
        The model object (from cache or freshly loaded)
    """
    with _registry_lock:
        key = (class_name, model_path)
        if key in _registry:
            return _registry[key]
        model = loader()
        _registry[key] = model
        return model


def clear() -> None:
    """Clear the entire model registry."""
    with _registry_lock:
        _registry.clear()


def size() -> int:
    """Return the number of cached models."""
    with _registry_lock:
        return len(_registry)
