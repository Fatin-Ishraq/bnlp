import pickle
from sklearn_crfsuite import CRF
from nltk.tag.util import untag
from bnlp.utils.registry import get_or_load

def features(sentence, index):
    """sentence: [w1, w2, ...], index: the index of the word
    
    Optimized: caches sentence[index] to avoid repeated lookups,
    computes .upper()/.lower() once, and avoids redundant
    string operations.
    """
    word = sentence[index]
    word_upper = word.upper()
    word_lower = word.lower()
    is_first = index == 0
    is_last = index == len(sentence) - 1
    
    return {
        "word": word,
        "is_first": is_first,
        "is_last": is_last,
        "is_capitalized": word_upper[0] == word[0],
        "is_all_caps": word_upper == word,
        "is_all_lower": word_lower == word,
        "prefix-1": word[0],
        "prefix-2": word[:2],
        "prefix-3": word[:3],
        "suffix-1": word[-1],
        "suffix-2": word[-2:],
        "suffix-3": word[-3:],
        "prev_word": "" if is_first else sentence[index - 1],
        "next_word": "" if is_last else sentence[index + 1],
        "has_hyphen": "-" in word,
        "is_numeric": word.isdigit(),
        "capitals_inside": word[1:].lower() != word[1:],
    }

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        try:
            X.append([features(untag(tagged), index) for index in range(len(tagged))])
            y.append([tag for _, tag in tagged])
        except Exception as e:
            print(e)

    return X, y

def load_pickle_model(model_path: str) -> CRF:
    """Load a pickled CRF model, using the global model registry to avoid
    re-loading the same model from disk on every instantiation."""
    def _load():
        with open(model_path, "rb") as pkl_model:
            return pickle.load(pkl_model)
    return get_or_load("CRF", model_path, _load)
