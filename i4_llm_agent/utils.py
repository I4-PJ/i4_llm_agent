# [[START MODIFIED utils.py]]
# i4_llm_agent/utils.py
import logging
import difflib # Added for SequenceMatcher
from typing import Any, Optional

# --- Tiktoken Import Handling ---
# Make dependency optional at the library level, but functions using it will fail if not installed.
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # logger.debug("tiktoken library loaded successfully.") # Optional: log on import
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    # logger.warning("tiktoken library not found. Token counting functions will not work.") # Optional: log on import

logger = logging.getLogger(__name__) # i4_llm_agent.utils

# --- Token Counting Function ---
def count_tokens(text: Optional[str], tokenizer: Optional[Any]) -> int:
    """
    Counts the number of tokens in a given text using the provided tokenizer.

    Args:
        text: The text string to count tokens for.
        tokenizer: An initialized tokenizer instance (e.g., from tiktoken)
                   with an .encode() method.

    Returns:
        The number of tokens, or 0 if input is invalid, tokenizer is missing,
        or an error occurs during encoding.
    """
    if not text or not isinstance(text, str):
        # logger.debug("count_tokens: Input text is empty or not a string, returning 0.")
        return 0
    if not tokenizer:
        logger.warning("count_tokens: Tokenizer instance is missing, returning 0.")
        return 0
    if not hasattr(tokenizer, 'encode'):
        logger.error("count_tokens: Provided tokenizer object lacks an 'encode' method, returning 0.")
        return 0

    try:
        # Attempt to encode the text and return the length of the resulting token list
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"count_tokens: Error during tokenization encode: {e}", exc_info=False) # Keep logs cleaner
        return 0

# --- NEW: String Similarity Function ---
def calculate_string_similarity(
    text_a: Optional[str],
    text_b: Optional[str],
    method: str = 'sequencematcher',
    lowercase: bool = True
) -> float:
    """
    Calculates similarity between two strings using the specified method.

    Args:
        text_a: The first string.
        text_b: The second string.
        method: The similarity method ('sequencematcher'). Default is 'sequencematcher'.
        lowercase: Whether to convert texts to lowercase before comparison. Default is True.

    Returns:
        A similarity score between 0.0 and 1.0. Returns 0.0 if inputs are invalid,
        the method is unknown, or an error occurs.
    """
    if not text_a or not isinstance(text_a, str) or not text_b or not isinstance(text_b, str):
        # logger.debug("calculate_string_similarity: One or both inputs invalid, returning 0.0.")
        return 0.0

    str_a = text_a
    str_b = text_b

    if lowercase:
        str_a = str_a.lower()
        str_b = str_b.lower()

    try:
        if method == 'sequencematcher':
            # Use difflib's SequenceMatcher. ratio() gives a float in [0, 1].
            # It measures the similarity as twice the number of matching characters
            # divided by the total number of characters in both sequences.
            # isjunk=None means no characters are ignored.
            # autojunk=False prevents the heuristic junk detection.
            similarity = difflib.SequenceMatcher(None, str_a, str_b, autojunk=False).ratio()
            # logger.debug(f"SequenceMatcher similarity: {similarity:.4f}") # Optional debug
            return similarity
        # Add other methods like 'jaccard' here if needed later
        # elif method == 'jaccard':
        #     set_a = set(str_a.split())
        #     set_b = set(str_b.split())
        #     intersection = len(set_a.intersection(set_b))
        #     union = len(set_a.union(set_b))
        #     return float(intersection) / union if union > 0 else 0.0
        else:
            logger.warning(f"calculate_string_similarity: Unknown method '{method}'. Returning 0.0.")
            return 0.0
    except Exception as e:
        logger.error(f"calculate_string_similarity: Error calculating similarity ({method}): {e}", exc_info=False)
        return 0.0
# [[END MODIFIED utils.py]]