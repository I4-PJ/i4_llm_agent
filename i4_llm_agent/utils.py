# i4_llm_agent/utils.py
import logging
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
