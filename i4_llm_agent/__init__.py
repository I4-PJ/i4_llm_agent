# i4_llm_agent/__init__.py
import logging

# --- Core Functionality ---
from .api_client import call_google_llm_api
from .history import format_history_for_llm, get_recent_turns, get_dialogue_history
from .prompting import (
    format_refiner_prompt, construct_final_llm_payload,
    assemble_tagged_context, extract_tagged_context, clean_context_tags
)
from .memory import manage_tier1_summarization

from .history import (
    format_history_for_llm,
    get_recent_turns,
    get_dialogue_history,
    select_turns_for_t0,
)

# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Optional: Print message on import ---
print("--- i4_llm_agent package initialized (api_client, history, prompting, memory) ---")

# --- Define __all__ ---
__all__ = [
    "call_google_llm_api", "format_history_for_llm", "get_recent_turns",
    "get_dialogue_history", "format_refiner_prompt", "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context", "clean_context_tags",
    "manage_tier1_summarization", "select_turns_for_t0",
]