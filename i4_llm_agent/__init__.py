# i4_llm_agent/__init__.py
import logging

# --- Core Functionality ---
from .api_client import call_google_llm_api
from .history import (
    format_history_for_llm,
    get_recent_turns,
    get_dialogue_history,
    select_turns_for_t0,
)
from .prompting import (
    format_refiner_prompt, construct_final_llm_payload,
    assemble_tagged_context, extract_tagged_context, clean_context_tags,
    generate_rag_query, combine_background_context # <-- Added new prompting functions
)
from .memory import manage_tier1_summarization

# --- Utilities ---
# Import specific functions or the whole module based on preference
from .utils import count_tokens # <-- Added utils function

# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Avoids "No handler found" warnings

# --- Optional: Print message on import ---
# Consider making this conditional or removing it for cleaner library use
# print("--- i4_llm_agent package initialized (api_client, history, prompting, memory, utils) ---")

# --- Define __all__ ---
# List all public functions/classes intended for export
__all__ = [
    # api_client
    "call_google_llm_api",
    # history
    "format_history_for_llm", "get_recent_turns",
    "get_dialogue_history", "select_turns_for_t0",
    # prompting
    "format_refiner_prompt", "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context", "clean_context_tags",
    "generate_rag_query", "combine_background_context", # <-- Added
    # memory
    "manage_tier1_summarization",
    # utils
    "count_tokens", # <-- Added
]