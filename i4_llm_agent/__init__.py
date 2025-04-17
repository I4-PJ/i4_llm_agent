# i4_llm_agent/__init__.py
import logging

# --- Core Functionality ---
from .api_client import call_google_llm_api
from .history import (
    format_history_for_llm,
    get_recent_turns,
    get_dialogue_history,
    select_turns_for_t0,
    DIALOGUE_ROLES,
)
from .prompting import (
    format_refiner_prompt, construct_final_llm_payload,
    assemble_tagged_context, extract_tagged_context, clean_context_tags,
    generate_rag_query, combine_background_context,
    process_system_prompt # <-- Already here
)
from .memory import manage_tier1_summarization

# --- Utilities ---
from .utils import count_tokens

# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Define __all__ ---
__all__ = [
    # api_client
    "call_google_llm_api",
    # history
    "format_history_for_llm", "get_recent_turns",
    "get_dialogue_history", "select_turns_for_t0",
    "DIALOGUE_ROLES",
    # prompting
    "format_refiner_prompt", "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context", "clean_context_tags",
    "generate_rag_query", "combine_background_context",
    "process_system_prompt", # <-- Already here
    # memory
    "manage_tier1_summarization",
    # utils
    "count_tokens",
]