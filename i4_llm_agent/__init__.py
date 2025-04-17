# i4_llm_agent/__init__.py
import logging

# --- Core Functionality ---
# API Client
from .api_client import call_google_llm_api

# History Management
from .history import (
    format_history_for_llm,
    get_recent_turns,
    get_dialogue_history,
)

# Prompting & Context Utilities
from .prompting import (
    format_refiner_prompt,
    construct_final_llm_payload,
    assemble_tagged_context,
    extract_tagged_context,
    clean_context_tags,
)

# Memory Management
from .memory import (
    manage_tier1_summarization,
    # _select_history_slice_by_tokens # Keep this internal to memory.py for now
)

# --- Configure basic logging for the library ---
# This allows applications using the library to configure logging handlers easily.
# Libraries should generally NOT add handlers, only loggers.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Prevents "No handler found" warnings

# --- Optional: Print message on import ---
print(
    "--- i4_llm_agent package initialized (api_client, history, prompting, memory) ---"
)

# --- Define __all__ for explicit public API (optional but good practice) ---
__all__ = [
    "call_google_llm_api",
    "format_history_for_llm",
    "get_recent_turns",
    "get_dialogue_history",
    "format_refiner_prompt",
    "construct_final_llm_payload",
    "assemble_tagged_context",
    "extract_tagged_context",
    "clean_context_tags",
    "manage_tier1_summarization",
]
