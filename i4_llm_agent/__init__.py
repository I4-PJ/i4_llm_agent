# i4_llm_agent/__init__.py
import logging

# --- Core Functionality ---
from .api_client import call_google_llm_api
from .history import (
    format_history_for_llm, get_recent_turns, get_dialogue_history,
    select_turns_for_t0, DIALOGUE_ROLES,
)
from .prompting import (
    # Default Templates
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE, # Existing stateless default
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,        # NEW Step 1 default
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT, # NEW Step 2 default
    # Formatting Functions
    format_stateless_refiner_prompt,           # Existing stateless formatter
    format_cache_update_prompt,                # NEW Step 1 formatter
    format_final_context_selection_prompt,     # NEW Step 2 formatter
    # Other Prompting Utilities
    refine_external_context, # Existing stateless orchestrator
    construct_final_llm_payload, assemble_tagged_context, extract_tagged_context,
    clean_context_tags, generate_rag_query, combine_background_context, # <<< combine_background_context is here
    process_system_prompt,
)
from .memory import manage_tier1_summarization # T1/T2 memory management

# --- NEW: RAG Cache Functionality ---
from .cache import (
    initialize_rag_cache_table,   # For pipe startup
    update_rag_cache,             # NEW Step 1 orchestrator
    select_final_context,         # NEW Step 2 orchestrator
    # DB Helpers (optional export if needed directly)
    # get_rag_cache,
    # add_or_update_rag_cache
)

# --- Utilities ---
from .utils import count_tokens

# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Define __all__ ---
# List all functions/constants intended for public use by consumers
__all__ = [
    # api_client
    "call_google_llm_api",
    # history
    "format_history_for_llm", "get_recent_turns", "get_dialogue_history",
    "select_turns_for_t0", "DIALOGUE_ROLES",
    # prompting
    "DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE", # Renamed from DEFAULT_REFINER_PROMPT_TEMPLATE if applicable
    "DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT",        # NEW
    "DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT", # NEW
    "format_stateless_refiner_prompt",
    "format_cache_update_prompt",                # NEW
    "format_final_context_selection_prompt",     # NEW
    "refine_external_context", # Stateless orchestrator
    "construct_final_llm_payload", "assemble_tagged_context", "extract_tagged_context",
    "clean_context_tags", "generate_rag_query",
    "combine_background_context",                # <<< Added to __all__
    "process_system_prompt",
    # memory
    "manage_tier1_summarization",
    # cache (NEW)
    "initialize_rag_cache_table",
    "update_rag_cache",             # NEW Step 1
    "select_final_context",         # NEW Step 2
    # utils
    "count_tokens",
]

# Cleanup potential duplicate export if renamed stateless default template was present before
if "DEFAULT_REFINER_PROMPT_TEMPLATE" in __all__:
     __all__.remove("DEFAULT_REFINER_PROMPT_TEMPLATE")