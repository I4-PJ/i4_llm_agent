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
    construct_final_llm_payload, # <<< SIGNATURE UPDATED: includes long_term_goal
    assemble_tagged_context, extract_tagged_context,
    clean_context_tags, generate_rag_query, combine_background_context,
    process_system_prompt,
)
from .memory import manage_tier1_summarization # T1/T2 memory management

# --- RAG Cache Functionality ---
from .cache import (
    initialize_rag_cache_table,   # For pipe startup
    update_rag_cache,             # Step 1 orchestrator
    select_final_context,         # Step 2 orchestrator
    # DB Helpers
    get_rag_cache, # Export sync getter
    add_or_update_rag_cache # Export sync writer if needed elsewhere
)

# --- Utilities ---
from .utils import count_tokens, calculate_string_similarity

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
    "DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE",
    "DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT",
    "DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT",
    "format_stateless_refiner_prompt",
    "format_cache_update_prompt",
    "format_final_context_selection_prompt",
    "refine_external_context", # Stateless orchestrator
    "construct_final_llm_payload", # Function signature updated, but name is same
    "assemble_tagged_context", "extract_tagged_context",
    "clean_context_tags", "generate_rag_query",
    "combine_background_context",
    "process_system_prompt",
    # memory
    "manage_tier1_summarization",
    # cache
    "initialize_rag_cache_table",
    "update_rag_cache",
    "select_final_context",
    "get_rag_cache",
    "add_or_update_rag_cache",
    # utils
    "count_tokens",
    "calculate_string_similarity",
]