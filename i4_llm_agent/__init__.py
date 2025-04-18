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
    # NEW: Default template for RAG Cache refiner
    DEFAULT_RAG_CACHE_REFINER_TEMPLATE_TEXT,
    # NEW: Formatting function for RAG Cache refiner prompt
    format_curated_refiner_prompt,
    # Existing stateless refiner function (if kept)
    refine_external_context,
    # Existing prompting functions
    construct_final_llm_payload,
    assemble_tagged_context,
    extract_tagged_context,
    clean_context_tags,
    generate_rag_query,
    combine_background_context,
    process_system_prompt,
    # Note: format_refiner_prompt is the one for stateless refinement
    format_refiner_prompt as format_stateless_refiner_prompt
)
from .memory import manage_tier1_summarization # T1/T2 memory management

# --- NEW: RAG Cache Functionality ---
from .cache import (
    initialize_rag_cache_table,   # For pipe startup
    refine_and_update_rag_cache,  # Main orchestration function for pipe
    get_rag_cache,                # Possibly useful for direct inspection?
    add_or_update_rag_cache       # Possibly useful for direct manipulation?
)

# --- Utilities ---
from .utils import count_tokens

# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Define __all__ ---
# List all functions/constants intended for public use by consumers (like the pipe)
__all__ = [
    # api_client
    "call_google_llm_api",
    # history
    "format_history_for_llm", "get_recent_turns",
    "get_dialogue_history", "select_turns_for_t0",
    "DIALOGUE_ROLES",
    # prompting
    "DEFAULT_RAG_CACHE_REFINER_TEMPLATE_TEXT", # NEW Constant
    "format_curated_refiner_prompt", # NEW Function
    "refine_external_context", # Existing stateless refiner
    "format_stateless_refiner_prompt", # Existing stateless formatter
    "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context", "clean_context_tags",
    "generate_rag_query", "combine_background_context",
    "process_system_prompt",
    # memory
    "manage_tier1_summarization",
    # cache (NEW)
    "initialize_rag_cache_table",
    "refine_and_update_rag_cache",
    "get_rag_cache",
    "add_or_update_rag_cache",
    # utils
    "count_tokens",
]