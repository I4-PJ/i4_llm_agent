# [[START MODIFIED __init__.py]]
# i4_llm_agent/__init__.py
import logging

# --- Core Functionality ---
from .api_client import call_google_llm_api # Existing API client
# History Utils (Existing)
from .history import (
    format_history_for_llm, get_recent_turns, get_dialogue_history,
    select_turns_for_t0, DIALOGUE_ROLES,
)
# Prompting Utils (Existing + New Defaults)
from .prompting import (
    # Default Templates
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
    # Formatting Functions
    format_stateless_refiner_prompt,
    format_cache_update_prompt,
    format_final_context_selection_prompt,
    # Other Prompting Utilities
    refine_external_context, # Stateless orchestrator
    construct_final_llm_payload,
    assemble_tagged_context, extract_tagged_context,
    clean_context_tags, generate_rag_query, combine_background_context,
    process_system_prompt,
)
# Memory Management (Existing)
from .memory import manage_tier1_summarization # T1/T2 memory management

# RAG Cache Functionality (Existing Orchestrators)
from .cache import (
    # DB Helpers moved to database.py, but keep orchestrators here for now
    update_rag_cache,             # Step 1 orchestrator
    select_final_context,         # Step 2 orchestrator
)

# --- NEW: Session Management ---
from .session import SessionManager

# --- NEW: Database Operations ---
from .database import (
    # Initialization
    initialize_sqlite_tables,
    # SQLite T1 Summaries
    add_tier1_summary, get_recent_tier1_summaries, get_tier1_summary_count,
    get_oldest_tier1_summary, delete_tier1_summary,
    get_max_t1_end_index,
    check_t1_summary_exists, # <<< ADDED
    # SQLite RAG Cache
    add_or_update_rag_cache, get_rag_cache,
    # ChromaDB T2
    get_or_create_chroma_collection, add_to_chroma_collection,
    query_chroma_collection, get_chroma_collection_count,
    CHROMADB_AVAILABLE, # Export flag
)

# --- NEW: Orchestration ---
from .orchestration import SessionPipeOrchestrator

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
    "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context",
    "clean_context_tags", "generate_rag_query",
    "combine_background_context",
    "process_system_prompt",
    # memory
    "manage_tier1_summarization",
    # cache (Existing Orchestrators)
    "update_rag_cache",
    "select_final_context",
    # session (NEW)
    "SessionManager",
    # database (NEW - selective export)
    "initialize_sqlite_tables",
    "add_tier1_summary", "get_recent_tier1_summaries", "get_tier1_summary_count",
    "get_oldest_tier1_summary", "delete_tier1_summary",
    "get_max_t1_end_index",
    "check_t1_summary_exists", # <<< ADDED
    "add_or_update_rag_cache", "get_rag_cache",
    "get_or_create_chroma_collection", "add_to_chroma_collection",
    "query_chroma_collection", "get_chroma_collection_count",
    "CHROMADB_AVAILABLE",
    # orchestration (NEW)
    "SessionPipeOrchestrator",
    # utils
    "count_tokens",
    "calculate_string_similarity",
]
# [[END MODIFIED __init__.py]]