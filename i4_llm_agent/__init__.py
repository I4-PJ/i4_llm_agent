# --- START OF FILE __init__.py ---

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
# Prompting Utils (Modified: Added Inventory Template/Formatter)
from .prompting import (
    # Default Templates
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT, # Updated template
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,        # <<< NEW Inventory Update Template
    # Formatting Functions
    format_stateless_refiner_prompt,
    format_cache_update_prompt,
    format_final_context_selection_prompt,
    format_inventory_update_prompt,                # <<< NEW Inventory Update Formatter
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
    update_rag_cache,             # Step 1 orchestrator
    select_final_context,         # Step 2 orchestrator
)

# --- Session Management (Existing) ---
from .session import SessionManager

# --- Database Operations (Modified: Added Inventory Table/Functions) ---
from .database import (
    # Initialization
    initialize_sqlite_tables,
    # SQLite T1 Summaries (Existing)
    add_tier1_summary, get_recent_tier1_summaries, get_tier1_summary_count,
    get_oldest_tier1_summary, delete_tier1_summary,
    get_max_t1_end_index,
    check_t1_summary_exists,
    # SQLite RAG Cache (Existing)
    add_or_update_rag_cache, get_rag_cache,
    # SQLite Inventory (Added previously)
    initialize_inventory_table,
    get_character_inventory_data,
    add_or_update_character_inventory,
    get_all_inventories_for_session,
    # ChromaDB T2 (Existing)
    get_or_create_chroma_collection, add_to_chroma_collection,
    query_chroma_collection, get_chroma_collection_count,
    CHROMADB_AVAILABLE, # Export flag
)

# --- Orchestration (Existing) ---
from .orchestration import SessionPipeOrchestrator

# --- Utilities (Existing) ---
from .utils import count_tokens, calculate_string_similarity

# --- Inventory Management (NEW Module Import) ---
try:
    from .inventory import (
        format_inventory_for_prompt,
        update_inventories_from_llm,
        # _modify_inventory_json is internal, typically not exported
    )
    INVENTORY_MODULE_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import inventory module: {e}", exc_info=True)
    INVENTORY_MODULE_AVAILABLE = False
    # Define dummy functions if import fails
    def format_inventory_for_prompt(*args, **kwargs) -> str: return "[Inventory Module Error]"
    async def update_inventories_from_llm(*args, **kwargs) -> bool: return False


# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Define __all__ (Consolidated) ---
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
    "DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT", # Updated template
    "DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT",        # <<< NEW Inventory Update Template
    "format_stateless_refiner_prompt",
    "format_cache_update_prompt",
    "format_final_context_selection_prompt",
    "format_inventory_update_prompt",                # <<< NEW Inventory Update Formatter
    "refine_external_context",
    "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context",
    "clean_context_tags", "generate_rag_query",
    "combine_background_context",
    "process_system_prompt",
    # memory
    "manage_tier1_summarization",
    # cache
    "update_rag_cache",
    "select_final_context",
    # session
    "SessionManager",
    # database
    "initialize_sqlite_tables",
    # -- T1 Summaries --
    "add_tier1_summary", "get_recent_tier1_summaries", "get_tier1_summary_count",
    "get_oldest_tier1_summary", "delete_tier1_summary",
    "get_max_t1_end_index",
    "check_t1_summary_exists",
    # -- RAG Cache --
    "add_or_update_rag_cache", "get_rag_cache",
    # -- Inventory (DB) --
    "initialize_inventory_table",
    "get_character_inventory_data",
    "add_or_update_character_inventory",
    "get_all_inventories_for_session",
    # -- ChromaDB T2 --
    "get_or_create_chroma_collection", "add_to_chroma_collection",
    "query_chroma_collection", "get_chroma_collection_count",
    "CHROMADB_AVAILABLE",
    # orchestration
    "SessionPipeOrchestrator",
    # utils
    "count_tokens",
    "calculate_string_similarity",
    # inventory (Module Functions)
    "format_inventory_for_prompt",
    "update_inventories_from_llm",
    "INVENTORY_MODULE_AVAILABLE", # Flag
]
# [[END MODIFIED __init__.py]]
# --- END OF FILE __init__.py ---