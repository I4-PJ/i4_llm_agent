# === START OF FILE i4_llm_agent/__init__.py ===

# [[START MODIFIED __init__.py - Add Summarizer/RAGQ Prompt Exports]]
# i4_llm_agent/__init__.py
import logging
import asyncio

# --- Core Functionality ---
from .api_client import call_google_llm_api # Existing API client
# History Utils (Existing)
from .history import (
    format_history_for_llm, get_recent_turns, get_dialogue_history,
    select_turns_for_t0, DIALOGUE_ROLES,
)
# Prompting Utils (Existing)
from .prompting import (
    # Default Templates
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
    # === NEW EXPORTS ===
    DEFAULT_SUMMARIZER_SYSTEM_PROMPT,
    DEFAULT_RAGQ_LLM_PROMPT,
    # === END NEW ===
    # Formatting Functions
    format_stateless_refiner_prompt,
    format_cache_update_prompt,
    format_final_context_selection_prompt,
    format_inventory_update_prompt,
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

# --- Database Operations (Modified: Added Scene State Table/Functions) ---
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
    # SQLite Inventory (Existing)
    initialize_inventory_table,
    get_character_inventory_data,
    add_or_update_character_inventory,
    get_all_inventories_for_session,
    # SQLite World State (Existing)
    initialize_world_state_table,
    get_world_state,
    set_world_state,
    # SQLite Scene State (Added)
    initialize_scene_state_table,
    get_scene_state,
    set_scene_state,
    # ChromaDB T2 (Existing)
    get_or_create_chroma_collection, add_to_chroma_collection,
    query_chroma_collection, get_chroma_collection_count,
    CHROMADB_AVAILABLE, # Export flag
)

# --- Orchestration (Existing) ---
from .orchestration import SessionPipeOrchestrator

# --- Utilities (Existing) ---
from .utils import count_tokens, calculate_string_similarity

# --- Inventory Management (Existing Module Import) ---
try:
    from .inventory import (
        format_inventory_for_prompt,
        update_inventories_from_llm,
    )
    INVENTORY_MODULE_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import inventory module: {e}", exc_info=True)
    INVENTORY_MODULE_AVAILABLE = False
    def format_inventory_for_prompt(*args, **kwargs) -> str: return "[Inventory Module Error]"
    async def update_inventories_from_llm(*args, **kwargs) -> bool: await asyncio.sleep(0); return False

# --- Event Hints (Existing Module Import) ---
try:
    # Now also import generate_event_hint function itself if needed elsewhere
    from .event_hints import generate_event_hint, DEFAULT_EVENT_HINT_TEMPLATE_TEXT
    EVENT_HINTS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import event_hints module: {e}", exc_info=True)
    EVENT_HINTS_AVAILABLE = False
    DEFAULT_EVENT_HINT_TEMPLATE_TEXT = "[Default Event Hint Template Load Error]"
    async def generate_event_hint(*args, **kwargs): await asyncio.sleep(0); return None, {} # Return tuple expected by orchestrator

# --- World State Parser (Existing Import) ---
try:
    from .world_state_parser import (
        parse_world_state_with_llm,
        confirm_weather_change_with_llm,
        DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT
    )
    WORLD_STATE_PARSER_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import world_state_parser module: {e}", exc_info=True)
    WORLD_STATE_PARSER_AVAILABLE = False
    DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT = "[Default World State Parse Template Load Error]"
    async def parse_world_state_with_llm(*args, **kwargs): await asyncio.sleep(0); return {}
    async def confirm_weather_change_with_llm(*args, **kwargs): await asyncio.sleep(0); return False

# --- Scene Generator (NEW Import) ---
try:
    from .scene_generator import (
        assess_and_generate_scene,
        DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT
    )
    SCENE_GENERATOR_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import scene_generator module: {e}", exc_info=True)
    SCENE_GENERATOR_AVAILABLE = False
    DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT = "[Default Scene Assessment Template Load Error]"
    async def assess_and_generate_scene(*args, **kwargs):
        # Fallback needs to return expected structure (dict), potentially previous state if passed
        # For simplicity, return empty default if module fails load
        await asyncio.sleep(0)
        return {"keywords": [], "description": ""}

# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Define __all__ (Consolidated & Updated) ---
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
    "DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT",
    "DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT", # Export scene prompt
    # === NEW EXPORTS ===
    "DEFAULT_SUMMARIZER_SYSTEM_PROMPT",
    "DEFAULT_RAGQ_LLM_PROMPT",
    # === END NEW ===
    "format_stateless_refiner_prompt",
    "format_cache_update_prompt",
    "format_final_context_selection_prompt",
    "format_inventory_update_prompt",
    "refine_external_context", # Stateless orchestrator
    "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context",
    "clean_context_tags", "generate_rag_query", "combine_background_context",
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
    # -- World State (DB) --
    "initialize_world_state_table",
    "get_world_state",
    "set_world_state",
    # -- Scene State (DB) --
    "initialize_scene_state_table",
    "get_scene_state",
    "set_scene_state",
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
    # event_hints
    "generate_event_hint", # Export function too
    "DEFAULT_EVENT_HINT_TEMPLATE_TEXT",
    "EVENT_HINTS_AVAILABLE", # Flag
    # world_state_parser
    "parse_world_state_with_llm", # Export functions too
    "confirm_weather_change_with_llm",
    "DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT",
    "WORLD_STATE_PARSER_AVAILABLE", # Flag
    # scene_generator
    "assess_and_generate_scene",
    "DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT",
    "SCENE_GENERATOR_AVAILABLE", # Flag
]
# [[END MODIFIED __init__.py - Add Summarizer/RAGQ Prompt Exports]]

# === END OF FILE i4_llm_agent/__init__.py ===