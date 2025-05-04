# === MODIFIED BASE FILE: i4_llm_agent/__init__.py (Added Cache Maintainer Exports) ===
# i4_llm_agent/__init__.py
import logging
import asyncio

# --- Core Functionality ---
from .api_client import call_google_llm_api
from .history import (
    format_history_for_llm, get_recent_turns, get_dialogue_history,
    select_turns_for_t0, DIALOGUE_ROLES,
)
# --- Memory Management ---
from .memory import manage_tier1_summarization

# --- <<< Cache Functionality (Maintainer Focus) >>> ---
from .cache import (
    initialize_rag_cache_table, # DB Util
    add_or_update_rag_cache,    # DB Util
    get_rag_cache,              # DB Util
    update_rag_cache_maintainer # <<< ADDED Maintainer Orchestrator
)

# --- Session Management ---
from .session import SessionManager
# --- Database Operations ---
from . import database # Import the module itself
# --- Context Processing ---
from .context_processor import process_context_and_prepare_payload # Now includes maintainer call
# --- Orchestration ---
from .orchestration import SessionPipeOrchestrator
# --- Utilities ---
from .utils import TIKTOKEN_AVAILABLE, count_tokens, calculate_string_similarity

# --- Import Prompting Functions ---
try:
    from .prompting import (
        # Basic Formatting Functions
        format_inventory_update_prompt,
        format_memory_aging_prompt,
        # <<< ADDED Cache Maintainer Formatter >>>
        format_cache_maintainer_prompt,
        # Other Prompting Utilities
        construct_final_llm_payload,
        assemble_tagged_context, extract_tagged_context, # May be legacy
        clean_context_tags, generate_rag_query, combine_background_context,
        process_system_prompt,
    )
    _prompting_funcs_available = True
except ImportError as e:
    logging.getLogger(__name__).critical(f"Failed to import core prompting functions: {e}", exc_info=True)
    _prompting_funcs_available = False
    # Define dummy functions if needed, or let subsequent errors occur
    def format_cache_maintainer_prompt(*args, **kwargs): return "[Formatter Load Error]"


# --- Import Prompting Constants Separately ---
try:
    if not _prompting_funcs_available: raise ImportError("Core prompting functions failed to load")
    from .prompting import (
        # Base Constants
        DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
        DEFAULT_SUMMARIZER_SYSTEM_PROMPT,
        DEFAULT_RAGQ_LLM_PROMPT,
        DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE,
        # <<< ADDED Cache Maintainer Constant >>>
        DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT,
        # <<< ADDED Cache Maintainer Flag Constant >>>
        NO_CACHE_UPDATE_FLAG,
    )
    _prompting_consts_available = True
except ImportError:
    _prompting_consts_available = False
    # Fallbacks
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT = "[Prompting Const Load Error]"
    DEFAULT_SUMMARIZER_SYSTEM_PROMPT = "[Prompting Const Load Error]"
    DEFAULT_RAGQ_LLM_PROMPT = "[Prompting Const Load Error]"
    DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE = "[Prompting Const Load Error]"
    DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT = "[Prompting Const Load Error]" # <<< ADDED fallback
    NO_CACHE_UPDATE_FLAG = "[NO_CACHE_UPDATE]" # <<< ADDED fallback
    logging.getLogger(__name__).error("Failed to import prompting constants.")


# --- Import Inventory Management ---
try:
    from .inventory import (
        format_inventory_for_prompt,
        update_inventories_from_llm,
    )
    INVENTORY_MODULE_AVAILABLE = True
except ImportError as e:
    # ... (fallback unchanged) ...
    logging.getLogger(__name__).error(f"Failed to import inventory module: {e}", exc_info=True)
    INVENTORY_MODULE_AVAILABLE = False
    def format_inventory_for_prompt(*args, **kwargs) -> str: return "[Inventory Module Error]"
    async def update_inventories_from_llm(*args, **kwargs) -> bool: await asyncio.sleep(0); return False


# --- Import Event Hints ---
try:
    from .event_hints import (
        generate_event_hint,
        DEFAULT_EVENT_HINT_TEMPLATE_TEXT # Import constant here
    )
    EVENT_HINTS_AVAILABLE = True
except ImportError as e:
    # ... (fallback unchanged) ...
    logging.getLogger(__name__).error(f"Failed to import event_hints module: {e}", exc_info=True)
    EVENT_HINTS_AVAILABLE = False
    DEFAULT_EVENT_HINT_TEMPLATE_TEXT = "[Default Event Hint Template Load Error]"
    async def generate_event_hint(*args, **kwargs): await asyncio.sleep(0); return None, {}


# --- Import State Assessment ---
try:
    from .state_assessment import (
        update_state_via_full_turn_assessment,
        DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT # Import constant here
    )
    STATE_ASSESSMENT_AVAILABLE = True
except ImportError as e:
    # ... (fallback unchanged) ...
    logging.getLogger(__name__).error(f"Failed to import state_assessment module: {e}", exc_info=True)
    STATE_ASSESSMENT_AVAILABLE = False
    DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT = "[Default Unified State Assessment Template Load Error]"
    async def update_state_via_full_turn_assessment(*args, **kwargs):
        await asyncio.sleep(0)
        previous_world_state = kwargs.get('previous_world_state', {})
        previous_scene_state = kwargs.get('previous_scene_state', {})
        return { "new_day": previous_world_state.get("day", 1), "new_time_of_day": previous_world_state.get("time_of_day", "Morning"), "new_weather": previous_world_state.get("weather", "Clear"), "new_season": previous_world_state.get("season", "Summer"), "new_scene_keywords": previous_scene_state.get("keywords", []), "new_scene_description": previous_scene_state.get("description", ""), "scene_changed_flag": False }

# --- Database Flags/Functions for Export ---
# Base DB functions remain available through the 'database' module import
CHROMADB_AVAILABLE = database.CHROMADB_AVAILABLE
initialize_sqlite_tables = database.initialize_sqlite_tables
add_tier1_summary = database.add_tier1_summary
get_recent_tier1_summaries = database.get_recent_tier1_summaries
get_tier1_summary_count = database.get_tier1_summary_count
get_oldest_tier1_summary = database.get_oldest_tier1_summary
delete_tier1_summary = database.delete_tier1_summary
get_max_t1_end_index = database.get_max_t1_end_index
check_t1_summary_exists = database.check_t1_summary_exists
get_oldest_t1_batch = database.get_oldest_t1_batch
delete_t1_batch = database.delete_t1_batch
# initialize_rag_cache_table = database.initialize_rag_cache_table # Imported from .cache above
# add_or_update_rag_cache = database.add_or_update_rag_cache # Imported from .cache above
# get_rag_cache = database.get_rag_cache # Imported from .cache above
initialize_inventory_table = database.initialize_inventory_table
get_character_inventory_data = database.get_character_inventory_data
add_or_update_character_inventory = database.add_or_update_character_inventory
get_all_inventories_for_session = database.get_all_inventories_for_session
initialize_world_state_table = database.initialize_world_state_table
get_world_state = database.get_world_state
set_world_state = database.set_world_state
initialize_scene_state_table = database.initialize_scene_state_table
get_scene_state = database.get_scene_state
set_scene_state = database.set_scene_state
initialize_aged_summaries_table = database.initialize_aged_summaries_table
add_aged_summary = database.add_aged_summary
get_recent_aged_summaries = database.get_recent_aged_summaries
get_or_create_chroma_collection = database.get_or_create_chroma_collection
add_to_chroma_collection = database.add_to_chroma_collection
query_chroma_collection = database.query_chroma_collection
get_chroma_collection_count = database.get_chroma_collection_count


# --- Configure basic logging for the library ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Define __all__ (Listing names available in this __init__ scope) ---
__all__ = [
    # api_client
    "call_google_llm_api",
    # history
    "format_history_for_llm", "get_recent_turns", "get_dialogue_history",
    "select_turns_for_t0", "DIALOGUE_ROLES",
    # prompting (Functions imported above)
    "format_inventory_update_prompt",
    "format_memory_aging_prompt",
    "format_cache_maintainer_prompt", # <<< ADDED
    "construct_final_llm_payload",
    "assemble_tagged_context", "extract_tagged_context",
    "clean_context_tags", "generate_rag_query", "combine_background_context",
    "process_system_prompt",
    # prompting (Constants imported conditionally)
    "DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT",
    "DEFAULT_SUMMARIZER_SYSTEM_PROMPT",
    "DEFAULT_RAGQ_LLM_PROMPT",
    "DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE",
    "DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT", # <<< ADDED
    "NO_CACHE_UPDATE_FLAG", # <<< ADDED
    # memory
    "manage_tier1_summarization",
    # cache (Imported above)
    "initialize_rag_cache_table",
    "add_or_update_rag_cache",
    "get_rag_cache",
    "update_rag_cache_maintainer", # <<< ADDED
    # session
    "SessionManager",
    # database (Functions assigned above)
    "initialize_sqlite_tables",
    "add_tier1_summary", "get_recent_tier1_summaries", "get_tier1_summary_count",
    "get_oldest_tier1_summary", "delete_tier1_summary",
    "get_max_t1_end_index",
    "check_t1_summary_exists",
    "get_oldest_t1_batch",
    "delete_t1_batch",
    # RAG Cache DB functions already listed under 'cache' section
    "initialize_inventory_table",
    "get_character_inventory_data",
    "add_or_update_character_inventory",
    "get_all_inventories_for_session",
    "initialize_world_state_table",
    "get_world_state",
    "set_world_state",
    "initialize_scene_state_table",
    "get_scene_state",
    "set_scene_state",
    "initialize_aged_summaries_table",
    "add_aged_summary",
    "get_recent_aged_summaries",
    "get_or_create_chroma_collection", "add_to_chroma_collection",
    "query_chroma_collection", "get_chroma_collection_count",
    "CHROMADB_AVAILABLE", # Flag assigned above
    # context_processor
    "process_context_and_prepare_payload", # Updated version
    # orchestration
    "SessionPipeOrchestrator",
    # utils
    "count_tokens",
    "calculate_string_similarity",
    "TIKTOKEN_AVAILABLE", # Flag assigned above
    # inventory (Functions/Flags imported conditionally)
    "format_inventory_for_prompt",
    "update_inventories_from_llm",
    "INVENTORY_MODULE_AVAILABLE",
    # event_hints (Functions/Flags/Constants imported conditionally)
    "generate_event_hint",
    "DEFAULT_EVENT_HINT_TEMPLATE_TEXT",
    "EVENT_HINTS_AVAILABLE",
    # state_assessment (Functions/Flags/Constants imported conditionally)
    "update_state_via_full_turn_assessment",
    "DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT",
    "STATE_ASSESSMENT_AVAILABLE",
]
# === END MODIFIED BASE FILE: i4_llm_agent/__init__.py ===