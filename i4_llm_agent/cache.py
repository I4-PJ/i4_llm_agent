# i4_llm_agent/cache.py

import logging
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple, Union

# --- Library Dependencies ---
# Import necessary functions from other modules within the library
try:
    from .history import get_recent_turns, format_history_for_llm, DIALOGUE_ROLES
    HISTORY_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import history utils in cache.py: {e}", exc_info=True)
    HISTORY_UTILS_AVAILABLE = False
    # Define fallbacks if needed, though errors are more likely fatal here
    DIALOGUE_ROLES = ["user", "assistant"]
    def get_recent_turns(*args, **kwargs) -> List[Dict]: return []
    def format_history_for_llm(*args, **kwargs) -> str: return ""

try:
    from .prompting import format_curated_refiner_prompt
    PROMPTING_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import prompting utils in cache.py: {e}", exc_info=True)
    PROMPTING_UTILS_AVAILABLE = False
    def format_curated_refiner_prompt(*args, **kwargs) -> str: return "[Error: Prompt Formatter Unavailable]"

# --- Logger ---
logger = logging.getLogger(__name__) # 'i4_llm_agent.cache'

# --- Constants ---
RAG_CACHE_TABLE_NAME = "session_rag_cache"


# ==============================================================================
# === SQLite Storage/Retrieval Functions for RAG Cache                       ===
# ==============================================================================

# --- Sync: Initialize Table ---
def _sync_initialize_rag_cache_table(cursor: sqlite3.Cursor) -> bool:
    """Synchronously creates the RAG cache table if it doesn't exist."""
    func_logger = logging.getLogger(__name__ + '._sync_initialize_rag_cache_table')
    if not cursor:
        func_logger.error("SQLite cursor is not available.")
        return False
    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {RAG_CACHE_TABLE_NAME} (
                session_id TEXT PRIMARY KEY,
                cached_context TEXT NOT NULL,
                last_updated_utc REAL NOT NULL,
                last_updated_iso TEXT
            )
        """)
        # Optional: Add index for faster lookups if needed, though PRIMARY KEY helps
        # cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{RAG_CACHE_TABLE_NAME}_session ON {RAG_CACHE_TABLE_NAME} (session_id)")
        func_logger.debug(f"Table '{RAG_CACHE_TABLE_NAME}' initialized successfully.")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"SQLite error initializing table '{RAG_CACHE_TABLE_NAME}': {e}", exc_info=True)
        return False
    except Exception as e:
        func_logger.error(f"Unexpected error initializing table '{RAG_CACHE_TABLE_NAME}': {e}", exc_info=True)
        return False

# --- Async: Initialize Table ---
async def initialize_rag_cache_table(cursor: sqlite3.Cursor) -> bool:
    """Asynchronously creates the RAG cache table if it doesn't exist."""
    # Consider if the cursor needs to be passed differently for async,
    # but assuming the pipe manages the connection/cursor lifetime.
    return await asyncio.to_thread(_sync_initialize_rag_cache_table, cursor)


# --- Sync: Add or Update Cache ---
def _sync_add_or_update_rag_cache(session_id: str, context_text: str, cursor: sqlite3.Cursor) -> bool:
    """Synchronously adds or updates the cached context for a session."""
    func_logger = logging.getLogger(__name__ + '._sync_add_or_update_rag_cache')
    if not cursor:
        func_logger.error(f"[{session_id}] SQLite cursor unavailable for RAG cache update.")
        return False
    if not session_id or not isinstance(session_id, str):
        func_logger.error("Invalid session_id provided for RAG cache update.")
        return False
    # Allow empty string context? Yes, maybe refinement resulted in empty relevant context.
    if not isinstance(context_text, str):
         func_logger.warning(f"[{session_id}] context_text is not a string (type: {type(context_text)}). Storing as empty string.")
         context_text = ""

    now_utc = datetime.now(timezone.utc)
    timestamp_utc = now_utc.timestamp()
    timestamp_iso = now_utc.isoformat()

    try:
        cursor.execute(f"""
            INSERT OR REPLACE INTO {RAG_CACHE_TABLE_NAME}
            (session_id, cached_context, last_updated_utc, last_updated_iso)
            VALUES (?, ?, ?, ?)
        """, (session_id, context_text, timestamp_utc, timestamp_iso))
        # No commit needed if connection has isolation_level=None (recommended)
        # func_logger.debug(f"[{session_id}] RAG cache updated successfully.")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error updating RAG cache: {e}", exc_info=True)
        return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error updating RAG cache: {e}", exc_info=True)
        return False

# --- Async: Add or Update Cache ---
async def add_or_update_rag_cache(session_id: str, context_text: str, cursor: sqlite3.Cursor) -> bool:
    """Asynchronously adds or updates the cached context for a session."""
    return await asyncio.to_thread(_sync_add_or_update_rag_cache, session_id, context_text, cursor)


# --- Sync: Get Cache ---
def _sync_get_rag_cache(session_id: str, cursor: sqlite3.Cursor) -> Optional[str]:
    """Synchronously retrieves the cached context text for a session."""
    func_logger = logging.getLogger(__name__ + '._sync_get_rag_cache')
    if not cursor:
        func_logger.error(f"[{session_id}] SQLite cursor unavailable for RAG cache retrieval.")
        return None
    if not session_id or not isinstance(session_id, str):
        func_logger.error("Invalid session_id provided for RAG cache retrieval.")
        return None

    try:
        cursor.execute(f"""
            SELECT cached_context FROM {RAG_CACHE_TABLE_NAME} WHERE session_id = ?
        """, (session_id,))
        result = cursor.fetchone()
        if result:
            # func_logger.debug(f"[{session_id}] RAG cache retrieved successfully.")
            return result[0] # Return the cached_context text
        else:
            # func_logger.debug(f"[{session_id}] No RAG cache found for this session.")
            return None # No cache entry found
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error retrieving RAG cache: {e}", exc_info=True)
        return None
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error retrieving RAG cache: {e}", exc_info=True)
        return None

# --- Async: Get Cache ---
async def get_rag_cache(session_id: str, cursor: sqlite3.Cursor) -> Optional[str]:
    """Asynchronously retrieves the cached context text for a session."""
    return await asyncio.to_thread(_sync_get_rag_cache, session_id, cursor)


# ==============================================================================
# === Main Orchestration Function for RAG Cache Refinement & Update          ===
# ==============================================================================

async def refine_and_update_rag_cache(
    session_id: str,
    current_owi_context: Optional[str], # Context from OWI this turn
    history_messages: List[Dict],
    latest_user_query: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # Pipe's LLM wrapper
    sqlite_cursor: sqlite3.Cursor, # Direct cursor access needed by helpers
    refiner_llm_config: Dict[str, Any], # url, key, temp, prompt_template (the revised one)
    history_count: int,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES,
    # Optional: Allow passing different cache funcs, but default to local ones
    _get_cache_func: Callable[[str, sqlite3.Cursor], Coroutine[Any, Any, Optional[str]]] = get_rag_cache,
    _update_cache_func: Callable[[str, str, sqlite3.Cursor], Coroutine[Any, Any, bool]] = add_or_update_rag_cache,
    caller_info: str = "RAGCacheManager"
) -> str:
    """
    Orchestrates the RAG cache refinement process:
    1. Retrieves previous cache.
    2. Formats inputs for the refiner LLM (including previous cache and current OWI context).
    3. Calls the refiner LLM.
    4. Updates the cache with the refined result (on success).
    5. Returns the refined context (or fallback on failure).

    Args:
        session_id: The unique session identifier.
        current_owi_context: The context string provided by OWI RAG for the current turn.
        history_messages: Full message history for the session.
        latest_user_query: The content of the latest user message.
        llm_call_func: The async wrapper provided by the pipe to call LLMs.
        sqlite_cursor: The SQLite cursor instance for DB operations.
        refiner_llm_config: Dict containing 'url', 'key', 'temp', 'prompt_template' for the refiner.
        history_count: Number of recent dialogue turns for refiner context.
        dialogue_only_roles: Roles considered dialogue.
        _get_cache_func: Async function to retrieve the cache (defaults to local get_rag_cache).
        _update_cache_func: Async function to update the cache (defaults to local add_or_update_rag_cache).
        caller_info: Identifier for logging.

    Returns:
        str: The refined context string to be used. On failure, returns a fallback
             (previous cache if available, otherwise the current_owi_context).
    """
    func_logger = logging.getLogger(__name__ + '.refine_and_update_rag_cache')
    func_logger.debug(f"[{caller_info}][{session_id}] Starting RAG cache refinement process...")

    # --- Input Checks ---
    if not latest_user_query or not latest_user_query.strip():
        func_logger.warning(f"[{caller_info}][{session_id}] Skipping refinement: Latest user query is empty.")
        # Return current OWI context if available, otherwise empty string
        return current_owi_context or ""
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        func_logger.error(f"[{caller_info}][{session_id}] Invalid async llm_call_func provided.")
        return current_owi_context or "" # Fallback
    if not sqlite_cursor:
        func_logger.error(f"[{caller_info}][{session_id}] SQLite cursor not provided.")
        return current_owi_context or "" # Fallback
    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not refiner_llm_config or not all(k in refiner_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in (refiner_llm_config or {})]
        func_logger.error(f"[{caller_info}][{session_id}] Missing refiner LLM config keys: {missing}")
        return current_owi_context or "" # Fallback
    if not refiner_llm_config['url'] or not refiner_llm_config['key']:
        func_logger.error(f"[{caller_info}][{session_id}] Missing refiner LLM URL or Key.")
        return current_owi_context or "" # Fallback
    if not HISTORY_UTILS_AVAILABLE or not PROMPTING_UTILS_AVAILABLE:
         func_logger.error(f"[{caller_info}][{session_id}] Missing required library utilities (History/Prompting). Cannot refine.")
         return current_owi_context or "" # Fallback


    # --- Retrieve Previous Cache ---
    cached_context: Optional[str] = None
    try:
        cached_context = await _get_cache_func(session_id, sqlite_cursor)
        if cached_context is None:
             func_logger.info(f"[{caller_info}][{session_id}] No previous RAG cache found.")
             # Use a placeholder for the prompt formatting function
             cached_context_for_prompt = "[No previous cache available]"
        else:
             func_logger.info(f"[{caller_info}][{session_id}] Retrieved previous RAG cache (length: {len(cached_context)}).")
             cached_context_for_prompt = cached_context # Use the actual retrieved text
    except Exception as e_get:
        func_logger.error(f"[{caller_info}][{session_id}] Error retrieving RAG cache: {e_get}", exc_info=True)
        cached_context_for_prompt = "[Error retrieving previous cache]" # Indicate error in prompt


    # --- Prepare Refiner Inputs ---
    # 1. Get Recent History String
    recent_history_str = "[No Recent History]"
    try:
        recent_history_list = get_recent_turns(
            messages=history_messages, count=history_count,
            roles=dialogue_only_roles, exclude_last=True
        )
        if recent_history_list:
            recent_history_str = format_history_for_llm(recent_history_list)
    except Exception as e_hist:
         func_logger.error(f"[{caller_info}][{session_id}] Error processing recent history: {e_hist}", exc_info=True)
         recent_history_str = "[Error processing history]"

    # 2. Format Refiner Prompt (using the dedicated formatter from prompting.py)
    refiner_prompt_text = "[Error: Prompt Formatting Failed]"
    try:
        prompt_template = refiner_llm_config.get('prompt_template') # Get the revised template
        if not prompt_template or not isinstance(prompt_template, str):
             func_logger.error(f"[{caller_info}][{session_id}] Invalid or missing prompt template in refiner config.")
             # Fallback logic: Return previous cache or current OWI context
             return cached_context if cached_context is not None else (current_owi_context or "")

        refiner_prompt_text = format_curated_refiner_prompt(
            current_owi_rag=(current_owi_context or "[No current OWI context provided]"),
            cached_pipe_rag=cached_context_for_prompt, # Use the retrieved text or placeholder
            recent_history_str=recent_history_str,
            query=latest_user_query,
            template=prompt_template
        )
        if "[Error:" in refiner_prompt_text: # Check if formatter returned an error string
             raise ValueError(f"Prompt formatter returned error: {refiner_prompt_text}")
        func_logger.debug(f"[{caller_info}][{session_id}] Formatted refiner prompt successfully.")

    except Exception as e_fmt:
        func_logger.error(f"[{caller_info}][{session_id}] Failed to format curated refiner prompt: {e_fmt}", exc_info=True)
        # Fallback logic: Return previous cache or current OWI context
        return cached_context if cached_context is not None else (current_owi_context or "")

    # 3. Prepare Payload
    refiner_payload = {"contents": [{"parts": [{"text": refiner_prompt_text}]}]}


    # --- Call Refiner LLM via Wrapper ---
    func_logger.info(f"[{caller_info}][{session_id}] Calling Refiner LLM for cache update...")
    try:
        success, response_or_error = await llm_call_func(
            api_url=refiner_llm_config['url'],
            api_key=refiner_llm_config['key'],
            payload=refiner_payload,
            temperature=refiner_llm_config['temp'],
            timeout=120, # Give refinement more time? Or keep standard 90s?
            caller_info=f"{caller_info}_LLMCall",
        )
    except Exception as e_call:
         func_logger.error(f"[{caller_info}][{session_id}] Exception during llm_call_func: {e_call}", exc_info=True)
         success = False
         response_or_error = {"error_type": "CallWrapperException", "message": f"Exception calling LLM wrapper: {type(e_call).__name__}"}


    # --- Process Result & Update Cache ---
    if success and isinstance(response_or_error, str):
        refined_context_output = response_or_error.strip()
        # Check if the LLM explicitly said nothing was relevant
        if refined_context_output == "[No relevant background context found for the current query]":
             refined_context_output = "" # Store empty string in cache if nothing relevant found
             func_logger.info(f"[{caller_info}][{session_id}] Refiner indicated no relevant context found.")
        else:
             func_logger.info(f"[{caller_info}][{session_id}] Refinement successful (Output length: {len(refined_context_output)}).")

        # Update the cache with the new result (even if empty)
        try:
            update_success = await _update_cache_func(session_id, refined_context_output, sqlite_cursor)
            if update_success:
                func_logger.info(f"[{caller_info}][{session_id}] RAG cache updated successfully.")
            else:
                # Log error, but proceed using the refined output for this turn anyway
                func_logger.error(f"[{caller_info}][{session_id}] Failed to update RAG cache in DB, but using refined context for this turn.")
            # Return the refined context for use in the final prompt
            return refined_context_output

        except Exception as e_update:
             func_logger.error(f"[{caller_info}][{session_id}] Exception during RAG cache update: {e_update}", exc_info=True)
             # Failed to update cache, but still return the refined output for this turn
             return refined_context_output

    else:
        # --- Handle Refinement Failure ---
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict):
            error_details = f"Type: {response_or_error.get('error_type', 'Unknown')}, Msg: {response_or_error.get('message', 'N/A')}"

        func_logger.warning(f"[{caller_info}][{session_id}] Refinement LLM call failed. Error: '{error_details}'.")

        # Fallback Strategy: Return the *previous* cache if it exists, otherwise the current OWI context.
        if cached_context is not None:
            func_logger.warning(f"[{caller_info}][{session_id}] Using previously cached context as fallback.")
            return cached_context
        elif current_owi_context:
            func_logger.warning(f"[{caller_info}][{session_id}] No previous cache, using current OWI context as fallback.")
            return current_owi_context
        else:
            func_logger.warning(f"[{caller_info}][{session_id}] No previous cache and no current OWI context available for fallback.")
            return "" # Return empty string if absolutely nothing is available
