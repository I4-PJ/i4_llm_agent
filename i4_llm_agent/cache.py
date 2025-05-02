# === START MODIFIED FILE: i4_llm_agent/cache.py (with Constant Toggle) ===
# i4_llm_agent/cache.py

import logging
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple, Union
import json
import os # Import os for path joining

## --- Library Dependencies ---
try:
    from .history import get_recent_turns, format_history_for_llm, DIALOGUE_ROLES
    HISTORY_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import history utils in cache.py: {e}", exc_info=True)
    HISTORY_UTILS_AVAILABLE = False
    DIALOGUE_ROLES = ["user", "assistant"]
    def get_recent_turns(*args, **kwargs) -> List[Dict]: return []
    def format_history_for_llm(*args, **kwargs) -> str: return ""

try:
    # Import the NEW prompt formatting functions
    from .prompting import format_cache_update_prompt, format_final_context_selection_prompt
    PROMPTING_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import prompting utils in cache.py: {e}", exc_info=True)
    PROMPTING_UTILS_AVAILABLE = False
    def format_cache_update_prompt(*args, **kwargs) -> str: return "[Error: Prompt Formatter Unavailable]"
    def format_final_context_selection_prompt(*args, **kwargs) -> str: return "[Error: Prompt Formatter Unavailable]"


# --- Logger ---
logger = logging.getLogger(__name__) # 'i4_llm_agent.cache'

# --- Constants ---
RAG_CACHE_TABLE_NAME = "session_rag_cache"
EMPTY_CACHE_PLACEHOLDER = "[No previous cache available]"
EMPTY_OWI_CONTEXT_PLACEHOLDER = "[No current OWI context provided]"
EMPTY_HISTORY_PLACEHOLDER = "[No recent history available]"
NO_RELEVANT_CONTEXT_FLAG = "[No relevant background context found]" # Specific flag LLM might return
NO_CACHE_UPDATE_FLAG = "[NO_CACHE_UPDATE]" # Flag for cache update step
DEBUG_LOG_SUFFIX_SELECT = ".DEBUG_CTX_SELECT" # Suffix for select_final_context log
DEBUG_LOG_SUFFIX_UPDATE = ".DEBUG_CACHE_UPDATE" # Suffix for update_rag_cache log

# <<< NEW CONSTANT TOGGLE >>>
# Set this to False to exclude the raw 'current_owi_context' from the input
# to the second LLM call (select_final_context). This forces the selection
# LLM to rely solely on the updated_cache from Step 1 and the history/query.
# Set to True to keep the original behavior (include both cache and OWI).
# TODO: Refactor this into a proper configuration valve passed down from orchestrator.
INCLUDE_OWI_IN_SELECTION = False # <<< Set to False to exclude OWI from Step 2 LLM
OWI_EXCLUDED_PLACEHOLDER = "[OWI Input Excluded by Constant]" # Placeholder used if excluded

# ==============================================================================
# === SQLite Storage/Retrieval Functions (Unchanged)                         ===
# ==============================================================================
def _sync_initialize_rag_cache_table(cursor: sqlite3.Cursor) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_initialize_rag_cache_table')
    if not cursor: func_logger.error("SQLite cursor is not available."); return False
    try:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {RAG_CACHE_TABLE_NAME} (
                session_id TEXT PRIMARY KEY, cached_context TEXT NOT NULL,
                last_updated_utc REAL NOT NULL, last_updated_iso TEXT
            )""")
        func_logger.debug(f"Table '{RAG_CACHE_TABLE_NAME}' initialized successfully.")
        return True
    except sqlite3.Error as e: func_logger.error(f"SQLite error initializing table '{RAG_CACHE_TABLE_NAME}': {e}"); return False
    except Exception as e: func_logger.error(f"Unexpected error initializing table '{RAG_CACHE_TABLE_NAME}': {e}"); return False
async def initialize_rag_cache_table(cursor: sqlite3.Cursor) -> bool:
    return await asyncio.to_thread(_sync_initialize_rag_cache_table, cursor)
def _sync_add_or_update_rag_cache(session_id: str, context_text: str, cursor: sqlite3.Cursor) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_add_or_update_rag_cache')
    if not cursor: func_logger.error(f"[{session_id}] SQLite cursor unavailable."); return False
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return False
    if not isinstance(context_text, str): func_logger.warning(f"[{session_id}] context_text not a string. Storing empty."); context_text = ""
    now_utc = datetime.now(timezone.utc); timestamp_utc = now_utc.timestamp(); timestamp_iso = now_utc.isoformat()
    try:
        cursor.execute(f"""INSERT OR REPLACE INTO {RAG_CACHE_TABLE_NAME} (session_id, cached_context, last_updated_utc, last_updated_iso) VALUES (?, ?, ?, ?)""",
                       (session_id, context_text, timestamp_utc, timestamp_iso))
        return True
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error updating RAG cache: {e}"); return False
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error updating RAG cache: {e}"); return False
async def add_or_update_rag_cache(session_id: str, context_text: str, cursor: sqlite3.Cursor) -> bool:
    return await asyncio.to_thread(_sync_add_or_update_rag_cache, session_id, context_text, cursor)
def _sync_get_rag_cache(session_id: str, cursor: sqlite3.Cursor) -> Optional[str]:
    func_logger = logging.getLogger(__name__ + '._sync_get_rag_cache')
    if not cursor: func_logger.error(f"[{session_id}] SQLite cursor unavailable."); return None
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return None
    try:
        cursor.execute(f"""SELECT cached_context FROM {RAG_CACHE_TABLE_NAME} WHERE session_id = ?""", (session_id,))
        result = cursor.fetchone()
        if result: return result[0]
        else: return None
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error retrieving RAG cache: {e}"); return None
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error retrieving RAG cache: {e}"); return None
async def get_rag_cache(session_id: str, cursor: sqlite3.Cursor) -> Optional[str]:
    return await asyncio.to_thread(_sync_get_rag_cache, session_id, cursor)


# ==============================================================================
# === Step 1 Orchestration: Update RAG Cache (Logging Added)                 ===
# ==============================================================================
async def update_rag_cache(
    session_id: str,
    current_owi_context: Optional[str],
    history_messages: List[Dict],
    latest_user_query: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    sqlite_cursor: sqlite3.Cursor,
    cache_update_llm_config: Dict[str, Any], # url, key, temp, prompt_template
    history_count: int,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES,
    caller_info: str = "CacheUpdater",
    # --- Argument for Debug Logging ---
    debug_log_path_getter: Optional[Callable[[str], Optional[str]]] = None
) -> str:
    """
    Executes Step 1: Retrieves previous cache, merges with current OWI context using LLM,
    and updates the cache in SQLite. Adds detailed logging.

    Returns:
        str: The updated cache text (or previous cache on failure).
    """
    func_logger = logging.getLogger(__name__ + '.update_rag_cache')
    log_prefix = f"[{caller_info}][{session_id}]" # For standard logs

    # --- Internal Helper for Direct File Logging ---
    debug_log_file_path_update: Optional[str] = None
    if debug_log_path_getter and callable(debug_log_path_getter):
        try:
            # Use the NEW suffix for this function's log
            debug_log_file_path_update = debug_log_path_getter(DEBUG_LOG_SUFFIX_UPDATE)
            if debug_log_file_path_update:
                 func_logger.info(f"{log_prefix} Direct debug logging for CACHE_UPDATE enabled to: {debug_log_file_path_update}")
            else:
                 func_logger.warning(f"{log_prefix} Debug log path getter provided but returned None for suffix '{DEBUG_LOG_SUFFIX_UPDATE}'. Direct logging disabled.")
        except Exception as e_get_path:
            func_logger.error(f"{log_prefix} Error calling debug_log_path_getter for CACHE_UPDATE: {e_get_path}. Direct logging disabled.", exc_info=True)
            debug_log_file_path_update = None

    async def _log_trace_update(message: str):
        """Appends a timestamped message to the dedicated CACHE_UPDATE debug log file."""
        if not debug_log_file_path_update:
            func_logger.debug(f"{log_prefix}[CACHE_UPDATE_TRACE_FALLBACK] {message}") # Standard logger fallback
            return

        ts = datetime.now(timezone.utc).isoformat()
        log_line = f"[{ts}] {message}\n"
        try:
            def sync_write():
                try:
                    log_dir = os.path.dirname(debug_log_file_path_update)
                    if log_dir: os.makedirs(log_dir, exist_ok=True)
                    with open(debug_log_file_path_update, "a", encoding="utf-8") as f:
                        f.write(log_line)
                except Exception as e_write_inner:
                     func_logger.error(f"{log_prefix} Failed to write to CACHE_UPDATE debug log file '{debug_log_file_path_update}': {e_write_inner}", exc_info=True)

            await asyncio.to_thread(sync_write)
        except Exception as e_thread:
             func_logger.error(f"{log_prefix} Error scheduling file write for CACHE_UPDATE debug log: {e_thread}", exc_info=True)

    # --- Start of Function Logic ---
    func_logger.debug(f"{log_prefix} Starting Step 1: RAG Cache Update...")
    await _log_trace_update("--- update_rag_cache START ---")

    # --- Log Inputs ---
    await _log_trace_update(f"INPUT session_id: {session_id}")
    await _log_trace_update(f"INPUT current_owi_context (len): {len(current_owi_context) if current_owi_context else 0}")
    await _log_trace_update(f"INPUT history_messages (count): {len(history_messages)}")
    await _log_trace_update(f"INPUT latest_user_query (len): {len(latest_user_query)}")
    await _log_trace_update(f"INPUT history_count: {history_count}")
    safe_config_log = {k: v for k, v in cache_update_llm_config.items() if k != 'key'}
    await _log_trace_update(f"INPUT cache_update_llm_config (partial): {json.dumps(safe_config_log)}")

    # --- Input Checks ---
    if not llm_call_func or not PROMPTING_UTILS_AVAILABLE or not HISTORY_UTILS_AVAILABLE:
        func_logger.error(f"{log_prefix} Missing core function dependencies. Aborting update.")
        await _log_trace_update("EXIT: Missing core function dependencies. Returning current_owi_context or empty.")
        return current_owi_context or ""
    if not sqlite_cursor:
        func_logger.error(f"{log_prefix} SQLite cursor not provided. Aborting update.")
        await _log_trace_update("EXIT: SQLite cursor not provided. Returning current_owi_context or empty.")
        return current_owi_context or ""
    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not cache_update_llm_config or not all(k in cache_update_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in (cache_update_llm_config or {})]
        func_logger.error(f"{log_prefix} Missing cache update LLM config: {missing}. Aborting update.")
        await _log_trace_update(f"EXIT: Missing cache update LLM config ({missing}). Returning current_owi_context or empty.")
        return current_owi_context or ""
    if not cache_update_llm_config.get('prompt_template'):
         func_logger.error(f"{log_prefix} Missing prompt template for cache update. Aborting update.")
         await _log_trace_update("EXIT: Missing prompt template. Returning current_owi_context or empty.")
         return current_owi_context or ""

    # --- Retrieve Previous Cache ---
    previous_cache_text = EMPTY_CACHE_PLACEHOLDER
    retrieved_cache: Optional[str] = None
    try:
        retrieved_cache = await get_rag_cache(session_id, sqlite_cursor)
        if retrieved_cache is not None:
            func_logger.info(f"{log_prefix} Retrieved previous RAG cache (len: {len(retrieved_cache)}).")
            previous_cache_text = retrieved_cache
            await _log_trace_update(f"DB_READ: Successfully retrieved previous cache (len: {len(retrieved_cache)}).")
        else:
             func_logger.info(f"{log_prefix} No previous RAG cache found.")
             await _log_trace_update("DB_READ: No previous cache found in DB.")
    except Exception as e_get:
        func_logger.error(f"{log_prefix} Error retrieving RAG cache: {e_get}", exc_info=True)
        previous_cache_text = "[Error retrieving previous cache]"
        await _log_trace_update(f"DB_READ: Error retrieving cache: {e_get}")

    # --- Prepare Inputs for LLM ---
    recent_history_str = EMPTY_HISTORY_PLACEHOLDER
    try:
        recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
        if recent_history_list: recent_history_str = format_history_for_llm(recent_history_list)
        await _log_trace_update(f"PREP recent_history_str (len): {len(recent_history_str)}")
    except Exception as e_hist:
        func_logger.error(f"{log_prefix} Error processing history: {e_hist}"); recent_history_str = "[Error processing history]"
        await _log_trace_update("PREP recent_history_str set to error placeholder due to exception.")

    current_owi_rag_text = current_owi_context if current_owi_context else EMPTY_OWI_CONTEXT_PLACEHOLDER
    await _log_trace_update(f"PREP current_owi_rag_text set (len): {len(current_owi_rag_text)}")

    # --- Format Prompt for Step 1 ---
    prompt_text = "[Error generating prompt]" # Default error value
    try:
        prompt_text = format_cache_update_prompt(
            previous_cache=previous_cache_text,
            current_owi_rag=current_owi_rag_text,
            recent_history_str=recent_history_str,
            query=latest_user_query or "[No query provided]",
            template=cache_update_llm_config['prompt_template']
        )
        await _log_trace_update(f"LLM_PROMPT:\n------\n{prompt_text}\n------")

        if not prompt_text or prompt_text.startswith("[Error:"):
            func_logger.error(f"{log_prefix} Failed to format cache update prompt: {prompt_text}. Aborting update.")
            await _log_trace_update(f"EXIT: Failed to format prompt ({prompt_text}). Returning fallback (retrieved_cache or current_owi_context).")
            # Return previous cache if available, otherwise current OWI
            return retrieved_cache if retrieved_cache is not None else (current_owi_context or "")

    except Exception as e_format:
        func_logger.error(f"{log_prefix} Exception during prompt formatting: {e_format}", exc_info=True)
        await _log_trace_update(f"EXIT: Exception during prompt formatting ({e_format}). Returning fallback (retrieved_cache or current_owi_context).")
        return retrieved_cache if retrieved_cache is not None else (current_owi_context or "")


    # --- Call Cache Update LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"{log_prefix} Calling LLM for cache update...")
    await _log_trace_update("LLM_CALL: Attempting LLM call for cache update.")
    success = False
    response_or_error = "Initialization Error"
    try:
        success, response_or_error = await llm_call_func(
            api_url=cache_update_llm_config['url'], api_key=cache_update_llm_config['key'],
            payload=payload, temperature=cache_update_llm_config['temp'],
            timeout=120, # Keep timeout potentially longer for merging/updating
            caller_info=f"{caller_info}_LLM"
        )
        await _log_trace_update(f"LLM_RAW_RESPONSE Success: {success}")
        await _log_trace_update(f"LLM_RAW_RESPONSE Content:\n------\n{json.dumps(response_or_error, indent=2)}\n------")

    except Exception as e_call:
        func_logger.error(f"{log_prefix} Exception during LLM call: {e_call}", exc_info=True)
        success = False
        response_or_error = f"LLM Call Exception: {e_call}"
        await _log_trace_update(f"LLM_RAW_RESPONSE Success: {success}")
        await _log_trace_update(f"LLM_RAW_RESPONSE Content (Exception):\n------\n{response_or_error}\n------")

    # --- Process Result & Update DB ---
    updated_cache_text_final = retrieved_cache if retrieved_cache is not None else (current_owi_context or "") # Fallback value
    if success and isinstance(response_or_error, str):
        processed_response = response_or_error.strip()
        await _log_trace_update(f"LLM response processing: Stripped response (len {len(processed_response)}).")

        # Handle specific flags
        if processed_response == NO_RELEVANT_CONTEXT_FLAG:
             updated_cache_text_final = "" # Store empty if nothing relevant
             func_logger.info(f"{log_prefix} Cache update LLM indicated no relevant context found ('{NO_RELEVANT_CONTEXT_FLAG}'). Storing empty cache.")
             await _log_trace_update(f"DECISION: LLM returned '{NO_RELEVANT_CONTEXT_FLAG}'. Setting cache update result to empty string.")
        elif processed_response == NO_CACHE_UPDATE_FLAG:
             # No change needed, keep the previously retrieved cache content
             updated_cache_text_final = retrieved_cache if retrieved_cache is not None else "" # Use empty if no previous cache existed
             func_logger.info(f"{log_prefix} Cache update LLM indicated no update needed ('{NO_CACHE_UPDATE_FLAG}'). Keeping previous cache content (or empty).")
             await _log_trace_update(f"DECISION: LLM returned '{NO_CACHE_UPDATE_FLAG}'. Setting cache update result to previous cache content (len: {len(updated_cache_text_final)}).")
        else:
             updated_cache_text_final = processed_response # Use the LLM's generated update
             func_logger.info(f"{log_prefix} Cache update LLM call successful (Output len: {len(updated_cache_text_final)}).")
             await _log_trace_update(f"DECISION: LLM success. Setting cache update result to processed response (len {len(updated_cache_text_final)}).")

        # --- Save the determined final state to DB ---
        save_success = False
        try:
            # Save the 'updated_cache_text_final' which reflects the decision based on flags
            save_success = await add_or_update_rag_cache(session_id, updated_cache_text_final, sqlite_cursor)
            if save_success:
                func_logger.info(f"{log_prefix} Successfully saved updated RAG cache to DB.")
                await _log_trace_update(f"DB_WRITE: Successfully saved cache state (len: {len(updated_cache_text_final)}).")
            else:
                func_logger.error(f"{log_prefix} Failed to save updated RAG cache to DB!")
                await _log_trace_update(f"DB_WRITE: FAILED to save cache state (len: {len(updated_cache_text_final)}).")
        except Exception as e_save:
            func_logger.error(f"{log_prefix} Exception saving updated RAG cache: {e_save}", exc_info=True)
            await _log_trace_update(f"DB_WRITE: Exception saving cache state: {e_save}")

        # Return the text that was *determined* to be the correct state (might be old cache, empty, or new)
        await _log_trace_update(f"FINAL RETURN value (len): {len(updated_cache_text_final)}")
        await _log_trace_update("--- update_rag_cache END ---")
        return updated_cache_text_final

    else:
        # --- Handle LLM Call Failure ---
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"{log_prefix} Cache update LLM call failed. Error: '{error_details}'.")
        await _log_trace_update(f"DECISION: LLM call failed or invalid response. Using fallback (retrieved_cache or current_owi_context).")

        # Fallback: Return the previously retrieved cache content if available, otherwise the current OWI context
        fallback_return = retrieved_cache if retrieved_cache is not None else (current_owi_context or "")
        func_logger.warning(f"{log_prefix} Returning previous cache or OWI context as fallback (len: {len(fallback_return)}).")
        await _log_trace_update(f"FINAL RETURN value (len): {len(fallback_return)} (Fallback)")
        await _log_trace_update("--- update_rag_cache END ---")
        return fallback_return


# ==============================================================================
# === Step 2 Orchestration: Select Final Context (MODIFIED WITH CONSTANT TOGGLE) ===
# ==============================================================================
async def select_final_context(
    updated_cache_text: str, # Result from Step 1
    current_owi_context: Optional[str],
    history_messages: List[Dict],
    latest_user_query: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    context_selection_llm_config: Dict[str, Any], # url, key, temp, prompt_template
    history_count: int,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES,
    caller_info: str = "ContextSelector",
    # --- Argument for Debug Logging ---
    debug_log_path_getter: Optional[Callable[[str], Optional[str]]] = None
    # Parameter include_owi_in_selection removed - using constant instead
) -> str:
    """
    Executes Step 2: Takes the updated cache and optionally current OWI context,
    and selects the most relevant snippets for the current turn using LLM.
    Adds detailed logging directly to a debug file if path getter is provided.
    Uses the INCLUDE_OWI_IN_SELECTION constant to control OWI inclusion.

    Returns:
        str: The selected context snippets for the final prompt (or fallback).
    """
    func_logger = logging.getLogger(__name__ + '.select_final_context')
    # Using simplified log prefix as requested by user
    log_prefix = f"[{caller_info}][CACHE]" # For standard logs

    # --- Internal Helper for Direct File Logging ---
    debug_log_file_path_select: Optional[str] = None
    if debug_log_path_getter and callable(debug_log_path_getter):
        try:
            # Use the specific suffix for this function's log
            debug_log_file_path_select = debug_log_path_getter(DEBUG_LOG_SUFFIX_SELECT)
            if debug_log_file_path_select:
                 func_logger.info(f"{log_prefix} Direct debug logging for CTX_SELECT enabled to: {debug_log_file_path_select}")
            else:
                 func_logger.warning(f"{log_prefix} Debug log path getter provided but returned None for suffix '{DEBUG_LOG_SUFFIX_SELECT}'. Direct logging disabled.")
        except Exception as e_get_path:
            func_logger.error(f"{log_prefix} Error calling debug_log_path_getter for CTX_SELECT: {e_get_path}. Direct logging disabled.", exc_info=True)
            debug_log_file_path_select = None

    async def _log_trace_select(message: str):
        """Appends a timestamped message to the dedicated CTX_SELECT debug log file."""
        if not debug_log_file_path_select:
            func_logger.debug(f"{log_prefix}[CTX_SELECT_TRACE_FALLBACK] {message}") # Standard logger fallback
            return

        ts = datetime.now(timezone.utc).isoformat()
        log_line = f"[{ts}] {message}\n"
        try:
            def sync_write():
                try:
                    log_dir = os.path.dirname(debug_log_file_path_select)
                    if log_dir: os.makedirs(log_dir, exist_ok=True)
                    with open(debug_log_file_path_select, "a", encoding="utf-8") as f:
                        f.write(log_line)
                except Exception as e_write_inner:
                     func_logger.error(f"{log_prefix} Failed to write to CTX_SELECT debug log file '{debug_log_file_path_select}': {e_write_inner}", exc_info=True)

            await asyncio.to_thread(sync_write)
        except Exception as e_thread:
             func_logger.error(f"{log_prefix} Error scheduling file write for CTX_SELECT debug log: {e_thread}", exc_info=True)

    # --- Start of Function Logic ---
    func_logger.debug(f"{log_prefix} Starting Step 2: Final Context Selection...")
    await _log_trace_select("--- select_final_context START ---")

    # --- Log Inputs ---
    await _log_trace_select(f"INPUT updated_cache_text (len): {len(updated_cache_text)}")
    await _log_trace_select(f"INPUT current_owi_context (len): {len(current_owi_context) if current_owi_context else 0}")
    await _log_trace_select(f"INPUT history_messages (count): {len(history_messages)}")
    await _log_trace_select(f"INPUT latest_user_query (len): {len(latest_user_query)}")
    await _log_trace_select(f"INPUT history_count: {history_count}")
    safe_config_log = {k: v for k, v in context_selection_llm_config.items() if k != 'key'}
    await _log_trace_select(f"INPUT context_selection_llm_config (partial): {json.dumps(safe_config_log)}")
    # Log the state of the constant toggle
    await _log_trace_select(f"CONFIG CONSTANT INCLUDE_OWI_IN_SELECTION: {INCLUDE_OWI_IN_SELECTION}")


    # --- Input Checks ---
    if not llm_call_func or not PROMPTING_UTILS_AVAILABLE or not HISTORY_UTILS_AVAILABLE:
        func_logger.error(f"{log_prefix} Missing core function dependencies. Aborting selection.")
        await _log_trace_select("EXIT: Missing core function dependencies. Returning fallback (updated_cache_text).")
        return updated_cache_text
    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not context_selection_llm_config or not all(k in context_selection_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in (context_selection_llm_config or {})]
        func_logger.error(f"{log_prefix} Missing context selection LLM config: {missing}. Aborting selection.")
        await _log_trace_select(f"EXIT: Missing context selection LLM config ({missing}). Returning fallback (updated_cache_text).")
        return updated_cache_text
    if not context_selection_llm_config.get('prompt_template'):
         func_logger.error(f"{log_prefix} Missing prompt template for context selection. Aborting selection.")
         await _log_trace_select("EXIT: Missing prompt template. Returning fallback (updated_cache_text).")
         return updated_cache_text
    if not latest_user_query or not latest_user_query.strip():
         func_logger.warning(f"{log_prefix} Latest user query is empty. Selection might be less effective.")
         await _log_trace_select("WARN: Latest user query is empty.")


    # --- Prepare Inputs for LLM ---
    recent_history_str = EMPTY_HISTORY_PLACEHOLDER
    try:
        recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
        if recent_history_list: recent_history_str = format_history_for_llm(recent_history_list)
        await _log_trace_select(f"PREP recent_history_str (len): {len(recent_history_str)}")
    except Exception as e_hist:
        func_logger.error(f"{log_prefix} Error processing history: {e_hist}"); recent_history_str = "[Error processing history]"
        await _log_trace_select("PREP recent_history_str set to error placeholder due to exception.")

    # --- MODIFICATION START: Use constant to determine OWI inclusion ---
    current_owi_rag_text_for_prompt = ""
    if INCLUDE_OWI_IN_SELECTION:
        current_owi_rag_text_for_prompt = current_owi_context if current_owi_context else EMPTY_OWI_CONTEXT_PLACEHOLDER
        func_logger.debug(f"{log_prefix} Including OWI context in selection prompt based on constant.")
        await _log_trace_select(f"PREP current_owi_rag_text_for_prompt set to actual OWI content (len): {len(current_owi_rag_text_for_prompt)}")
    else:
        current_owi_rag_text_for_prompt = OWI_EXCLUDED_PLACEHOLDER
        func_logger.debug(f"{log_prefix} Excluding OWI context from selection prompt based on constant.")
        await _log_trace_select(f"PREP current_owi_rag_text_for_prompt set to placeholder: '{OWI_EXCLUDED_PLACEHOLDER}'")
    # --- MODIFICATION END ---

    # --- Format Prompt for Step 2 ---
    prompt_text = "[Error generating prompt]" # Default error value
    try:
        prompt_text = format_final_context_selection_prompt(
            updated_cache=updated_cache_text or "[Cache is empty]", # Handle empty cache from step 1
            current_owi_rag=current_owi_rag_text_for_prompt, # Use the potentially modified variable
            recent_history_str=recent_history_str,
            query=latest_user_query or "[No query provided]",
            template=context_selection_llm_config['prompt_template']
        )
        # Log the prompt being sent to the LLM
        await _log_trace_select(f"LLM_PROMPT:\n------\n{prompt_text}\n------")

        if not prompt_text or prompt_text.startswith("[Error:"):
            func_logger.error(f"{log_prefix} Failed to format final selection prompt: {prompt_text}. Aborting selection.")
            await _log_trace_select(f"EXIT: Failed to format prompt ({prompt_text}). Returning fallback (updated_cache_text).")
            return updated_cache_text

    except Exception as e_format:
         func_logger.error(f"{log_prefix} Exception during prompt formatting: {e_format}", exc_info=True)
         await _log_trace_select(f"EXIT: Exception during prompt formatting ({e_format}). Returning fallback (updated_cache_text).")
         return updated_cache_text


    # --- Call Context Selection LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"{log_prefix} Calling LLM for final context selection...")
    success = False
    response_or_error = "Initialization Error"
    try:
        success, response_or_error = await llm_call_func(
            api_url=context_selection_llm_config['url'], api_key=context_selection_llm_config['key'],
            payload=payload, temperature=context_selection_llm_config['temp'],
            timeout=90, # Selection should be relatively fast
            caller_info=f"{caller_info}_LLM"
        )
        # Log the RAW response received from the LLM
        await _log_trace_select(f"LLM_RAW_RESPONSE Success: {success}")
        await _log_trace_select(f"LLM_RAW_RESPONSE Content:\n------\n{json.dumps(response_or_error, indent=2)}\n------")

    except Exception as e_call:
        func_logger.error(f"{log_prefix} Exception during LLM call: {e_call}", exc_info=True)
        success = False
        response_or_error = f"LLM Call Exception: {e_call}"
        await _log_trace_select(f"LLM_RAW_RESPONSE Success: {success}")
        await _log_trace_select(f"LLM_RAW_RESPONSE Content (Exception):\n------\n{response_or_error}\n------")


    # --- Process Result ---
    final_selected_context = updated_cache_text # Default to fallback
    if success and isinstance(response_or_error, str):
        processed_response = response_or_error.strip()
        await _log_trace_select(f"LLM response processing: Stripped response (len {len(processed_response)}).")

        # Handle case where LLM indicates nothing was relevant (use constant)
        if processed_response == NO_RELEVANT_CONTEXT_FLAG:
             final_selected_context = "" # Use empty string if nothing selected
             func_logger.info(f"{log_prefix} Context selection LLM indicated no relevant context found ('{NO_RELEVANT_CONTEXT_FLAG}'). Setting result to empty string.")
             await _log_trace_select(f"DECISION: LLM returned '{NO_RELEVANT_CONTEXT_FLAG}'. Setting final_selected_context to empty string.")
        else:
             final_selected_context = processed_response
             func_logger.info(f"{log_prefix} Context selection LLM call successful (Output len: {len(final_selected_context)}).")
             await _log_trace_select(f"DECISION: LLM success. Setting final_selected_context to processed response (len {len(final_selected_context)}).")

        # Log the final decision before returning
        await _log_trace_select(f"FINAL RETURN value (len): {len(final_selected_context)}")
        await _log_trace_select("--- select_final_context END ---")
        func_logger.debug(f"{log_prefix} select_final_context returning (len: {len(final_selected_context)})")
        return final_selected_context
    else:
        # --- Handle Selection Failure ---
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"{log_prefix} Context selection LLM call failed. Error: '{error_details}'.")
        # Fallback Strategy: Return the full updated cache text (output of Step 1)
        func_logger.warning(f"{log_prefix} Returning full updated cache content (len: {len(updated_cache_text)}) as fallback due to selection failure.")
        await _log_trace_select(f"DECISION: LLM call failed or invalid response. Using fallback (updated_cache_text, len {len(updated_cache_text)}).")
        # Log the final decision before returning
        await _log_trace_select(f"FINAL RETURN value (len): {len(updated_cache_text)} (Fallback)")
        await _log_trace_select("--- select_final_context END ---")
        func_logger.debug(f"{log_prefix} select_final_context returning fallback (len: {len(updated_cache_text)})")
        return updated_cache_text

# === END MODIFIED FILE: i4_llm_agent/cache.py (with Constant Toggle) ===