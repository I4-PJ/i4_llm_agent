# === MODIFIED BASE FILE: i4_llm_agent/cache.py (Added Cache Maintainer Logic) ===
# i4_llm_agent/cache.py

import logging
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple, Union
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
    # Import the NEW Cache Maintainer prompt and formatter
    from .prompting import (
        format_cache_maintainer_prompt, # <<< ADDED
        DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT, # <<< ADDED
        NO_CACHE_UPDATE_FLAG # <<< ADDED Flag Constant
    )
    PROMPTING_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import prompting utils in cache.py: {e}", exc_info=True)
    PROMPTING_UTILS_AVAILABLE = False
    def format_cache_maintainer_prompt(*args, **kwargs) -> str: return "[Error: Prompt Formatter Unavailable]" # <<< ADDED Fallback
    # Define dummy constants if import fails
    DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT = "[Prompting Const Load Error]"
    NO_CACHE_UPDATE_FLAG = "[NO_CACHE_UPDATE]" # Needs a fallback value

# --- Logger ---
logger = logging.getLogger(__name__) # 'i4_llm_agent.cache'

# --- Constants ---
RAG_CACHE_TABLE_NAME = "session_rag_cache"
# Placeholders for empty inputs to the prompt formatter
EMPTY_OWI_CONTEXT_PLACEHOLDER = "[No current OWI context provided]"
EMPTY_HISTORY_PLACEHOLDER = "[No recent history available]"
EMPTY_PREVIOUS_CACHE_PLACEHOLDER = "[Cache is empty or this is the first turn]"
# Debug log suffix for this specific function
DEBUG_LOG_SUFFIX_MAINTAIN = ".DEBUG_CACHE_MAINTAIN" # Suffix for update_rag_cache_maintainer log

# ==============================================================================
# === SQLite Storage/Retrieval Functions (Unchanged from Base)               ===
# ==============================================================================
def _sync_initialize_rag_cache_table(cursor: sqlite3.Cursor) -> bool:
    # ... (implementation unchanged from base) ...
    func_logger = logging.getLogger(__name__ + '._sync_initialize_rag_cache_table')
    if not cursor: func_logger.error("SQLite cursor is not available."); return False
    try:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {RAG_CACHE_TABLE_NAME} (
                session_id TEXT PRIMARY KEY, cached_context TEXT NOT NULL,
                last_updated_utc REAL NOT NULL, last_updated_iso TEXT
            )""")
        func_logger.debug(f"Table '{RAG_CACHE_TABLE_NAME}' checked/initialized successfully.")
        return True
    except sqlite3.Error as e: func_logger.error(f"SQLite error initializing table '{RAG_CACHE_TABLE_NAME}': {e}"); return False
    except Exception as e: func_logger.error(f"Unexpected error initializing table '{RAG_CACHE_TABLE_NAME}': {e}"); return False
async def initialize_rag_cache_table(cursor: sqlite3.Cursor) -> bool:
    # ... (implementation unchanged from base) ...
    return await asyncio.to_thread(_sync_initialize_rag_cache_table, cursor)
def _sync_add_or_update_rag_cache(session_id: str, context_text: str, cursor: sqlite3.Cursor) -> bool:
    # ... (implementation unchanged from base) ...
    func_logger = logging.getLogger(__name__ + '._sync_add_or_update_rag_cache')
    if not cursor: func_logger.error(f"[{session_id}] SQLite cursor unavailable."); return False
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return False
    if not isinstance(context_text, str): func_logger.warning(f"[{session_id}] context_text not a string. Storing empty."); context_text = ""
    now_utc = datetime.now(timezone.utc); timestamp_utc = now_utc.timestamp(); timestamp_iso = now_utc.isoformat()
    try:
        cursor.execute(f"""INSERT OR REPLACE INTO {RAG_CACHE_TABLE_NAME} (session_id, cached_context, last_updated_utc, last_updated_iso) VALUES (?, ?, ?, ?)""",
                       (session_id, context_text, timestamp_utc, timestamp_iso))
        func_logger.debug(f"[{session_id}] Successfully added/updated RAG cache entry.")
        return True
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error updating RAG cache: {e}"); return False
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error updating RAG cache: {e}"); return False
async def add_or_update_rag_cache(session_id: str, context_text: str, cursor: sqlite3.Cursor) -> bool:
    # ... (implementation unchanged from base) ...
    return await asyncio.to_thread(_sync_add_or_update_rag_cache, session_id, context_text, cursor)
def _sync_get_rag_cache(session_id: str, cursor: sqlite3.Cursor) -> Optional[str]:
    # ... (implementation unchanged from base) ...
    func_logger = logging.getLogger(__name__ + '._sync_get_rag_cache')
    if not cursor: func_logger.error(f"[{session_id}] SQLite cursor unavailable."); return None
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return None
    try:
        cursor.execute(f"""SELECT cached_context FROM {RAG_CACHE_TABLE_NAME} WHERE session_id = ?""", (session_id,))
        result = cursor.fetchone()
        if result: func_logger.debug(f"[{session_id}] Found RAG cache entry."); return result[0]
        else: func_logger.debug(f"[{session_id}] No RAG cache entry found."); return None
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error retrieving RAG cache: {e}"); return None
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error retrieving RAG cache: {e}"); return None
async def get_rag_cache(session_id: str, cursor: sqlite3.Cursor) -> Optional[str]:
    # ... (implementation unchanged from base) ...
    return await asyncio.to_thread(_sync_get_rag_cache, session_id, cursor)


# ==============================================================================
# === <<< NEW: Cache Maintainer Orchestration Function >>>                   ===
# ==============================================================================
async def update_rag_cache_maintainer(
    session_id: str,
    current_owi_context: Optional[str],
    history_messages: List[Dict],
    latest_user_query: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    sqlite_cursor: sqlite3.Cursor,
    # Config dict specific to the Cache Maintainer LLM
    cache_maintainer_llm_config: Dict[str, Any], # Expected keys: url, key, temp, prompt_template
    history_count: int,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES,
    caller_info: str = "CacheMaintainer",
    # --- Argument for Debug Logging ---
    debug_log_path_getter: Optional[Callable[[str], Optional[str]]] = None
) -> str:
    """
    Orchestrates the Cache Maintainer logic:
    1. Retrieves the previous cache state.
    2. Calls an LLM using the Cache Maintainer prompt.
    3. Compares the LLM output to the NO_CACHE_UPDATE_FLAG.
    4. If update needed: Saves the LLM's output (the new complete cache) to the DB.
    5. Returns the cache text to be used downstream (either the updated text or the unchanged previous text).

    Returns:
        str: The cache text to use for the current turn's context processing.
             This is the PREVIOUS cache text if no update was needed or if an error occurred.
             This is the NEW cache text if the LLM provided an update.
    """
    func_logger = logging.getLogger(__name__ + '.update_rag_cache_maintainer')
    log_prefix = f"[{caller_info}][{session_id}]" # For standard logs

    # --- Internal Helper for Direct File Logging ---
    debug_log_file_path_maintain: Optional[str] = None
    if debug_log_path_getter and callable(debug_log_path_getter):
        try:
            # Use the specific suffix for this function's log
            debug_log_file_path_maintain = debug_log_path_getter(DEBUG_LOG_SUFFIX_MAINTAIN)
            if debug_log_file_path_maintain:
                 func_logger.info(f"{log_prefix} Direct debug logging for CACHE_MAINTAIN enabled to: {debug_log_file_path_maintain}")
            else:
                 func_logger.warning(f"{log_prefix} Debug log path getter provided but returned None for suffix '{DEBUG_LOG_SUFFIX_MAINTAIN}'. Direct logging disabled.")
        except Exception as e_get_path:
            func_logger.error(f"{log_prefix} Error calling debug_log_path_getter for CACHE_MAINTAIN: {e_get_path}. Direct logging disabled.", exc_info=True)
            debug_log_file_path_maintain = None

    async def _log_trace_maintain(message: str):
        """Appends a timestamped message to the dedicated CACHE_MAINTAIN debug log file."""
        if not debug_log_file_path_maintain:
            func_logger.debug(f"{log_prefix}[CACHE_MAINTAIN_TRACE_FALLBACK] {message}") # Standard logger fallback
            return
        ts = datetime.now(timezone.utc).isoformat()
        log_line = f"[{ts}] {message}\n"
        try:
            def sync_write():
                try:
                    log_dir = os.path.dirname(debug_log_file_path_maintain)
                    if log_dir: os.makedirs(log_dir, exist_ok=True)
                    with open(debug_log_file_path_maintain, "a", encoding="utf-8") as f:
                        f.write(log_line)
                except Exception as e_write_inner:
                     func_logger.error(f"{log_prefix} Failed to write to CACHE_MAINTAIN debug log file '{debug_log_file_path_maintain}': {e_write_inner}", exc_info=True)
            await asyncio.to_thread(sync_write)
        except Exception as e_thread:
             func_logger.error(f"{log_prefix} Error scheduling file write for CACHE_MAINTAIN debug log: {e_thread}", exc_info=True)

    # --- Start of Function Logic ---
    func_logger.debug(f"{log_prefix} Starting Cache Maintainer process...")
    await _log_trace_maintain("--- update_rag_cache_maintainer START ---")

    # --- Log Inputs ---
    await _log_trace_maintain(f"INPUT session_id: {session_id}")
    await _log_trace_maintain(f"INPUT current_owi_context (len): {len(current_owi_context) if current_owi_context else 0}")
    await _log_trace_maintain(f"INPUT history_messages (count): {len(history_messages)}")
    await _log_trace_maintain(f"INPUT latest_user_query (len): {len(latest_user_query)}")
    await _log_trace_maintain(f"INPUT history_count: {history_count}")
    # Log config, hiding the key
    safe_config_log = {k: v for k, v in cache_maintainer_llm_config.items() if k != 'key'}
    # Check if prompt template is default or custom for logging
    is_default_template = cache_maintainer_llm_config.get('prompt_template') == DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT
    safe_config_log['prompt_template_source'] = 'Default' if is_default_template else 'Custom/Override'
    if 'prompt_template' in safe_config_log: del safe_config_log['prompt_template'] # Avoid logging full template
    await _log_trace_maintain(f"INPUT cache_maintainer_llm_config (partial): {safe_config_log}")


    # --- Input Checks ---
    if not llm_call_func or not PROMPTING_UTILS_AVAILABLE or not HISTORY_UTILS_AVAILABLE:
        func_logger.error(f"{log_prefix} Missing core function dependencies (LLM call, Prompting, History). Aborting maintainer.")
        await _log_trace_maintain("EXIT: Missing core function dependencies. Returning empty string.")
        # Fallback to empty string if dependencies missing, as we can't reliably get previous cache
        return ""
    if not sqlite_cursor:
        func_logger.error(f"{log_prefix} SQLite cursor not provided. Aborting maintainer.")
        await _log_trace_maintain("EXIT: SQLite cursor not provided. Returning empty string.")
        return ""
    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not cache_maintainer_llm_config or not all(k in cache_maintainer_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in (cache_maintainer_llm_config or {})]
        func_logger.error(f"{log_prefix} Missing cache maintainer LLM config: {missing}. Aborting maintainer.")
        await _log_trace_maintain(f"EXIT: Missing cache maintainer LLM config ({missing}). Returning empty string.")
        return ""
    # Check if the necessary formatting function is available
    if not callable(format_cache_maintainer_prompt):
         func_logger.error(f"{log_prefix} Cache maintainer formatting function unavailable. Aborting maintainer.")
         await _log_trace_maintain("EXIT: Cache maintainer formatter unavailable. Returning empty string.")
         return ""


    # --- Retrieve Previous Cache Text ---
    previous_cache_text: str = "" # Default to empty string if not found or error
    try:
        retrieved_cache = await get_rag_cache(session_id, sqlite_cursor)
        if retrieved_cache is not None: # Explicit check for None vs empty string
            previous_cache_text = retrieved_cache
            func_logger.info(f"{log_prefix} Successfully retrieved previous cache text (len: {len(previous_cache_text)}).")
            await _log_trace_maintain(f"DB_READ: Success. Previous cache len: {len(previous_cache_text)}")
        else:
            func_logger.info(f"{log_prefix} No previous cache found in DB for this session.")
            await _log_trace_maintain("DB_READ: No previous cache entry found.")
            previous_cache_text = "" # Ensure it's an empty string if None was returned
    except Exception as e_get_cache:
        func_logger.error(f"{log_prefix} Exception retrieving previous cache: {e_get_cache}", exc_info=True)
        await _log_trace_maintain(f"DB_READ: Exception retrieving cache: {e_get_cache}. Using empty string fallback.")
        previous_cache_text = "" # Use empty string on error

    # Fallback return value in case of errors before LLM call returns
    # We return the state we read (or empty if read failed)
    cache_to_use_downstream = previous_cache_text

    # --- Prepare Inputs for LLM ---
    recent_history_str = EMPTY_HISTORY_PLACEHOLDER
    try:
        recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
        if recent_history_list:
            recent_history_str = format_history_for_llm(recent_history_list)
        await _log_trace_maintain(f"PREP recent_history_str (len): {len(recent_history_str)}")
    except Exception as e_hist:
        func_logger.error(f"{log_prefix} Error processing history: {e_hist}"); recent_history_str = "[Error processing history]"
        await _log_trace_maintain("PREP recent_history_str set to error placeholder due to exception.")

    # Prepare OWI context, using placeholder if None or empty
    current_owi_rag_text = current_owi_context if current_owi_context else EMPTY_OWI_CONTEXT_PLACEHOLDER
    await _log_trace_maintain(f"PREP current_owi_rag_text set (len): {len(current_owi_rag_text)}")

    # Prepare previous cache text for prompt, using placeholder if it was originally empty
    previous_cache_for_prompt = previous_cache_text if previous_cache_text else EMPTY_PREVIOUS_CACHE_PLACEHOLDER
    await _log_trace_maintain(f"PREP previous_cache_for_prompt set (len): {len(previous_cache_for_prompt)}")


    # --- Format Prompt for Cache Maintainer LLM ---
    prompt_text = "[Error generating prompt]"
    try:
        # Call the new formatting function
        prompt_text = format_cache_maintainer_prompt(
            query=latest_user_query or "[No query provided]",
            recent_history_str=recent_history_str,
            previous_cache_text=previous_cache_for_prompt,
            current_owi_context=current_owi_rag_text,
            template=cache_maintainer_llm_config['prompt_template']
        )
        await _log_trace_maintain(f"LLM_PROMPT (Maintainer):\n------\n{prompt_text}\n------")
        if not prompt_text or prompt_text.startswith("[Error:"):
            raise ValueError(f"Prompt formatting failed: {prompt_text}")
    except Exception as e_format:
        func_logger.error(f"{log_prefix} Failed/Exception formatting maintainer prompt: {e_format}", exc_info=True)
        await _log_trace_maintain(f"EXIT: Failed/Exception formatting maintainer prompt ({e_format}). Returning previous cache state.")
        # Return the unchanged previous cache on formatting error
        return cache_to_use_downstream

    # --- Call Cache Maintainer LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"{log_prefix} Calling LLM for cache maintenance decision...")
    await _log_trace_maintain("LLM_CALL: Attempting LLM call for cache maintenance.")
    success = False
    response_or_error = "Initialization Error"
    try:
        # Use the specific config dict for the maintainer
        success, response_or_error = await llm_call_func(
            api_url=cache_maintainer_llm_config['url'], api_key=cache_maintainer_llm_config['key'],
            payload=payload, temperature=cache_maintainer_llm_config['temp'],
            timeout=120, # Allow reasonable time for comparison/synthesis
            caller_info=f"{caller_info}_LLM_Maintain"
        )
        # Logging raw response
        await _log_trace_maintain(f"LLM_RAW_RESPONSE Success: {success}")
        log_content = str(response_or_error)
        await _log_trace_maintain(f"LLM_RAW_RESPONSE Content:\n------\n{log_content}\n------")

    except Exception as e_call:
        func_logger.error(f"{log_prefix} Exception during LLM call for cache maintenance: {e_call}", exc_info=True)
        success = False; response_or_error = f"LLM Call Exception: {e_call}"
        await _log_trace_maintain(f"LLM_RAW_RESPONSE Success: {success}"); await _log_trace_maintain(f"LLM_RAW_RESPONSE Content (Exception):\n------\n{response_or_error}\n------")

    # --- Process LLM Output and Update DB ---
    if success and isinstance(response_or_error, str):
        llm_output_str = response_or_error # Keep original case and whitespace for exact flag check

        # Check for the specific "no update" flag
        if llm_output_str == NO_CACHE_UPDATE_FLAG:
            func_logger.info(f"{log_prefix} LLM indicated no cache update needed ('{NO_CACHE_UPDATE_FLAG}'). Using previous cache.")
            await _log_trace_maintain(f"DECISION: LLM returned '{NO_CACHE_UPDATE_FLAG}'. Using previous cache state (len: {len(previous_cache_text)}). No DB write needed.")
            # No DB update needed, return the retrieved previous cache text
            cache_to_use_downstream = previous_cache_text
        else:
            # LLM returned updated cache text
            updated_cache_text = llm_output_str.strip()
            func_logger.info(f"{log_prefix} LLM provided updated cache text (len: {len(updated_cache_text)}). Saving to DB.")
            await _log_trace_maintain(f"DECISION: LLM returned new cache content (stripped len: {len(updated_cache_text)}). Will attempt DB write.")

            # Save the updated cache text to the database
            save_success = False
            try:
                save_success = await add_or_update_rag_cache(session_id, updated_cache_text, sqlite_cursor)
                if save_success:
                    func_logger.info(f"{log_prefix} Successfully saved updated cache to DB.")
                    await _log_trace_maintain(f"DB_WRITE: Success. Saved updated cache (len: {len(updated_cache_text)}).")
                    # Use the newly updated cache text downstream
                    cache_to_use_downstream = updated_cache_text
                else:
                    func_logger.error(f"{log_prefix} Failed to save updated cache state to DB!")
                    await _log_trace_maintain(f"DB_WRITE: FAILED to save updated cache state (len: {len(updated_cache_text)}). Using previous cache as fallback.")
                    # Fallback to previous cache if save fails
                    cache_to_use_downstream = previous_cache_text
            except Exception as e_save:
                func_logger.error(f"{log_prefix} Exception saving updated cache state: {e_save}", exc_info=True)
                await _log_trace_maintain(f"DB_WRITE: Exception saving updated cache state: {e_save}. Using previous cache as fallback.")
                # Fallback to previous cache on exception
                cache_to_use_downstream = previous_cache_text

    # --- If LLM call failed ---
    elif not success:
        error_details = str(response_or_error);
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"{log_prefix} Cache Maintainer LLM call failed. Error: '{error_details}'. Using previous cache state.")
        await _log_trace_maintain(f"DECISION: LLM call failed. Using previous cache state (len: {len(previous_cache_text)}). No DB write needed.")
        # Use the previously retrieved cache text on failure
        cache_to_use_downstream = previous_cache_text

    # --- Final Return ---
    await _log_trace_maintain(f"FINAL RETURN value (len): {len(cache_to_use_downstream)}")
    await _log_trace_maintain("--- update_rag_cache_maintainer END ---")
    func_logger.debug(f"{log_prefix} Cache Maintainer process finished. Returning cache text (len: {len(cache_to_use_downstream)}).")
    return cache_to_use_downstream

# === END MODIFIED BASE FILE: i4_llm_agent/cache.py ===