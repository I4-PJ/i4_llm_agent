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
NO_RELEVANT_CONTEXT_FLAG = "[No relevant background context found for the current query]" # Specific flag LLM might return
DEBUG_LOG_SUFFIX = ".DEBUG_CTX_SELECT" # Suffix for the debug file

# ==============================================================================
# === SQLite Storage/Retrieval Functions (Keep as generated before)          ===
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
# === Step 1 Orchestration: Update RAG Cache                                 ===
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
    caller_info: str = "CacheUpdater"
) -> str:
    """
    Executes Step 1: Retrieves previous cache, merges with current OWI context using LLM,
    and updates the cache in SQLite.

    Returns:
        str: The updated cache text (or previous cache on failure).
    """
    func_logger = logging.getLogger(__name__ + '.update_rag_cache')
    func_logger.debug(f"[{caller_info}][{session_id}] Starting Step 1: RAG Cache Update...")

    # --- Input Checks ---
    if not llm_call_func or not PROMPTING_UTILS_AVAILABLE or not HISTORY_UTILS_AVAILABLE:
        func_logger.error(f"[{caller_info}][{session_id}] Missing core function dependencies. Aborting update.")
        return current_owi_context or ""
    if not sqlite_cursor:
        func_logger.error(f"[{caller_info}][{session_id}] SQLite cursor not provided. Aborting update.")
        return current_owi_context or ""
    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not cache_update_llm_config or not all(k in cache_update_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in (cache_update_llm_config or {})]
        func_logger.error(f"[{caller_info}][{session_id}] Missing cache update LLM config: {missing}. Aborting update.")
        return current_owi_context or ""
    if not cache_update_llm_config.get('prompt_template'):
         func_logger.error(f"[{caller_info}][{session_id}] Missing prompt template for cache update. Aborting update.")
         return current_owi_context or ""

    # --- Retrieve Previous Cache ---
    previous_cache_text = EMPTY_CACHE_PLACEHOLDER
    retrieved_cache: Optional[str] = None
    try:
        retrieved_cache = await get_rag_cache(session_id, sqlite_cursor)
        if retrieved_cache is not None:
            func_logger.info(f"[{caller_info}][{session_id}] Retrieved previous RAG cache (len: {len(retrieved_cache)}).")
            previous_cache_text = retrieved_cache
        else:
             func_logger.info(f"[{caller_info}][{session_id}] No previous RAG cache found.")
    except Exception as e_get:
        func_logger.error(f"[{caller_info}][{session_id}] Error retrieving RAG cache: {e_get}", exc_info=True)
        previous_cache_text = "[Error retrieving previous cache]"

    # --- Prepare Inputs for LLM ---
    recent_history_str = EMPTY_HISTORY_PLACEHOLDER
    try:
        recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
        if recent_history_list: recent_history_str = format_history_for_llm(recent_history_list)
    except Exception as e_hist: func_logger.error(f"[{caller_info}][{session_id}] Error processing history: {e_hist}"); recent_history_str = "[Error processing history]"

    current_owi_rag_text = current_owi_context if current_owi_context else EMPTY_OWI_CONTEXT_PLACEHOLDER

    # --- Format Prompt for Step 1 ---
    prompt_text = format_cache_update_prompt(
        previous_cache=previous_cache_text,
        current_owi_rag=current_owi_rag_text,
        recent_history_str=recent_history_str,
        query=latest_user_query or "[No query provided]",
        template=cache_update_llm_config['prompt_template']
    )

    if not prompt_text or prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}][{session_id}] Failed to format cache update prompt: {prompt_text}. Aborting update.")
        return retrieved_cache if retrieved_cache is not None else (current_owi_context or "")

    # --- Call Cache Update LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"[{caller_info}][{session_id}] Calling LLM for cache update...")
    try:
        success, response_or_error = await llm_call_func(
            api_url=cache_update_llm_config['url'], api_key=cache_update_llm_config['key'],
            payload=payload, temperature=cache_update_llm_config['temp'],
            timeout=120,
            caller_info=f"{caller_info}_LLM"
        )
    except Exception as e_call: func_logger.error(f"[{caller_info}][{session_id}] Exception during LLM call: {e_call}", exc_info=True); success = False; response_or_error = "LLM Call Exception"

    # --- Process Result & Update DB ---
    if success and isinstance(response_or_error, str):
        updated_cache_text = response_or_error.strip()
        # Handle case where LLM indicates no relevant context (use constant)
        if updated_cache_text == NO_RELEVANT_CONTEXT_FLAG: # Use flag constant
             updated_cache_text = "" # Store empty if nothing relevant
             func_logger.info(f"[{caller_info}][{session_id}] Cache update LLM indicated no relevant context found ('{NO_RELEVANT_CONTEXT_FLAG}').")
        else:
            func_logger.info(f"[{caller_info}][{session_id}] Cache update LLM call successful (Output len: {len(updated_cache_text)}).")

        try:
            save_success = await add_or_update_rag_cache(session_id, updated_cache_text, sqlite_cursor)
            if save_success: func_logger.info(f"[{caller_info}][{session_id}] Successfully saved updated RAG cache to DB.")
            else: func_logger.error(f"[{caller_info}][{session_id}] Failed to save updated RAG cache to DB!")
        except Exception as e_save: func_logger.error(f"[{caller_info}][{session_id}] Exception saving updated RAG cache: {e_save}", exc_info=True)

        return updated_cache_text
    else:
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"[{caller_info}][{session_id}] Cache update LLM call failed. Error: '{error_details}'.")
        if retrieved_cache is not None:
            func_logger.warning(f"[{caller_info}][{session_id}] Returning previously retrieved cache content as fallback.")
            return retrieved_cache
        else:
             func_logger.warning(f"[{caller_info}][{session_id}] No previous cache to return. Returning current OWI context as fallback.")
             return current_owi_context or ""


# ==============================================================================
# === Step 2 Orchestration: Select Final Context (MODIFIED WITH DIRECT FILE LOGGING) ===
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
    # --- NEW ARGUMENT ---
    debug_log_path_getter: Optional[Callable[[str], Optional[str]]] = None
) -> str:
    """
    Executes Step 2: Takes the updated cache and current OWI context,
    and selects the most relevant snippets for the current turn using LLM.
    Adds detailed logging directly to a debug file if path getter is provided.

    Returns:
        str: The selected context snippets for the final prompt (or fallback).
    """
    func_logger = logging.getLogger(__name__ + '.select_final_context')
    log_prefix = f"[{caller_info}]" # For standard logs

    # --- Internal Helper for Direct File Logging ---
    debug_log_file_path: Optional[str] = None
    if debug_log_path_getter and callable(debug_log_path_getter):
        try:
            debug_log_file_path = debug_log_path_getter(DEBUG_LOG_SUFFIX)
            if debug_log_file_path:
                 func_logger.info(f"{log_prefix} Direct debug logging for CTX_SELECT enabled to: {debug_log_file_path}")
            else:
                 func_logger.warning(f"{log_prefix} Debug log path getter provided but returned None for suffix '{DEBUG_LOG_SUFFIX}'. Direct logging disabled.")
        except Exception as e_get_path:
            func_logger.error(f"{log_prefix} Error calling debug_log_path_getter: {e_get_path}. Direct logging disabled.", exc_info=True)
            debug_log_file_path = None

    async def _log_trace(message: str):
        """Appends a timestamped message to the dedicated debug log file."""
        if not debug_log_file_path:
            # Fallback to standard logger if file path isn't set
            func_logger.debug(f"{log_prefix}[CTX_SELECT_TRACE_FALLBACK] {message}")
            return

        ts = datetime.now(timezone.utc).isoformat()
        log_line = f"[{ts}] {message}\n"
        try:
            # Use asyncio.to_thread for file I/O to avoid blocking
            def sync_write():
                try:
                    # Ensure directory exists (important if base log path changes)
                    log_dir = os.path.dirname(debug_log_file_path)
                    if log_dir: # Check if dirname returned something (it might not for just a filename)
                        os.makedirs(log_dir, exist_ok=True)
                    with open(debug_log_file_path, "a", encoding="utf-8") as f:
                        f.write(log_line)
                except Exception as e_write_inner:
                     # Log error to standard logger if file write fails
                     func_logger.error(f"{log_prefix} Failed to write to debug log file '{debug_log_file_path}': {e_write_inner}", exc_info=True)

            await asyncio.to_thread(sync_write)
        except Exception as e_thread:
             func_logger.error(f"{log_prefix} Error scheduling file write for debug log: {e_thread}", exc_info=True)

    # --- Start of Function Logic ---
    func_logger.debug(f"{log_prefix} Starting Step 2: Final Context Selection...")
    await _log_trace("--- select_final_context START ---")

    # --- Log Inputs ---
    await _log_trace(f"INPUT updated_cache_text (len): {len(updated_cache_text)}")
    await _log_trace(f"INPUT current_owi_context (len): {len(current_owi_context) if current_owi_context else 0}")
    await _log_trace(f"INPUT history_messages (count): {len(history_messages)}")
    await _log_trace(f"INPUT latest_user_query (len): {len(latest_user_query)}")
    await _log_trace(f"INPUT history_count: {history_count}")
    safe_config_log = {k: v for k, v in context_selection_llm_config.items() if k != 'key'}
    await _log_trace(f"INPUT context_selection_llm_config (partial): {json.dumps(safe_config_log)}")


    # --- Input Checks ---
    if not llm_call_func or not PROMPTING_UTILS_AVAILABLE or not HISTORY_UTILS_AVAILABLE:
        func_logger.error(f"{log_prefix} Missing core function dependencies. Aborting selection.")
        await _log_trace("EXIT: Missing core function dependencies. Returning fallback (updated_cache_text).")
        return updated_cache_text
    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not context_selection_llm_config or not all(k in context_selection_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in (context_selection_llm_config or {})]
        func_logger.error(f"{log_prefix} Missing context selection LLM config: {missing}. Aborting selection.")
        await _log_trace(f"EXIT: Missing context selection LLM config ({missing}). Returning fallback (updated_cache_text).")
        return updated_cache_text
    if not context_selection_llm_config.get('prompt_template'):
         func_logger.error(f"{log_prefix} Missing prompt template for context selection. Aborting selection.")
         await _log_trace("EXIT: Missing prompt template. Returning fallback (updated_cache_text).")
         return updated_cache_text
    if not latest_user_query or not latest_user_query.strip():
         func_logger.warning(f"{log_prefix} Latest user query is empty. Selection might be less effective.")
         await _log_trace("WARN: Latest user query is empty.")


    # --- Prepare Inputs for LLM ---
    recent_history_str = EMPTY_HISTORY_PLACEHOLDER
    try:
        recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
        if recent_history_list: recent_history_str = format_history_for_llm(recent_history_list)
        await _log_trace(f"PREP recent_history_str (len): {len(recent_history_str)}")
    except Exception as e_hist:
        func_logger.error(f"{log_prefix} Error processing history: {e_hist}"); recent_history_str = "[Error processing history]"
        await _log_trace("PREP recent_history_str set to error placeholder due to exception.")


    current_owi_rag_text = current_owi_context if current_owi_context else EMPTY_OWI_CONTEXT_PLACEHOLDER
    await _log_trace(f"PREP current_owi_rag_text set (len): {len(current_owi_rag_text)}")

    # --- Format Prompt for Step 2 ---
    prompt_text = "[Error generating prompt]" # Default error value
    try:
        prompt_text = format_final_context_selection_prompt(
            updated_cache=updated_cache_text or "[Cache is empty]", # Handle empty cache from step 1
            current_owi_rag=current_owi_rag_text,
            recent_history_str=recent_history_str,
            query=latest_user_query or "[No query provided]",
            template=context_selection_llm_config['prompt_template']
        )
        # Log the prompt being sent to the LLM
        await _log_trace(f"LLM_PROMPT:\n------\n{prompt_text}\n------")

        if not prompt_text or prompt_text.startswith("[Error:"):
            func_logger.error(f"{log_prefix} Failed to format final selection prompt: {prompt_text}. Aborting selection.")
            await _log_trace(f"EXIT: Failed to format prompt ({prompt_text}). Returning fallback (updated_cache_text).")
            return updated_cache_text

    except Exception as e_format:
         func_logger.error(f"{log_prefix} Exception during prompt formatting: {e_format}", exc_info=True)
         await _log_trace(f"EXIT: Exception during prompt formatting ({e_format}). Returning fallback (updated_cache_text).")
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
        await _log_trace(f"LLM_RAW_RESPONSE Success: {success}")
        await _log_trace(f"LLM_RAW_RESPONSE Content:\n------\n{json.dumps(response_or_error, indent=2)}\n------")

    except Exception as e_call:
        func_logger.error(f"{log_prefix} Exception during LLM call: {e_call}", exc_info=True)
        success = False
        response_or_error = f"LLM Call Exception: {e_call}"
        await _log_trace(f"LLM_RAW_RESPONSE Success: {success}")
        await _log_trace(f"LLM_RAW_RESPONSE Content (Exception):\n------\n{response_or_error}\n------")


    # --- Process Result ---
    final_selected_context = updated_cache_text # Default to fallback
    if success and isinstance(response_or_error, str):
        processed_response = response_or_error.strip()
        await _log_trace(f"LLM response processing: Stripped response (len {len(processed_response)}).")

        # Handle case where LLM indicates nothing was relevant (use constant)
        if processed_response == NO_RELEVANT_CONTEXT_FLAG:
             final_selected_context = "" # Use empty string if nothing selected
             func_logger.info(f"{log_prefix} Context selection LLM indicated no relevant context found ('{NO_RELEVANT_CONTEXT_FLAG}'). Setting result to empty string.")
             await _log_trace(f"DECISION: LLM returned '{NO_RELEVANT_CONTEXT_FLAG}'. Setting final_selected_context to empty string.")
        else:
             final_selected_context = processed_response
             func_logger.info(f"{log_prefix} Context selection LLM call successful (Output len: {len(final_selected_context)}).")
             await _log_trace(f"DECISION: LLM success. Setting final_selected_context to processed response (len {len(final_selected_context)}).")

        # Log the final decision before returning
        await _log_trace(f"FINAL RETURN value (len): {len(final_selected_context)}")
        func_logger.debug(f"{log_prefix} select_final_context returning (len: {len(final_selected_context)})")
        return final_selected_context
    else:
        # --- Handle Selection Failure ---
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"{log_prefix} Context selection LLM call failed. Error: '{error_details}'.")
        # Fallback Strategy: Return the full updated cache text (output of Step 1)
        func_logger.warning(f"{log_prefix} Returning full updated cache content (len: {len(updated_cache_text)}) as fallback due to selection failure.")
        await _log_trace(f"DECISION: LLM call failed or invalid response. Using fallback (updated_cache_text, len {len(updated_cache_text)}).")
        # Log the final decision before returning
        await _log_trace(f"FINAL RETURN value (len): {len(updated_cache_text)} (Fallback)")
        func_logger.debug(f"{log_prefix} select_final_context returning fallback (len: {len(updated_cache_text)})")
        return updated_cache_text