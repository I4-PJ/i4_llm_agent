# i4_llm_agent/cache.py

import logging
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple, Union

# --- Library Dependencies ---
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
        # Fallback: Return current OWI if available, else empty
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
        previous_cache_text = "[Error retrieving previous cache]" # Signal error


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
        query=latest_user_query or "[No query provided]", # Use placeholder if query empty
        template=cache_update_llm_config['prompt_template']
    )

    if not prompt_text or prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}][{session_id}] Failed to format cache update prompt: {prompt_text}. Aborting update.")
        # Fallback: Return previous cache if it was retrieved, else current OWI
        return retrieved_cache if retrieved_cache is not None else (current_owi_context or "")

    # --- Call Cache Update LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"[{caller_info}][{session_id}] Calling LLM for cache update...")
    try:
        success, response_or_error = await llm_call_func(
            api_url=cache_update_llm_config['url'], api_key=cache_update_llm_config['key'],
            payload=payload, temperature=cache_update_llm_config['temp'],
            timeout=120, # Allow reasonable time for update
            caller_info=f"{caller_info}_LLM"
        )
    except Exception as e_call: func_logger.error(f"[{caller_info}][{session_id}] Exception during LLM call: {e_call}", exc_info=True); success = False; response_or_error = "LLM Call Exception"

    # --- Process Result & Update DB ---
    if success and isinstance(response_or_error, str):
        updated_cache_text = response_or_error.strip()
        # Handle case where LLM indicates no relevant context
        if updated_cache_text == "[No relevant background context found]":
             updated_cache_text = "" # Store empty if nothing relevant
             func_logger.info(f"[{caller_info}][{session_id}] Cache update LLM indicated no relevant context found.")
        else:
            func_logger.info(f"[{caller_info}][{session_id}] Cache update LLM call successful (Output len: {len(updated_cache_text)}).")

        # Attempt to save the updated cache
        try:
            save_success = await add_or_update_rag_cache(session_id, updated_cache_text, sqlite_cursor)
            if save_success: func_logger.info(f"[{caller_info}][{session_id}] Successfully saved updated RAG cache to DB.")
            else: func_logger.error(f"[{caller_info}][{session_id}] Failed to save updated RAG cache to DB!") # Log error, but proceed
        except Exception as e_save: func_logger.error(f"[{caller_info}][{session_id}] Exception saving updated RAG cache: {e_save}", exc_info=True) # Log error, proceed

        # Return the text generated by the LLM (the newly updated cache content)
        return updated_cache_text
    else:
        # --- Handle Cache Update Failure ---
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"[{caller_info}][{session_id}] Cache update LLM call failed. Error: '{error_details}'.")
        # Fallback: Return the previously retrieved cache content, if any.
        if retrieved_cache is not None:
            func_logger.warning(f"[{caller_info}][{session_id}] Returning previously retrieved cache content as fallback.")
            return retrieved_cache
        else:
             func_logger.warning(f"[{caller_info}][{session_id}] No previous cache to return. Returning current OWI context as fallback.")
             return current_owi_context or ""


# ==============================================================================
# === Step 2 Orchestration: Select Final Context                             ===
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
    caller_info: str = "ContextSelector"
) -> str:
    """
    Executes Step 2: Takes the updated cache and current OWI context,
    and selects the most relevant snippets for the current turn using LLM.

    Returns:
        str: The selected context snippets for the final prompt (or fallback).
    """
    func_logger = logging.getLogger(__name__ + '.select_final_context')
    func_logger.debug(f"[{caller_info}][SessionID Hidden] Starting Step 2: Final Context Selection...") # Hide session ID here potentially

    # --- Input Checks ---
    if not llm_call_func or not PROMPTING_UTILS_AVAILABLE or not HISTORY_UTILS_AVAILABLE:
        func_logger.error(f"[{caller_info}] Missing core function dependencies. Aborting selection.")
        # Fallback: Return the updated cache text directly if selection fails early
        return updated_cache_text
    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not context_selection_llm_config or not all(k in context_selection_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in (context_selection_llm_config or {})]
        func_logger.error(f"[{caller_info}] Missing context selection LLM config: {missing}. Aborting selection.")
        return updated_cache_text
    if not context_selection_llm_config.get('prompt_template'):
         func_logger.error(f"[{caller_info}] Missing prompt template for context selection. Aborting selection.")
         return updated_cache_text
    if not latest_user_query or not latest_user_query.strip():
         func_logger.warning(f"[{caller_info}] Latest user query is empty. Selection might be less effective.")
         # Proceed, but selection might just return generic info or empty

    # --- Prepare Inputs for LLM ---
    recent_history_str = EMPTY_HISTORY_PLACEHOLDER
    try:
        recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
        if recent_history_list: recent_history_str = format_history_for_llm(recent_history_list)
    except Exception as e_hist: func_logger.error(f"[{caller_info}] Error processing history: {e_hist}"); recent_history_str = "[Error processing history]"

    current_owi_rag_text = current_owi_context if current_owi_context else EMPTY_OWI_CONTEXT_PLACEHOLDER

    # --- Format Prompt for Step 2 ---
    prompt_text = format_final_context_selection_prompt(
        updated_cache=updated_cache_text or "[Cache is empty]", # Handle empty cache from step 1
        current_owi_rag=current_owi_rag_text,
        recent_history_str=recent_history_str,
        query=latest_user_query or "[No query provided]",
        template=context_selection_llm_config['prompt_template']
    )

    if not prompt_text or prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed to format final selection prompt: {prompt_text}. Aborting selection.")
        # Fallback: Return the updated cache text directly
        return updated_cache_text

    # --- Call Context Selection LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling LLM for final context selection...")
    try:
        success, response_or_error = await llm_call_func(
            api_url=context_selection_llm_config['url'], api_key=context_selection_llm_config['key'],
            payload=payload, temperature=context_selection_llm_config['temp'],
            timeout=90, # Selection should be relatively fast
            caller_info=f"{caller_info}_LLM"
        )
    except Exception as e_call: func_logger.error(f"[{caller_info}] Exception during LLM call: {e_call}", exc_info=True); success = False; response_or_error = "LLM Call Exception"

    # --- Process Result ---
    if success and isinstance(response_or_error, str):
        final_selected_context = response_or_error.strip()
        # Handle case where LLM indicates nothing was relevant
        if final_selected_context == "[No relevant background context found for the current query]":
             final_selected_context = "" # Use empty string if nothing selected
             func_logger.info(f"[{caller_info}] Context selection LLM indicated no relevant context found.")
        else:
             func_logger.info(f"[{caller_info}] Context selection LLM call successful (Output len: {len(final_selected_context)}).")
        # Return the selected context snippets
        return final_selected_context
    else:
        # --- Handle Selection Failure ---
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"[{caller_info}] Context selection LLM call failed. Error: '{error_details}'.")
        # Fallback Strategy: Return the full updated cache text (output of Step 1)
        # This is better than returning nothing, providing broader context if selection fails.
        func_logger.warning(f"[{caller_info}] Returning full updated cache content as fallback due to selection failure.")
        return updated_cache_text