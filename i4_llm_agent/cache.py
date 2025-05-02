# === START MODIFIED FILE: i4_llm_agent/cache.py (Delta Generation Logic) ===
# i4_llm_agent/cache.py

import logging
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple, Union, OrderedDict
import json
import os # Import os for path joining
import re # Import re for parsing headings
from collections import OrderedDict # To preserve section order

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
NO_RELEVANT_CONTEXT_FLAG = "[No relevant background context found]" # Specific flag LLM might return (Less relevant now?)
# NO_CACHE_UPDATE_FLAG = "[NO_CACHE_UPDATE]" # No longer used, empty delta means no update
DEBUG_LOG_SUFFIX_SELECT = ".DEBUG_CTX_SELECT" # Suffix for select_final_context log
DEBUG_LOG_SUFFIX_UPDATE = ".DEBUG_CACHE_UPDATE" # Suffix for update_rag_cache log

# Constant toggle for Step 2 (select_final_context)
INCLUDE_OWI_IN_SELECTION = False
OWI_EXCLUDED_PLACEHOLDER = "[OWI Input Excluded by Constant]"

# Regex to identify heading lines (e.g., "# Heading", "## Subheading", "=== Heading ===")
# Assumes heading starts the line, allows leading/trailing whitespace on the line itself
HEADING_REGEX = re.compile(r"^\s*([#=]{1,5}\s+.+?)\s*$", re.MULTILINE)

# ==============================================================================
# === Internal Helper Functions for Delta Logic                            ===
# ==============================================================================

def _parse_cache_into_sections(cache_text: str) -> OrderedDict[str, str]:
    """Parses cache text into an ordered dictionary mapping heading -> content."""
    sections = OrderedDict()
    if not cache_text:
        return sections

    last_match_end = 0
    first_heading_found = False

    for match in HEADING_REGEX.finditer(cache_text):
        heading_start, heading_end = match.span()
        heading_text = match.group(1).strip() # Get the captured heading text, stripped

        # Content is the text between the end of the last match/start and the start of this heading
        content_start = last_match_end
        content = cache_text[content_start:heading_start].strip()

        if not first_heading_found:
            # Handle text before the very first heading
            if content:
                sections["[PREAMBLE]"] = content # Use a special key for pre-heading text
            first_heading_found = True
        else:
            # Find the *previous* heading added to sections to assign this content block
            if sections:
                 # Get the last added heading
                 last_heading = next(reversed(sections))
                 # Assign the content to the *previous* heading
                 sections[last_heading] = content

        # Add the current heading with empty content for now (will be filled by next iteration or end)
        sections[heading_text] = ""
        last_match_end = heading_end

    # Handle content after the last heading
    if first_heading_found: # Only if at least one heading was found
        final_content = cache_text[last_match_end:].strip()
        if final_content:
            last_heading = next(reversed(sections))
            sections[last_heading] = final_content
    elif not first_heading_found and cache_text:
         # If no headings were found at all, treat the whole text as preamble
         sections["[PREAMBLE]"] = cache_text.strip()


    # Clean up empty preamble if it exists and is empty
    if "[PREAMBLE]" in sections and not sections["[PREAMBLE]"]:
        del sections["[PREAMBLE]"]

    return sections

def _reconstruct_cache_from_sections(sections: OrderedDict[str, str]) -> str:
    """Reconstructs the cache text from the ordered dictionary of sections."""
    parts = []
    for heading, content in sections.items():
        if heading == "[PREAMBLE]":
            if content: # Only add preamble if it has content
                parts.append(content)
        else:
            parts.append(heading) # Add the heading line
            if content: # Add content only if it exists
                parts.append(content)
    # Join sections with double newline for readability
    return "\n\n".join(parts).strip()


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
# === Step 1 Orchestration: Update RAG Cache (Delta Generation Logic)        ===
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
    Executes Step 1 (Delta Generation):
    1. Retrieves previous cache text.
    2. Calls LLM to generate a JSON delta describing changes based on OWI context.
    3. Parses the previous cache into sections.
    4. Applies the JSON delta (add/modify/delete) to the sections.
    5. Reconstructs the updated cache text.
    6. Saves the updated cache text to SQLite.
    Includes detailed logging for the delta process.

    Returns:
        str: The updated cache text (or previous cache text on failure).
    """
    func_logger = logging.getLogger(__name__ + '.update_rag_cache')
    log_prefix = f"[{caller_info}][{session_id}]" # For standard logs

    # --- Internal Helper for Direct File Logging ---
    debug_log_file_path_update: Optional[str] = None
    if debug_log_path_getter and callable(debug_log_path_getter):
        try:
            debug_log_file_path_update = debug_log_path_getter(DEBUG_LOG_SUFFIX_UPDATE)
            if debug_log_file_path_update:
                 func_logger.info(f"{log_prefix} Direct debug logging for CACHE_UPDATE (Delta) enabled to: {debug_log_file_path_update}")
            else:
                 func_logger.warning(f"{log_prefix} Debug log path getter provided but returned None for suffix '{DEBUG_LOG_SUFFIX_UPDATE}'. Direct logging disabled.")
        except Exception as e_get_path:
            func_logger.error(f"{log_prefix} Error calling debug_log_path_getter for CACHE_UPDATE (Delta): {e_get_path}. Direct logging disabled.", exc_info=True)
            debug_log_file_path_update = None

    async def _log_trace_update(message: str):
        """Appends a timestamped message to the dedicated CACHE_UPDATE debug log file."""
        if not debug_log_file_path_update:
            func_logger.debug(f"{log_prefix}[CACHE_UPDATE_DELTA_TRACE_FALLBACK] {message}") # Standard logger fallback
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
                     func_logger.error(f"{log_prefix} Failed to write to CACHE_UPDATE (Delta) debug log file '{debug_log_file_path_update}': {e_write_inner}", exc_info=True)
            await asyncio.to_thread(sync_write)
        except Exception as e_thread:
             func_logger.error(f"{log_prefix} Error scheduling file write for CACHE_UPDATE (Delta) debug log: {e_thread}", exc_info=True)

    # --- Start of Function Logic ---
    func_logger.debug(f"{log_prefix} Starting Step 1: RAG Cache Update (Delta Generation)...")
    await _log_trace_update("--- update_rag_cache START (Delta Generation) ---")

    # --- Log Inputs ---
    await _log_trace_update(f"INPUT session_id: {session_id}")
    await _log_trace_update(f"INPUT current_owi_context (len): {len(current_owi_context) if current_owi_context else 0}")
    await _log_trace_update(f"INPUT history_messages (count): {len(history_messages)}")
    await _log_trace_update(f"INPUT latest_user_query (len): {len(latest_user_query)}")
    await _log_trace_update(f"INPUT history_count: {history_count}")
    safe_config_log = {k: v for k, v in cache_update_llm_config.items() if k != 'key'}
    await _log_trace_update(f"INPUT cache_update_llm_config (partial): {json.dumps(safe_config_log)}")

    # --- Input Checks (Same as before) ---
    if not llm_call_func or not PROMPTING_UTILS_AVAILABLE or not HISTORY_UTILS_AVAILABLE:
        func_logger.error(f"{log_prefix} Missing core function dependencies. Aborting update.")
        await _log_trace_update("EXIT: Missing core function dependencies. Returning current_owi_context or empty.")
        return current_owi_context or ""
    # ... (other input checks remain the same) ...
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
    previous_cache_text = "" # Default to empty string if not found or error
    retrieved_cache: Optional[str] = None
    try:
        retrieved_cache = await get_rag_cache(session_id, sqlite_cursor)
        if retrieved_cache is not None:
            func_logger.info(f"{log_prefix} Retrieved previous RAG cache (len: {len(retrieved_cache)}).")
            previous_cache_text = retrieved_cache # Assign retrieved text
            await _log_trace_update(f"DB_READ: Successfully retrieved previous cache (len: {len(retrieved_cache)}).")
        else:
             func_logger.info(f"{log_prefix} No previous RAG cache found. Starting with empty cache.")
             await _log_trace_update("DB_READ: No previous cache found in DB. Base cache is empty.")
    except Exception as e_get:
        func_logger.error(f"{log_prefix} Error retrieving RAG cache: {e_get}", exc_info=True)
        await _log_trace_update(f"DB_READ: Error retrieving cache: {e_get}. Base cache is empty.")
        previous_cache_text = "" # Ensure it's empty on error

    # Fallback variable in case of errors later
    fallback_cache_text = previous_cache_text

    # --- Prepare Inputs for LLM (Same as before) ---
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

    # --- Format Prompt for Step 1 (Using Delta Prompt Template) ---
    prompt_text = "[Error generating prompt]" # Default error value
    try:
        # Pass the *actual* previous cache text to the formatter
        prompt_text = format_cache_update_prompt(
            previous_cache=previous_cache_text if previous_cache_text else EMPTY_CACHE_PLACEHOLDER,
            current_owi_rag=current_owi_rag_text,
            recent_history_str=recent_history_str,
            query=latest_user_query or "[No query provided]",
            template=cache_update_llm_config['prompt_template'] # This should now be the Delta prompt
        )
        await _log_trace_update(f"LLM_PROMPT (Delta):\n------\n{prompt_text}\n------")

        if not prompt_text or prompt_text.startswith("[Error:"):
            func_logger.error(f"{log_prefix} Failed to format cache update delta prompt: {prompt_text}. Aborting update.")
            await _log_trace_update(f"EXIT: Failed to format delta prompt ({prompt_text}). Returning fallback (previous cache).")
            return fallback_cache_text

    except Exception as e_format:
        func_logger.error(f"{log_prefix} Exception during delta prompt formatting: {e_format}", exc_info=True)
        await _log_trace_update(f"EXIT: Exception during delta prompt formatting ({e_format}). Returning fallback (previous cache).")
        return fallback_cache_text

    # --- Call Cache Update LLM (Expecting JSON Delta) ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"{log_prefix} Calling LLM for cache update delta...")
    await _log_trace_update("LLM_CALL: Attempting LLM call for cache update delta.")
    success = False
    response_or_error = "Initialization Error"
    try:
        success, response_or_error = await llm_call_func(
            api_url=cache_update_llm_config['url'], api_key=cache_update_llm_config['key'],
            payload=payload, temperature=cache_update_llm_config['temp'],
            timeout=120, # Keep timeout potentially longer
            caller_info=f"{caller_info}_LLM_Delta"
        )
        await _log_trace_update(f"LLM_RAW_RESPONSE Success: {success}")
        # Log the response assuming it might be JSON
        log_content = str(response_or_error)
        if success and isinstance(response_or_error, str):
             try: log_content = json.dumps(json.loads(response_or_error), indent=2) # Pretty print if valid JSON
             except json.JSONDecodeError: pass # Keep as string if not valid JSON
        elif not success and isinstance(response_or_error, dict):
             log_content = json.dumps(response_or_error, indent=2)

        await _log_trace_update(f"LLM_RAW_RESPONSE Content:\n------\n{log_content}\n------")

    except Exception as e_call:
        func_logger.error(f"{log_prefix} Exception during LLM call for delta: {e_call}", exc_info=True)
        success = False
        response_or_error = f"LLM Call Exception: {e_call}"
        await _log_trace_update(f"LLM_RAW_RESPONSE Success: {success}")
        await _log_trace_update(f"LLM_RAW_RESPONSE Content (Exception):\n------\n{response_or_error}\n------")


    # --- Process JSON Delta Response & Apply Changes ---
    updated_cache_text_final = fallback_cache_text # Start with fallback
    if success and isinstance(response_or_error, str):
        llm_output_str = response_or_error.strip()
        parsed_delta = None
        delta_changes = []
        validation_passed = False

        # 1. Parse JSON
        try:
            # Handle potential markdown fences
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_output_str, re.IGNORECASE)
            if match: json_string_to_parse = match.group(1).strip()
            else: json_string_to_parse = llm_output_str

            if not json_string_to_parse:
                func_logger.warning(f"{log_prefix} LLM returned empty string after stripping fences.")
                parsed_delta = {"changes": []} # Treat as no changes
            else:
                 parsed_delta = json.loads(json_string_to_parse)
            await _log_trace_update(f"JSON_PARSE: Successfully parsed LLM response.")

        except json.JSONDecodeError as e:
            func_logger.error(f"{log_prefix} Failed to decode JSON delta from LLM response: {e}. Response: {llm_output_str[:500]}...")
            await _log_trace_update(f"JSON_PARSE: FAILED JSONDecodeError: {e}. Response: {llm_output_str[:500]}...")
            # Keep fallback, proceed to save/return

        # 2. Validate Delta Structure
        if parsed_delta is not None:
            if isinstance(parsed_delta, dict) and "changes" in parsed_delta and isinstance(parsed_delta["changes"], list):
                delta_changes = parsed_delta["changes"]
                validation_passed = True # Initial assumption
                for i, change in enumerate(delta_changes):
                    if not isinstance(change, dict):
                        func_logger.error(f"{log_prefix} Invalid change item at index {i}: Not a dictionary. Item: {change}")
                        await _log_trace_update(f"DELTA_VALIDATION: FAILED - Item at index {i} is not a dict.")
                        validation_passed = False; break
                    action = change.get("action")
                    heading = change.get("heading")
                    content = change.get("content")
                    if action not in ["add", "modify", "delete"]:
                        func_logger.error(f"{log_prefix} Invalid action '{action}' in change item at index {i}.")
                        await _log_trace_update(f"DELTA_VALIDATION: FAILED - Invalid action '{action}' at index {i}.")
                        validation_passed = False; break
                    if not heading or not isinstance(heading, str):
                        func_logger.error(f"{log_prefix} Missing or invalid 'heading' in change item at index {i}.")
                        await _log_trace_update(f"DELTA_VALIDATION: FAILED - Missing/invalid heading at index {i}.")
                        validation_passed = False; break
                    if action in ["add", "modify"] and (content is None or not isinstance(content, str)):
                        # Allow empty string content, but not None or non-string
                        if content is None:
                             func_logger.error(f"{log_prefix} Missing 'content' for '{action}' action at index {i}.")
                             await _log_trace_update(f"DELTA_VALIDATION: FAILED - Missing content for '{action}' at index {i}.")
                             validation_passed = False; break
                        elif not isinstance(content, str):
                             func_logger.error(f"{log_prefix} Invalid 'content' type (must be string) for '{action}' action at index {i}.")
                             await _log_trace_update(f"DELTA_VALIDATION: FAILED - Invalid content type for '{action}' at index {i}.")
                             validation_passed = False; break
                if validation_passed:
                    func_logger.info(f"{log_prefix} LLM delta JSON structure validated ({len(delta_changes)} changes).")
                    await _log_trace_update(f"DELTA_VALIDATION: PASSED ({len(delta_changes)} changes).")
            else:
                func_logger.error(f"{log_prefix} Invalid delta structure from LLM. Expected {{'changes': [...]}}. Got: {parsed_delta}")
                await _log_trace_update(f"DELTA_VALIDATION: FAILED - Invalid top-level structure.")
                # Keep fallback

        # 3. Apply Changes if Validated
        if validation_passed:
            if not delta_changes:
                func_logger.info(f"{log_prefix} LLM indicated no changes needed (empty delta array). Keeping previous cache.")
                await _log_trace_update("DELTA_APPLICATION: No changes in delta array. Keeping previous cache.")
                updated_cache_text_final = fallback_cache_text # Explicitly keep previous
            else:
                try:
                    await _log_trace_update("DELTA_APPLICATION: Starting application of changes...")
                    # Parse previous cache into sections
                    current_sections = _parse_cache_into_sections(previous_cache_text)
                    await _log_trace_update(f"DELTA_APPLICATION: Parsed previous cache into {len(current_sections)} sections.")

                    # Apply changes
                    applied_count = 0
                    skipped_count = 0
                    new_sections = current_sections.copy() # Work on a copy

                    for change in delta_changes:
                        action = change["action"]
                        heading = change["heading"].strip() # Ensure heading is stripped
                        content = change.get("content", "") # Default to empty string if missing (already validated for add/modify)

                        if action == "add":
                            if heading in new_sections:
                                func_logger.warning(f"{log_prefix} Delta action 'add' skipped: Heading '{heading}' already exists.")
                                await _log_trace_update(f"DELTA_APPLICATION: Skipped 'add' for existing heading: '{heading}'")
                                skipped_count += 1
                            else:
                                new_sections[heading] = content
                                func_logger.debug(f"{log_prefix} Applied delta 'add' for heading: '{heading}'")
                                await _log_trace_update(f"DELTA_APPLICATION: Applied 'add' for heading: '{heading}' (Content len: {len(content)})")
                                applied_count += 1
                        elif action == "modify":
                             # Try exact match first
                             if heading in new_sections:
                                 new_sections[heading] = content
                                 func_logger.debug(f"{log_prefix} Applied delta 'modify' for heading: '{heading}'")
                                 await _log_trace_update(f"DELTA_APPLICATION: Applied 'modify' for exact heading: '{heading}' (New content len: {len(content)})")
                                 applied_count += 1
                             else:
                                 # Optional: Add fuzzy matching here if needed, but increases complexity
                                 func_logger.warning(f"{log_prefix} Delta action 'modify' skipped: Heading '{heading}' not found in cache.")
                                 await _log_trace_update(f"DELTA_APPLICATION: Skipped 'modify', heading not found: '{heading}'")
                                 skipped_count += 1

                        elif action == "delete":
                             if heading in new_sections:
                                 del new_sections[heading]
                                 func_logger.debug(f"{log_prefix} Applied delta 'delete' for heading: '{heading}'")
                                 await _log_trace_update(f"DELTA_APPLICATION: Applied 'delete' for heading: '{heading}'")
                                 applied_count += 1
                             else:
                                 func_logger.warning(f"{log_prefix} Delta action 'delete' skipped: Heading '{heading}' not found.")
                                 await _log_trace_update(f"DELTA_APPLICATION: Skipped 'delete', heading not found: '{heading}'")
                                 skipped_count += 1

                    # Reconstruct the cache
                    updated_cache_text_final = _reconstruct_cache_from_sections(new_sections)
                    func_logger.info(f"{log_prefix} Cache reconstructed after applying delta. Applied: {applied_count}, Skipped: {skipped_count}. Final len: {len(updated_cache_text_final)}")
                    await _log_trace_update(f"DELTA_APPLICATION: Completed. Applied: {applied_count}, Skipped: {skipped_count}. Reconstructed cache len: {len(updated_cache_text_final)}")

                except Exception as e_apply:
                    func_logger.error(f"{log_prefix} Exception applying delta changes: {e_apply}", exc_info=True)
                    await _log_trace_update(f"DELTA_APPLICATION: FAILED with exception: {e_apply}")
                    updated_cache_text_final = fallback_cache_text # Revert to fallback on error

    # --- If LLM call failed or JSON/Delta was invalid ---
    elif not success:
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"{log_prefix} Cache update delta LLM call failed. Error: '{error_details}'.")
        await _log_trace_update(f"DECISION: LLM call failed. Using fallback (previous cache).")
        updated_cache_text_final = fallback_cache_text # Ensure fallback is used
    # else: (JSON parsing/validation failed cases already logged and use fallback)


    # --- Save the final determined state to DB ---
    save_success = False
    try:
        # Save the 'updated_cache_text_final' which is either the reconstructed cache or the fallback
        save_success = await add_or_update_rag_cache(session_id, updated_cache_text_final, sqlite_cursor)
        if save_success:
            func_logger.info(f"{log_prefix} Successfully saved final RAG cache state to DB.")
            await _log_trace_update(f"DB_WRITE: Successfully saved final cache state (len: {len(updated_cache_text_final)}).")
        else:
            func_logger.error(f"{log_prefix} Failed to save final RAG cache state to DB!")
            await _log_trace_update(f"DB_WRITE: FAILED to save final cache state (len: {len(updated_cache_text_final)}).")
    except Exception as e_save:
        func_logger.error(f"{log_prefix} Exception saving final RAG cache state: {e_save}", exc_info=True)
        await _log_trace_update(f"DB_WRITE: Exception saving final cache state: {e_save}")

    # Return the final determined cache text
    await _log_trace_update(f"FINAL RETURN value (len): {len(updated_cache_text_final)}")
    await _log_trace_update("--- update_rag_cache END (Delta Generation) ---")
    return updated_cache_text_final


# ==============================================================================
# === Step 2 Orchestration: Select Final Context (Constant Toggle Logic)     ===
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

    # --- Use constant to determine OWI inclusion ---
    current_owi_rag_text_for_prompt = ""
    if INCLUDE_OWI_IN_SELECTION:
        current_owi_rag_text_for_prompt = current_owi_context if current_owi_context else EMPTY_OWI_CONTEXT_PLACEHOLDER
        func_logger.debug(f"{log_prefix} Including OWI context in selection prompt based on constant.")
        await _log_trace_select(f"PREP current_owi_rag_text_for_prompt set to actual OWI content (len): {len(current_owi_rag_text_for_prompt)}")
    else:
        current_owi_rag_text_for_prompt = OWI_EXCLUDED_PLACEHOLDER
        func_logger.debug(f"{log_prefix} Excluding OWI context from selection prompt based on constant.")
        await _log_trace_select(f"PREP current_owi_rag_text_for_prompt set to placeholder: '{OWI_EXCLUDED_PLACEHOLDER}'")
    # --- End OWI inclusion logic ---

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

# === END MODIFIED FILE: i4_llm_agent/cache.py (Delta Generation Logic) ===