# [[START COMPLETE CORRECTED utils.py - Added asyncio import]]
# i4_llm_agent/utils.py
import logging
import difflib
import os
import json
import asyncio  # <<< ADDED IMPORT
from datetime import datetime, timezone
from typing import Any, Optional, Dict, Callable, Coroutine

# --- Tiktoken Import Handling ---
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__) # i4_llm_agent.utils
if not TIKTOKEN_AVAILABLE:
     logger.warning("tiktoken library not found. Token counting functions will not work.")

# ==============================================================================
# === Core Utilities                                                         ===
# ==============================================================================

# --- Token Counting Function ---
def count_tokens(text: Optional[str], tokenizer: Optional[Any]) -> int:
    """
    Counts the number of tokens in a given text using the provided tokenizer.
    (Implementation remains the same as provided previously)
    """
    if not text or not isinstance(text, str): return 0
    if not tokenizer: logger.warning("count_tokens: Tokenizer instance missing."); return 0
    if not hasattr(tokenizer, 'encode'): logger.error("count_tokens: Tokenizer lacks 'encode' method."); return 0
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"count_tokens: Error during tokenization encode: {e}", exc_info=False)
        return 0

# --- String Similarity Function ---
def calculate_string_similarity(
    text_a: Optional[str],
    text_b: Optional[str],
    method: str = 'sequencematcher',
    lowercase: bool = True
) -> float:
    """
    Calculates similarity between two strings using the specified method.
    (Implementation remains the same as provided previously)
    """
    if not text_a or not isinstance(text_a, str) or not text_b or not isinstance(text_b, str): return 0.0
    str_a = text_a.lower() if lowercase else text_a
    str_b = text_b.lower() if lowercase else text_b
    try:
        if method == 'sequencematcher':
            similarity = difflib.SequenceMatcher(None, str_a, str_b, autojunk=False).ratio()
            return similarity
        else:
            logger.warning(f"calculate_string_similarity: Unknown method '{method}'.")
            return 0.0
    except Exception as e:
        logger.error(f"calculate_string_similarity: Error calculating similarity ({method}): {e}", exc_info=False)
        return 0.0

# ==============================================================================
# === Debug Logging Utilities (Moved from Orchestrator)                      ===
# ==============================================================================

def get_debug_log_path(suffix: str, config: Any, logger_instance: logging.Logger) -> Optional[str]:
    """
    Gets the path for a specific debug log file based on the main log path and a suffix.
    Moved from Orchestrator.

    Args:
        suffix: The suffix to append to the base log filename (e.g., ".DEBUG_PAYLOAD").
        config: The configuration object (must have 'log_file_path' attribute).
        logger_instance: The logger instance to use for reporting errors.

    Returns:
        The full path string to the debug log file, or None if path cannot be determined.
    """
    func_logger = logger_instance
    func_logger.debug(f"Utils Get Debug Path: Called with suffix: '{suffix}'")
    try:
        base_log_path = getattr(config, "log_file_path", None)
        if not base_log_path or not isinstance(base_log_path, str):
            func_logger.error("Utils Get Debug Path: Main log_file_path config is missing or invalid.")
            return None

        log_dir = os.path.dirname(base_log_path)
        if not log_dir: # Handle case where path might be just a filename
             log_dir = "."
        func_logger.debug(f"Utils Get Debug Path: Target log directory: '{log_dir}'")

        try:
            os.makedirs(log_dir, exist_ok=True)
        except PermissionError as pe:
            func_logger.error(f"Utils Get Debug Path: PERMISSION ERROR creating log directory '{log_dir}': {pe}")
            return None
        except Exception as e_mkdir:
            func_logger.error(f"Utils Get Debug Path: Error creating log directory '{log_dir}': {e_mkdir}", exc_info=True)
            return None

        base_name, _ = os.path.splitext(os.path.basename(base_log_path))
        # Sanitize suffix: allow alphanumeric, hyphen, underscore, dot
        safe_suffix = "".join(c for c in suffix if c.isalnum() or c in ('-', '_', '.'))
        if not safe_suffix.startswith('.'): # Ensure it starts like an extension
             safe_suffix = "." + safe_suffix

        debug_filename = f"{base_name}{safe_suffix}.log"
        final_path = os.path.join(log_dir, debug_filename)
        func_logger.debug(f"Utils Get Debug Path: Constructed debug log path: '{final_path}'")
        return final_path

    except AttributeError as ae:
        func_logger.error(f"Utils Get Debug Path: Config object missing attribute ('log_file_path'?): {ae}")
        return None
    except Exception as e:
        func_logger.error(f"Utils Get Debug Path: Failed get debug path for suffix '{suffix}': {e}", exc_info=True)
        return None

def log_debug_payload(session_id: str, payload_body: Dict, config: Any, logger_instance: logging.Logger):
    """
    Logs the final constructed LLM payload ('contents' structure) to a debug file.
    Moved from Orchestrator.

    Args:
        session_id: The session ID.
        payload_body: The dictionary containing the LLM payload (expects 'contents' key).
        config: The configuration object (needs 'debug_log_final_payload', 'version').
        logger_instance: The logger instance.
    """
    func_logger = logger_instance
    if not getattr(config, 'debug_log_final_payload', False):
        # func_logger.debug(f"[{session_id}] Utils Log Payload: Skipping log, debug valve is OFF.")
        return

    debug_log_path = get_debug_log_path(".DEBUG_PAYLOAD", config, func_logger)
    if not debug_log_path:
        func_logger.error(f"[{session_id}] Utils Log Payload: Cannot log final payload, no path determined.")
        return

    try:
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "ts": ts,
            "pipe_version": getattr(config, "version", "unknown"),
            "sid": session_id,
            "payload_contents": payload_body.get("contents"), # Extract contents
            "payload_other_keys": {k:v for k,v in payload_body.items() if k != 'contents'}
        }
        func_logger.debug(f"[{session_id}] Utils Log Payload: Attempting write FINAL PAYLOAD debug log to: {debug_log_path}")
        # Using simple write for now
        with open(debug_log_path, "a", encoding="utf-8") as f:
             f.write(f"--- [{ts}] SESSION: {session_id} - FINAL ORCHESTRATOR PAYLOAD --- START ---\n")
             json.dump(log_entry, f, indent=2)
             f.write(f"\n--- [{ts}] SESSION: {session_id} - FINAL ORCHESTRATOR PAYLOAD --- END ---\n\n")
        func_logger.debug(f"[{session_id}] Utils Log Payload: Successfully wrote FINAL PAYLOAD debug log.")
    except Exception as e:
        func_logger.error(f"[{session_id}] Utils Log Payload: Failed write debug final payload log: {e}", exc_info=True)


def log_inventory_debug(
    session_id: str,
    message: str,
    log_type: str, # e.g., "LLM_Raw_Output", "Applied_Changes"
    config: Any,
    logger_instance: logging.Logger
):
    """
    Logs inventory-specific debug messages (like raw LLM output or applied changes)
    to the dedicated inventory debug file. This is the synchronous core logic.

    Args:
        session_id: The session ID.
        message: The string message to log.
        log_type: A string indicating the type of message being logged.
        config: The configuration object.
        logger_instance: The logger instance.
    """
    func_logger = logger_instance
    debug_log_path = get_debug_log_path(".DEBUG_INVENTORY", config, func_logger)
    if not debug_log_path:
        func_logger.error(f"[{session_id}] Utils Inventory Log: Cannot log '{log_type}', no path determined.")
        return
    try:
        ts = datetime.now(timezone.utc).isoformat()
        # Format the log entry clearly indicating the type
        log_entry = f"--- [{ts}] SESSION: {session_id} - TYPE: {log_type} --- START ---\n{message}\n--- [{ts}] SESSION: {session_id} - TYPE: {log_type} --- END ---\n\n"

        # Using simple synchronous write for now, as async file I/O adds complexity (aiofiles)
        # If this becomes a bottleneck, switch to aiofiles.
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        # func_logger.debug(f"[{session_id}] Utils Inventory Log: Successfully wrote '{log_type}' to {debug_log_path}")

    except Exception as e:
        func_logger.error(f"[{session_id}] Utils Inventory Log: Failed write debug log for '{log_type}': {e}", exc_info=True)


# --- Async Wrapper for log_inventory_debug (Corrected) ---
async def awaitable_log_inventory_debug(
    session_id: str,
    message: str,
    log_type: str,
    config: Any,
    logger_instance: logging.Logger
):
     """ Awaitable wrapper around log_inventory_debug. Corrected with asyncio import. """
     try:
          # Call the synchronous logging function
          log_inventory_debug(session_id, message, log_type, config, logger_instance)
          # Yield control back to the event loop briefly (requires asyncio)
          await asyncio.sleep(0)
     except Exception as e:
          # Log any error occurring within this wrapper or the sync function it calls
          logger_instance.error(f"[{session_id}] Error in awaitable_log_inventory_debug wrapper: {e}", exc_info=True)


# [[END COMPLETE CORRECTED utils.py - Added asyncio import]]