# === START OF FILE i4_llm_agent/orchestration.py ===

# [[START CORRECTED orchestration.py - Restore Helper Logic]]
# i4_llm_agent/orchestration.py

import logging
import asyncio
import re
import sqlite3
import json
import uuid
import os
from datetime import datetime, timezone
from typing import (
    Tuple, Union, List, Dict, Optional, Any, Callable, Coroutine, AsyncGenerator, Sequence
)
import importlib.util

# --- Standard Library Imports ---
import urllib.parse

# --- i4_llm_agent Imports ---
from .session import SessionManager
from .database import (
    # T1 DB Functions used by Orchestrator
    add_tier1_summary, get_recent_tier1_summaries, get_tier1_summary_count,
    get_oldest_tier1_summary, delete_tier1_summary, get_max_t1_end_index,
    # T2 DB Functions (needed by _handle_tier2_transition)
    get_or_create_chroma_collection, add_to_chroma_collection,
    delete_tier1_summary, # Needed by _handle_tier2_transition
    CHROMADB_AVAILABLE, ChromaEmbeddingFunction, ChromaCollectionType,
    InvalidDimensionException,
    # Inventory DB (Update function still used here)
    get_character_inventory_data, # Needed for update func
    add_or_update_character_inventory, # Needed for update func
    # World State DB (Used directly by Orchestrator)
    get_world_state, set_world_state,
    # Scene State DB (Used directly by Orchestrator)
    get_scene_state, set_scene_state,
)
from .history import (
    format_history_for_llm, get_recent_turns, DIALOGUE_ROLES, select_turns_for_t0
)
from .memory import manage_tier1_summarization
from .api_client import call_google_llm_api # Still needed for direct calls (final, state, inv)

try:
    # Utils still needed for token counting status etc.
    from .utils import TIKTOKEN_AVAILABLE, count_tokens, calculate_string_similarity
except ImportError:
    TIKTOKEN_AVAILABLE = False
    def count_tokens(*args, **kwargs): return 0
    def calculate_string_similarity(*args, **kwargs): return 0.0
    logging.getLogger(__name__).warning("Orchestration: Failed to import utils (tiktoken?). Token counting/similarity may be affected.")

# --- Prompting Imports (REDUCED - most moved to context_processor) ---
from .prompting import (
    format_inventory_update_prompt, # Still needed for inventory update step
    # Default templates still needed for direct calls
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
)

# === Event Hint Import (Unchanged) ===
try:
    from .event_hints import generate_event_hint, EVENT_HANDLING_GUIDELINE_TEXT, format_hint_for_query, DEFAULT_EVENT_HINT_TEMPLATE_TEXT
    _EVENT_HINTS_AVAILABLE = True
    try:
        spec = importlib.util.find_spec(".event_hints", package="i4_llm_agent")
        if spec and spec.origin: _event_hints_import_path = spec.origin
        else: _event_hints_import_path = "Unknown (spec missing?)"
    except Exception: _event_hints_import_path = "Unknown (spec error)"
    logging.getLogger(__name__).info(f"Successfully imported generate_event_hint from: {_event_hints_import_path}")
except ImportError as e_import:
    logging.getLogger(__name__).error(f"Failed to import from .event_hints: {e_import}", exc_info=True)
    _EVENT_HINTS_AVAILABLE = False
    # Fallback still needs to return tuple expected by orchestrator
    async def generate_event_hint(*args, **kwargs):
        logging.getLogger(__name__).error("Executing FALLBACK generate_event_hint due to import error.")
        await asyncio.sleep(0)
        return None, {} # Return hint text and empty dict (for removed proposal)
    EVENT_HANDLING_GUIDELINE_TEXT = "[EVENT GUIDELINE LOAD FAILED]"
    def format_hint_for_query(hint): return f"[[Hint Load Failed: {hint}]]"
    DEFAULT_EVENT_HINT_TEMPLATE_TEXT = "[Default Event Hint Template Load Failed]"


# === Unified State Assessment Import (Unchanged) ===
try:
    from .state_assessment import update_state_via_full_turn_assessment, DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT
    _UNIFIED_STATE_ASSESSMENT_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported state_assessment module.")
except ImportError as e_state_assess:
    logging.getLogger(__name__).error(f"Failed to import state_assessment module: {e_state_assess}", exc_info=True)
    _UNIFIED_STATE_ASSESSMENT_AVAILABLE = False
    DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT = "[Default Unified State Assessment Template Load Failed]"
    # Define a fallback async function that matches the signature and returns previous state
    async def update_state_via_full_turn_assessment(
        session_id: str, previous_world_state: Dict[str, Any], previous_scene_state: Dict[str, Any],
        current_user_query: str, assistant_response_text: str, history_messages: List[Dict],
        llm_call_func: Callable, state_assessment_llm_config: Dict[str, Any],
        logger_instance: Optional[logging.Logger] = None, event_emitter: Optional[Callable] = None,
        weather_proposal: Optional[Dict[str, Optional[str]]] = None # Add new param here too
    ) -> Dict[str, Any]:
        lg = logger_instance or logging.getLogger(__name__)
        lg.error(f"[{session_id}] Executing FALLBACK update_state_via_full_turn_assessment due to import error.")
        await asyncio.sleep(0)
        # Return previous state structure
        return {
            "new_day": previous_world_state.get("day", 1),
            "new_time_of_day": previous_world_state.get("time_of_day", "Morning"),
            "new_weather": previous_world_state.get("weather", "Clear"),
            "new_season": previous_world_state.get("season", "Summer"),
            "new_scene_keywords": previous_scene_state.get("keywords", []),
            "new_scene_description": previous_scene_state.get("description", ""),
            "scene_changed_flag": False
        }

# === Inventory Module Import (Only Update Function needed here now) ===
try:
    # Format function no longer needed directly by orchestrator
    from .inventory import update_inventories_from_llm as _real_update_inventories_func
    _ORCH_INVENTORY_MODULE_AVAILABLE = True
    _dummy_update_inventories = None
except ImportError:
    _ORCH_INVENTORY_MODULE_AVAILABLE = False
    _real_update_inventories_func = None
    async def _dummy_update_inventories(*args, **kwargs): await asyncio.sleep(0); return False
    logging.getLogger(__name__).warning(
        "Orchestration: Inventory module not found. Inventory update feature disabled."
        )
# --- END LOCAL Inventory Import ---

# === NEW: Import Context Processor Function ===
try:
    from .context_processor import process_context_and_prepare_payload
    _CONTEXT_PROCESSOR_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported context_processor module.")
except ImportError as e_ctx_proc:
    _CONTEXT_PROCESSOR_AVAILABLE = False
    logging.getLogger(__name__).error(f"Failed to import context_processor: {e_ctx_proc}", exc_info=True)
    # Define a fallback async function
    async def process_context_and_prepare_payload(*args, **kwargs) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
        lg = kwargs.get('logger') or logging.getLogger(__name__)
        session_id = kwargs.get('session_id', 'unknown')
        lg.error(f"[{session_id}] Executing FALLBACK process_context_and_prepare_payload due to import error.")
        await asyncio.sleep(0)
        # Return empty payload and error status
        return None, {"error": "Context processor unavailable"}
# === END NEW IMPORT ===


logger = logging.getLogger(__name__) # i4_llm_agent.orchestration

OrchestratorResult = Union[Dict, str]

# ==============================================================================
# === Session Pipe Orchestrator Class (Refactored)                           ===
# ==============================================================================

class SessionPipeOrchestrator:
    """
    Orchestrates the core processing logic of the Session Memory Pipe.
    Coordinates memory management, state assessment, context processing,
    inventory updates, and final LLM calls/payload preparation.
    Relies on context_processor module for background prep and payload construction.
    """

    def __init__(
        self,
        config: object,
        session_manager: SessionManager,
        sqlite_cursor: sqlite3.Cursor,
        chroma_client: Optional[Any] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """Initializes the orchestrator with config, manager, DB, and clients."""
        self.config = config
        self.session_manager = session_manager
        self.sqlite_cursor = sqlite_cursor
        self.chroma_client = chroma_client if CHROMADB_AVAILABLE else None
        self.logger = logger_instance or logger
        self.pipe_logger = logger_instance or logger
        self.pipe_debug_path_getter = None # Set externally if needed

        # --- Log Config ---
        self.logger.debug("Orchestrator __init__: Received config object.")
        try:
            config_dump = {}
            if hasattr(config, 'model_dump'): config_dump = config.model_dump()
            elif hasattr(config, 'dict'): config_dump = config.dict()
            elif hasattr(config, '__dict__'): config_dump = config.__dict__
            safe_config_log = {
                k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                for k, v in config_dump.items()
                if "api_key" not in k.lower() and "prompt" not in k.lower() # Filter prompts
            }
            self.logger.debug(f"Orchestrator __init__: Received config values (filtered): {safe_config_log}")
        except Exception as e_dump:
            self.logger.error(f"Orchestrator __init__: Error dumping received config: {e_dump}")


        # --- Tokenizer Init ---
        self._tokenizer = None
        if TIKTOKEN_AVAILABLE and hasattr(self.config, 'tokenizer_encoding_name'):
            try:
                self.logger.info(f"Orchestrator: Initializing tokenizer '{self.config.tokenizer_encoding_name}'...")
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self.config.tokenizer_encoding_name)
                self.logger.info("Orchestrator: Tokenizer initialized.")
            except Exception as e:
                self.logger.error(f"Orchestrator: Tokenizer init failed: {e}. Token counting disabled.", exc_info=True)
        elif not TIKTOKEN_AVAILABLE:
             self.logger.warning("Orchestrator: tiktoken unavailable. Token counting disabled.")

        # --- Function Aliases (REDUCED) ---
        self._llm_call_func = call_google_llm_api # For final, state, inv calls
        self._format_history_func = format_history_for_llm # Used for inv update context
        self._get_recent_turns_func = get_recent_turns # Used for inv update context & T0 select
        self._manage_memory_func = manage_tier1_summarization # T1/T2 management
        self._count_tokens_func = count_tokens # For status reporting
        self._calculate_similarity_func = calculate_string_similarity # Used by cache logic (now in context_processor) - Keep for now?
        self._dialogue_roles = DIALOGUE_ROLES # Passed to context processor and memory
        # DB Aliases (Only those used directly by orchestrator now)
        self._get_world_state_db_func = get_world_state
        self._set_world_state_db_func = set_world_state
        self._get_scene_state_db_func = get_scene_state
        self._set_scene_state_db_func = set_scene_state
        # Event Hint alias
        self._generate_hint_func = generate_event_hint if _EVENT_HINTS_AVAILABLE else None
        # Unified State Assessment Alias
        self._unified_state_func = update_state_via_full_turn_assessment
        # Inventory Update Alias (Formatting moved, Update stays)
        if _ORCH_INVENTORY_MODULE_AVAILABLE:
            self._update_inventories_func = _real_update_inventories_func
        else:
            self._update_inventories_func = _dummy_update_inventories

        # === NEW ALIAS for context processor ===
        self._context_processor_func = process_context_and_prepare_payload if _CONTEXT_PROCESSOR_AVAILABLE else None

        self.logger.info("SessionPipeOrchestrator initialized (Refactored Context Processing).")
        self.logger.info(f"Context Processor Status Check (Init): Available={_CONTEXT_PROCESSOR_AVAILABLE}")


    # --- Internal Helper: Status Emitter ---
    async def _emit_status(
        self,
        event_emitter: Optional[Callable],
        session_id: str,
        description: str,
        done: bool = False
    ):
        """Emits status updates via the provided callable if configured."""
        # [[ Restored Full Implementation ]]
        if event_emitter and callable(event_emitter) and getattr(self.config, 'emit_status_updates', True):
            try:
                status_data = { "type": "status", "data": {"description": str(description), "done": bool(done)} }
                if asyncio.iscoroutinefunction(event_emitter):
                    await event_emitter(status_data)
                else:
                    event_emitter(status_data)
            except Exception as e_emit:
                self.logger.warning(f"[{session_id}] Orchestrator failed to emit status '{description}': {e_emit}")
        else:
             self.logger.debug(f"[{session_id}] Orchestrator status update (not emitted): '{description}' (Done: {done})")


    # --- Internal Helper: Async LLM Call Wrapper ---
    async def _async_llm_call_wrapper(
        self,
        api_url: str,
        api_key: str,
        payload: Dict[str, Any],
        temperature: float,
        timeout: int = 90,
        caller_info: str = "Orchestrator_LLM",
    ) -> Tuple[bool, Union[str, Dict]]:
        """ Wraps the library's LLM call function for error handling. """
        # [[ Restored Full Implementation ]]
        if not self._llm_call_func:
            self.logger.error(f"[{caller_info}] LLM func alias unavailable in orchestrator.")
            return False, {"error_type": "SetupError", "message": "LLM func alias unavailable"}
        if not asyncio.iscoroutinefunction(self._llm_call_func):
             self.logger.critical(f"[{caller_info}] LLM func alias is NOT async! Cannot proceed.")
             return False, {"error_type": "SetupError", "message": "LLM func alias is not async"}
        try:
            self.logger.debug(f"[{caller_info}] Awaiting result from LLM adapter function.")
            success, result_or_error = await self._llm_call_func(
                api_url=api_url, api_key=api_key, payload=payload,
                temperature=temperature, timeout=timeout, caller_info=caller_info
            )
            self.logger.debug(f"[{caller_info}] LLM adapter returned (Success: {success}).")
            return success, result_or_error
        except asyncio.CancelledError:
             self.logger.info(f"[{caller_info}] LLM call cancelled.")
             raise
        except Exception as e:
            self.logger.error(f"Orchestrator LLM Wrapper Error [{caller_info}]: Uncaught exception during await: {e}", exc_info=True)
            return False, {"error_type": "AsyncWrapperError", "message": f"{type(e).__name__}: {str(e)}"}


    # --- Debug Logging Helpers ---
    def _orchestrator_get_debug_log_path(self, suffix: str) -> Optional[str]:
        """ Gets the path for a debug log file based on the main log path and a suffix. """
        # [[ Restored Full Implementation ]]
        func_logger = getattr(self, 'pipe_logger', self.logger)
        func_logger.debug(f"_orchestrator_get_debug_log_path called with suffix: '{suffix}'")
        try:
            base_log_path = getattr(self.config, "log_file_path", None)
            if not base_log_path:
                func_logger.error("Orch Debug Path: Main log_file_path config is empty.")
                return None
            log_dir = os.path.dirname(base_log_path)
            func_logger.debug(f"Orch Debug Path: Target log directory: '{log_dir}'")
            try: os.makedirs(log_dir, exist_ok=True)
            except PermissionError as pe: func_logger.error(f"Orch Debug Path: PERMISSION ERROR creating log directory '{log_dir}': {pe}"); return None
            except Exception as e_mkdir: func_logger.error(f"Orch Debug Path: Error creating log directory '{log_dir}': {e_mkdir}", exc_info=True); return None
            base_name, _ = os.path.splitext(os.path.basename(base_log_path))
            safe_suffix = "".join(c for c in suffix if c.isalnum() or c in ('-', '_', '.')) # Sanitize suffix
            debug_filename = f"{base_name}{safe_suffix}.log"
            final_path = os.path.join(log_dir, debug_filename)
            func_logger.debug(f"Orch Debug Path: Constructed debug log path: '{final_path}'")
            return final_path
        except AttributeError as ae: func_logger.error(f"Orch Debug Path: Config object missing attribute ('log_file_path'?): {ae}"); return None
        except Exception as e: func_logger.error(f"Orch Debug Path: Failed get debug path '{suffix}': {e}", exc_info=True); return None

    def _orchestrator_log_debug_payload(self, session_id: str, payload_body: Dict):
        """ Logs the final constructed LLM payload to a debug file if enabled. """
        # [[ Restored Full Implementation ]]
        debug_log_path = self._orchestrator_get_debug_log_path(".DEBUG_PAYLOAD")
        if not debug_log_path: self.logger.error(f"[{session_id}] Orch: Cannot log final payload: No path determined."); return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_entry = { "ts": ts, "pipe_version": getattr(self.config, "version", "unknown"), "sid": session_id, "payload": payload_body, }
            self.logger.debug(f"[{session_id}] Orch: Attempting write FINAL PAYLOAD debug log to: {debug_log_path}")
            with open(debug_log_path, "a", encoding="utf-8") as f:
                 f.write(f"--- [{ts}] SESSION: {session_id} - FINAL ORCHESTRATOR PAYLOAD --- START ---\n")
                 # Handle potential large 'contents' separately if needed
                 if 'contents' in payload_body:
                      log_entry_payload = payload_body.copy()
                      log_entry['payload']['contents'] = log_entry_payload.pop('contents', None)
                      log_entry['payload_other_keys'] = log_entry_payload
                 json.dump(log_entry, f, indent=2)
                 f.write(f"\n--- [{ts}] SESSION: {session_id} - FINAL ORCHESTRATOR PAYLOAD --- END ---\n\n")
            self.logger.debug(f"[{session_id}] Orch: Successfully wrote FINAL PAYLOAD debug log.")
        except Exception as e: self.logger.error(f"[{session_id}] Orch: Failed write debug final payload log: {e}", exc_info=True)

    def _orchestrator_log_debug_inventory_llm(self, session_id: str, text: str, is_prompt: bool):
        """ Logs the inventory LLM prompt or response to the debug payload file. """
        # [[ Restored Full Implementation ]]
        debug_log_path = self._orchestrator_get_debug_log_path(".DEBUG_PAYLOAD") # Log to same file as payload
        if not debug_log_path: self.logger.error(f"[{session_id}] Orch: Cannot log inventory LLM text: No path determined."); return
        # Check the same debug flag as the main payload
        if not getattr(self.config, 'debug_log_final_payload', False): return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_type = "PROMPT" if is_prompt else "RESPONSE"
            self.logger.debug(f"[{session_id}] Orch: Attempting write INVENTORY LLM {log_type} debug log to: {debug_log_path}")
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- [{ts}] SESSION: {session_id} - INVENTORY LLM {log_type} --- START ---\n"); f.write(str(text)); f.write(f"\n--- [{ts}] SESSION: {session_id} - INVENTORY LLM {log_type} --- END ---\n\n")
            self.logger.debug(f"[{session_id}] Orch: Successfully wrote INVENTORY LLM {log_type} debug log.")
        except Exception as e: self.logger.error(f"[{session_id}] Orch: Failed write debug inventory LLM {log_type} log: {e}", exc_info=True)
    # --- END DEBUG LOGGING HELPERS ---

    # --- Helper Methods for process_turn ---

    # === _determine_effective_query ===
    async def _determine_effective_query(
        self, session_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool
    ) -> Tuple[str, List[Dict], Optional[str]]:
        """ Determines the effective user query, history slice, and last assistant response. """
        # [[ Restored Full Implementation ]]
        effective_user_message_index = -1
        last_assistant_message_str: Optional[str] = None
        history_for_processing: List[Dict] = []
        latest_user_query_str: str = ""

        if not current_active_history:
            self.logger.error(f"[{session_id}] Cannot determine query: Active history is empty.")
            # Return structure indicating failure/empty state
            return "", [], None

        # Find indices of user messages
        user_message_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") == "user"]

        if not user_message_indices:
            # Handle case with no user messages (e.g., initial system prompt only?)
            self.logger.error(f"[{session_id}] No user messages found in history.")
            history_for_processing = current_active_history # Process full history?
            # Find last assistant message if any
            assistant_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") in ("assistant", "model")]
            if assistant_indices:
                last_assistant_msg = current_active_history[assistant_indices[-1]]
                last_assistant_message_str = last_assistant_msg.get("content") if isinstance(last_assistant_msg, dict) else None
            return "", history_for_processing, last_assistant_message_str

        # Determine which user message to use based on regeneration flag
        if is_regeneration_heuristic:
            # Use the second-to-last user message if available, otherwise the last
            effective_user_message_index = user_message_indices[-2] if len(user_message_indices) >= 2 else user_message_indices[-1]
            log_level = self.logger.info if len(user_message_indices) >= 2 else self.logger.warning
            log_level(f"[{session_id}] Regen: Using user message at index {effective_user_message_index} as query base.")
        else:
            # Use the last user message for normal processing
            effective_user_message_index = user_message_indices[-1]
            self.logger.debug(f"[{session_id}] Normal: Using user message at index {effective_user_message_index} as query base.")

        # Validate index
        if effective_user_message_index < 0 or effective_user_message_index >= len(current_active_history):
             self.logger.error(f"[{session_id}] Effective user index {effective_user_message_index} out of bounds for history len {len(current_active_history)}.")
             return "", [], None # Indicate error

        # Extract the query and the history *before* that query
        effective_user_message = current_active_history[effective_user_message_index]
        history_for_processing = current_active_history[:effective_user_message_index] # Slice up to the effective query
        latest_user_query_str = effective_user_message.get("content", "") if isinstance(effective_user_message, dict) else ""

        # Find the last assistant message *within the history slice*
        assistant_indices_in_slice = [i for i, msg in enumerate(history_for_processing) if isinstance(msg, dict) and msg.get("role") in ("assistant", "model")]
        if assistant_indices_in_slice:
            last_assistant_msg_in_slice = history_for_processing[assistant_indices_in_slice[-1]]
            last_assistant_message_str = last_assistant_msg_in_slice.get("content") if isinstance(last_assistant_msg_in_slice, dict) else None
            self.logger.debug(f"[{session_id}] Found last assistant message at index {assistant_indices_in_slice[-1]} before query.")
        else:
            self.logger.debug(f"[{session_id}] No assistant message found before the effective user query.")
            last_assistant_message_str = None

        self.logger.debug(f"[{session_id}] Effective query set (len: {len(latest_user_query_str)}). History slice len: {len(history_for_processing)}. Last assistant msg len: {len(last_assistant_message_str or '')}.")
        return latest_user_query_str, history_for_processing, last_assistant_message_str


    # === _handle_tier1_summarization ===
    async def _handle_tier1_summarization(
        self, session_id: str, user_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool, event_emitter: Optional[Callable]
    ) -> Tuple[bool, Optional[str], int, int]:
        """ Handles checking for and executing Tier 1 summarization. """
        # [[ Restored Full Implementation ]]
        await self._emit_status(event_emitter, session_id, "Status: Checking summarization...")
        summarization_performed_successfully = False; generated_summary = None; summarization_prompt_tokens = -1; summarization_output_tokens = -1
        summarizer_url = getattr(self.config, 'summarizer_api_url', None)
        summarizer_key = getattr(self.config, 'summarizer_api_key', None)
        # Prompt is handled by memory.py using library default

        can_summarize = all([ self._manage_memory_func, self._tokenizer, self._count_tokens_func, self.sqlite_cursor, self._async_llm_call_wrapper, summarizer_url, summarizer_key, current_active_history ])
        if not can_summarize:
             missing_prereqs = [p for p, v in {"manage_func": self._manage_memory_func, "tokenizer": self._tokenizer, "count_func": self._count_tokens_func, "db_cursor": self.sqlite_cursor, "llm_wrapper": self._async_llm_call_wrapper, "summ_url": summarizer_url, "summ_key": summarizer_key, "history": bool(current_active_history)}.items() if not v]
             self.logger.warning(f"[{session_id}] Skipping T1 check: Missing prerequisites: {', '.join(missing_prereqs)}."); return False, None, -1, -1

        summarizer_llm_config = {
            "url": summarizer_url,
            "key": summarizer_key,
            "temp": getattr(self.config, 'summarizer_temperature', 0.5),
        }
        self.logger.debug(f"[{session_id}] Orchestrator: Passing summarizer config (URL/Key/Temp) to memory manager.")

        new_last_summary_idx = -1; prompt_tokens = -1; t0_end_idx = -1; db_max_index = None; current_last_summary_index_for_memory = -1
        try:
            db_max_index = await get_max_t1_end_index(self.sqlite_cursor, session_id)
            if isinstance(db_max_index, int) and db_max_index >= 0:
                current_last_summary_index_for_memory = db_max_index
                self.logger.debug(f"[{session_id}] T1: Start Index from DB: {current_last_summary_index_for_memory}")
            else:
                self.logger.debug(f"[{session_id}] T1: No valid start index in DB. Starting from -1.")
        except Exception as e_get_max: self.logger.error(f"[{session_id}] T1: Error getting start index: {e_get_max}. Starting from -1.", exc_info=True); current_last_summary_index_for_memory = -1

        async def _async_save_t1_summary(summary_id: str, session_id: str, user_id: str, summary_text: str, metadata: Dict) -> bool:
             try: return await add_tier1_summary(cursor=self.sqlite_cursor, summary_id=summary_id, session_id=session_id, user_id=user_id, summary_text=summary_text, metadata=metadata)
             except Exception as e_save: self.logger.error(f"[{session_id}] Exception in nested _async_save_t1_summary for {summary_id}: {e_save}", exc_info=True); return False

        try:
            self.logger.debug(f"[{session_id}] Calling manage_tier1_summarization with start index = {current_last_summary_index_for_memory} (Regen={is_regeneration_heuristic})")
            summarization_performed, generated_summary_text, new_last_summary_idx, prompt_tokens, t0_end_idx = await self._manage_memory_func( current_last_summary_index=current_last_summary_index_for_memory, active_history=current_active_history, t0_token_limit=getattr(self.config, 't0_active_history_token_limit', 4000), t1_chunk_size_target=getattr(self.config, 't1_summarization_chunk_token_target', 2000), tokenizer=self._tokenizer, llm_call_func=self._async_llm_call_wrapper, llm_config=summarizer_llm_config, add_t1_summary_func=_async_save_t1_summary, session_id=session_id, user_id=user_id, cursor=self.sqlite_cursor, is_regeneration=is_regeneration_heuristic, dialogue_only_roles=self._dialogue_roles,)
            if summarization_performed:
                summarization_performed_successfully = True; generated_summary = generated_summary_text; summarization_prompt_tokens = prompt_tokens
                if generated_summary and self._count_tokens_func and self._tokenizer:
                    try: summarization_output_tokens = self._count_tokens_func(generated_summary, self._tokenizer)
                    except Exception: summarization_output_tokens = -1
                self.logger.info(f"[{session_id}] T1 summary generated/saved. New Idx: {new_last_summary_idx}, PromptTok: {summarization_prompt_tokens}, OutTok: {summarization_output_tokens}.") # Keep INFO
                await self._emit_status(event_emitter, session_id, "Status: Summary generated.", done=False)
            else: self.logger.debug(f"[{session_id}] T1 summarization skipped or criteria not met (Returned Index: {new_last_summary_idx}).")
        except TypeError as e_type: self.logger.error(f"[{session_id}] Orchestrator TYPE ERROR calling T1 manage func: {e_type}. Signature mismatch?", exc_info=True)
        except Exception as e_manage: self.logger.error(f"[{session_id}] Orchestrator EXCEPTION during T1 manage call: {e_manage}", exc_info=True)
        return summarization_performed_successfully, generated_summary, summarization_prompt_tokens, summarization_output_tokens


    # === _handle_tier2_transition ===
    async def _handle_tier2_transition(
        self, session_id: str, t1_success: bool, chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> None:
        """ Handles checking and transitioning oldest T1 summary to T2 if limits exceeded. """
        # [[ Restored Full Implementation ]]
        await self._emit_status(event_emitter, session_id, "Status: Checking long-term memory capacity...")
        tier2_collection = None
        max_t1_blocks = getattr(self.config, 'max_stored_summary_blocks', 0)
        # Ensure database functions needed are available (imported at top level)
        can_transition = all([
            t1_success,
            self.chroma_client is not None,
            chroma_embed_wrapper is not None,
            self.sqlite_cursor is not None,
            get_tier1_summary_count is not None, # Check DB func availability
            get_oldest_tier1_summary is not None,
            add_to_chroma_collection is not None,
            delete_tier1_summary is not None,
            max_t1_blocks > 0
        ])

        if not can_transition:
            self.logger.debug(f"[{session_id}] Skipping T1->T2 transition check: {'(T1 did not run)' if not t1_success else '(Prerequisites not met)'}.")
            return

        try:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_')
            safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]
            tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63] # Max length for Chroma
            # get_or_create_chroma_collection needs to be available
            if get_or_create_chroma_collection is None: raise ImportError("get_or_create_chroma_collection not available")
            tier2_collection = await get_or_create_chroma_collection(self.chroma_client, tier2_collection_name, chroma_embed_wrapper)
            if not tier2_collection:
                self.logger.error(f"[{session_id}] Failed to get/create T2 collection '{tier2_collection_name}'. Skipping transition.")
                return
        except Exception as e_get_coll:
            self.logger.error(f"[{session_id}] Error getting T2 collection: {e_get_coll}. Skipping transition.", exc_info=True)
            return

        try:
            current_tier1_count = await get_tier1_summary_count(self.sqlite_cursor, session_id)
            if current_tier1_count == -1:
                self.logger.error(f"[{session_id}] Failed get T1 count. Skipping T1->T2 check.")
                return
            elif current_tier1_count > max_t1_blocks:
                self.logger.debug(f"[{session_id}] T1 limit ({max_t1_blocks}) exceeded ({current_tier1_count}). Transitioning oldest...")
                await self._emit_status(event_emitter, session_id, "Status: Archiving oldest summary...")
                oldest_summary_data = await get_oldest_tier1_summary(self.sqlite_cursor, session_id)
                if not oldest_summary_data:
                    self.logger.warning(f"[{session_id}] T1 count exceeded limit, but couldn't retrieve oldest summary.")
                    return

                oldest_id, oldest_text, oldest_metadata = oldest_summary_data
                embedding_vector = None; embedding_successful = False
                try:
                    # Embedding needs to run in a thread if sync
                    embedding_list = await asyncio.to_thread(chroma_embed_wrapper, [oldest_text])
                    if isinstance(embedding_list, list) and len(embedding_list) == 1 and isinstance(embedding_list[0], list) and len(embedding_list[0]) > 0:
                        embedding_vector = embedding_list[0]; embedding_successful = True
                    else: self.logger.error(f"[{session_id}] T1->T2 Embed: Invalid structure returned: {type(embedding_list)}")
                except Exception as embed_e:
                    self.logger.error(f"[{session_id}] EXCEPTION embedding T1->T2 summary {oldest_id}: {embed_e}", exc_info=True)

                if embedding_successful and embedding_vector:
                    added_to_t2 = False; deleted_from_t1 = False
                    chroma_metadata = oldest_metadata.copy()
                    chroma_metadata["transitioned_from_t1"] = True
                    chroma_metadata["original_t1_id"] = oldest_id
                    # Sanitize metadata for ChromaDB (must be str, int, float, bool)
                    sanitized_chroma_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in chroma_metadata.items() if v is not None}
                    tier2_id = f"t2_{oldest_id}" # Ensure unique ID for T2

                    self.logger.debug(f"[{session_id}] Adding summary {tier2_id} to T2 collection '{tier2_collection.name}'...")
                    added_to_t2 = await add_to_chroma_collection(tier2_collection, ids=[tier2_id], embeddings=[embedding_vector], metadatas=[sanitized_chroma_metadata], documents=[oldest_text])

                    if added_to_t2:
                         self.logger.debug(f"[{session_id}] Added {tier2_id} to T2. Deleting T1 summary {oldest_id}...")
                         deleted_from_t1 = await delete_tier1_summary(self.sqlite_cursor, oldest_id)
                         if deleted_from_t1:
                             self.logger.info(f"[{session_id}] Successfully archived T1 summary {oldest_id} to T2.") # Keep INFO for final success
                             await self._emit_status(event_emitter, session_id, "Status: Summary archive complete.", done=False)
                         else: self.logger.critical(f"[{session_id}] Added {tier2_id} to T2, but FAILED TO DELETE T1 {oldest_id}!")
                    else: self.logger.error(f"[{session_id}] Failed to add summary {tier2_id} to T2 collection.")
                else: self.logger.error(f"[{session_id}] Skipping T2 addition for T1 summary {oldest_id}: Embedding failed.")
            else: self.logger.debug(f"[{session_id}] T1 count ({current_tier1_count}) within limit ({max_t1_blocks}). No transition needed.")
        except Exception as e_t2_trans:
            self.logger.error(f"[{session_id}] Unexpected error during T1->T2 transition: {e_t2_trans}", exc_info=True)


    # === _get_t1_summaries ===
    async def _get_t1_summaries(self, session_id: str) -> Tuple[List[str], int]:
        """ Retrieves recent T1 summaries from the database. """
        # [[ Restored Full Implementation ]]
        recent_t1_summaries = []; t1_retrieved_count = 0
        max_blocks_t1 = getattr(self.config, 'max_stored_summary_blocks', 0)
        if self.sqlite_cursor and get_recent_tier1_summaries and max_blocks_t1 > 0:
             try:
                 recent_t1_summaries = await get_recent_tier1_summaries(self.sqlite_cursor, session_id, max_blocks_t1)
                 t1_retrieved_count = len(recent_t1_summaries)
             except Exception as e_get_t1:
                 self.logger.error(f"[{session_id}] Error retrieving T1 summaries: {e_get_t1}", exc_info=True)
                 recent_t1_summaries = []; t1_retrieved_count = 0
        elif not self.sqlite_cursor:
            self.logger.warning(f"[{session_id}] Cannot get T1 summaries: SQLite cursor unavailable.")
        elif not get_recent_tier1_summaries:
             self.logger.warning(f"[{session_id}] Cannot get T1 summaries: get_recent_tier1_summaries function unavailable.")
        elif max_blocks_t1 <= 0:
            self.logger.debug(f"[{session_id}] Skipping T1 summary retrieval: max_stored_summary_blocks is {max_blocks_t1}.")

        if t1_retrieved_count > 0:
            self.logger.debug(f"[{session_id}] Retrieved {t1_retrieved_count} T1 summaries for context.")
        return recent_t1_summaries, t1_retrieved_count


    # === _select_t0_history_slice ===
    async def _select_t0_history_slice(self, session_id: str, history_for_processing: List[Dict]) -> Tuple[List[Dict], int]:
        """ Selects the T0 history slice based on token limits or fallback count. """
        # [[ Restored Full Implementation ]]
        t0_raw_history_slice = []; t0_dialogue_tokens = -1
        t0_token_limit = getattr(self.config, 't0_active_history_token_limit', 4000)
        try:
             if self._tokenizer and select_turns_for_t0:
                  t0_raw_history_slice = select_turns_for_t0(
                      full_history=history_for_processing,
                      target_tokens=t0_token_limit,
                      tokenizer=self._tokenizer,
                      dialogue_only_roles=self._dialogue_roles
                  )
                  self.logger.debug(f"[{session_id}] T0 Slice: Selected {len(t0_raw_history_slice)} dialogue msgs using select_turns_for_t0.")
             else:
                 # Fallback if tokenizer or function is missing
                 self.logger.warning(f"[{session_id}] Tokenizer or select_turns_for_t0 unavailable. Using simple turn count fallback for T0.")
                 fallback_turns = 10 # Define a fallback number of turns
                 # Filter only dialogue roles for the fallback count
                 dialogue_history = [msg for msg in history_for_processing if isinstance(msg, dict) and msg.get("role") in self._dialogue_roles]
                 start_idx = max(0, len(dialogue_history) - fallback_turns)
                 t0_raw_history_slice = dialogue_history[start_idx:]
             # Calculate tokens for the selected slice
             if t0_raw_history_slice and self._count_tokens_func and self._tokenizer:
                 try:
                     t0_dialogue_tokens = sum(self._count_tokens_func(msg["content"], self._tokenizer) for msg in t0_raw_history_slice if isinstance(msg, dict) and isinstance(msg.get("content"), str))
                 except Exception as e_tok_t0:
                     t0_dialogue_tokens = -1
                     self.logger.error(f"[{session_id}] Error calculating T0 tokens: {e_tok_t0}")
             elif not t0_raw_history_slice:
                 t0_dialogue_tokens = 0
             else:
                 # Tokenizer not available for counting
                 t0_dialogue_tokens = -1
        except Exception as e_select_t0:
            self.logger.error(f"[{session_id}] Error during T0 slice selection: {e_select_t0}", exc_info=True)
            t0_raw_history_slice = []; t0_dialogue_tokens = -1
        return t0_raw_history_slice, t0_dialogue_tokens


    # === _calculate_and_format_status (MODIFIED - Takes status_info dict) ===
    async def _calculate_and_format_status(
        self, session_id: str,
        t1_retrieved_count: int, # Still passed directly
        summarization_prompt_tokens: int, # Still passed directly
        summarization_output_tokens: int, # Still passed directly
        t0_dialogue_tokens: int, # Still passed directly
        inventory_prompt_tokens: int, # Still passed directly
        final_llm_payload_contents: Optional[List[Dict]], # Still passed directly
        scene_changed_flag: bool, # Still passed directly
        world_state_status_str: str, # Still passed directly
        context_status_info: Dict[str, Any], # NEW: Dict from context processor
        session_process_owi_rag: bool, # Still needed for context
    ) -> Tuple[str, int]:
        """ Calculates final status string using info from context processor. """
        # [[ Restored Full Implementation (with modifications for status_info) ]]
        final_payload_tokens = -1
        if final_llm_payload_contents and self._count_tokens_func and self._tokenizer:
            try: final_payload_tokens = sum( self._count_tokens_func(part["text"], self._tokenizer) for turn in final_llm_payload_contents if isinstance(turn, dict) for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str))
            except Exception as e_tok_final: final_payload_tokens = -1; self.logger.error(f"[{session_id}] Error calculating final payload tokens: {e_tok_final}")
        elif not final_llm_payload_contents: final_payload_tokens = 0

        # Extract info from the context_status_info dict
        t2_retrieved_count = context_status_info.get("t2_retrieved_count", 0)
        initial_owi_context_tokens = context_status_info.get("initial_owi_context_tokens", -1)
        refined_context_tokens = context_status_info.get("refined_context_tokens", -1)
        # cache_update_performed = context_status_info.get("cache_update_performed", False) # Not directly needed for status string
        cache_update_skipped = context_status_info.get("cache_update_skipped", False)
        final_context_selection_performed = context_status_info.get("final_context_selection_performed", False)
        stateless_refinement_performed = context_status_info.get("stateless_refinement_performed", False)

        status_parts = []
        status_parts.append(f"T1={t1_retrieved_count}")
        status_parts.append(f"T2={t2_retrieved_count}")

        # Refinement indicator logic remains the same, using flags from status_info
        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False); enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False); refinement_indicator = None
        if enable_rag_cache_global and final_context_selection_performed: refinement_indicator = f"Cache(S1Skip={'Y' if cache_update_skipped else 'N'})"
        elif enable_stateless_refin_global and stateless_refinement_performed: refinement_indicator = "StatelessRef"
        if refinement_indicator: status_parts.append(refinement_indicator)

        # Scene status based on flag
        scene_status = "Scene=NEW" if scene_changed_flag else "Scene=OK"
        status_parts.append(scene_status)

        # Token reporting uses values from status_info and direct args
        self.logger.debug(f"[{session_id}] Status Tokens Check: OWI={initial_owi_context_tokens}, Ref={refined_context_tokens}, Hist={t0_dialogue_tokens}, Inv={inventory_prompt_tokens}, Final={final_payload_tokens}")
        token_parts = []
        if initial_owi_context_tokens >= 0: token_parts.append(f"OWI={initial_owi_context_tokens}")
        if refined_context_tokens >= 0: token_parts.append(f"Ref={refined_context_tokens}")
        if t0_dialogue_tokens >= 0: token_parts.append(f"Hist={t0_dialogue_tokens}")
        if inventory_prompt_tokens >= 0: token_parts.append(f"Inv={inventory_prompt_tokens}")
        if final_payload_tokens >= 0: token_parts.append(f"Final={final_payload_tokens}")
        token_string = ""
        if token_parts: self.logger.debug(f"[{session_id}] Status Tokens: Adding token section: {token_parts}"); token_string = f" | Tok: {' '.join(token_parts)}"
        else: self.logger.debug(f"[{session_id}] Status Tokens: Skipping token section (no valid tokens >= 0).")

        status_message = ", ".join(status_parts) + world_state_status_str + token_string
        return status_message, final_payload_tokens


    # === _execute_or_prepare_output ===
    async def _execute_or_prepare_output(
        self, session_id: str, body: Dict, final_llm_payload_contents: Optional[List[Dict]],
        event_emitter: Optional[Callable], status_message: str, final_payload_tokens: int # final_payload_tokens not used here but part of signature
    ) -> OrchestratorResult:
        """ Executes final LLM call if configured, otherwise returns constructed payload. """
        # [[ Restored Full Implementation ]]
        output_body = body.copy() if isinstance(body, dict) else {}
        if not final_llm_payload_contents:
            self.logger.error(f"[{session_id}] Final payload construction failed (input to _execute_or_prepare_output was None).")
            await self._emit_status(event_emitter, session_id, "ERROR: Final payload preparation failed.", done=True)
            return {"error": "Orchestrator: Final payload construction failed.", "status_code": 500}

        output_body["messages"] = final_llm_payload_contents # Use the generated payload

        # Preserve relevant keys from original request body for pass-through
        preserved_keys = ["model", "stream", "options", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"]
        keys_preserved = [k for k in preserved_keys if k in body]
        for k in keys_preserved:
            output_body[k] = body[k]
        self.logger.debug(f"[{session_id}] Output body constructed/updated. Preserved keys: {keys_preserved}.")

        # Debug log the final payload if enabled
        if getattr(self.config, 'debug_log_final_payload', False):
            self.logger.debug(f"[{session_id}] Logging final constructed payload dict due to debug valve.")
            self._orchestrator_log_debug_payload(session_id, {"contents": final_llm_payload_contents})
        else: self.logger.debug(f"[{session_id}] Skipping final payload log: Debug valve is OFF.")

        # Check if final LLM call is configured
        final_url = getattr(self.config, 'final_llm_api_url', None)
        final_key = getattr(self.config, 'final_llm_api_key', None)
        url_present = bool(final_url and isinstance(final_url, str) and final_url.strip())
        key_present = bool(final_key and isinstance(final_key, str) and final_key.strip())
        self.logger.debug(f"[{session_id}] Checking Final LLM Trigger. URL Present:{url_present}, Key Present:{key_present}")
        final_llm_triggered = url_present and key_present

        if final_llm_triggered:
            # Execute the final LLM call using the wrapper
            self.logger.info(f"[{session_id}] Final LLM Call via Pipe TRIGGERED (Non-Streaming, using Adapter).")
            await self._emit_status(event_emitter, session_id, "Status: Executing final LLM Call...", done=False)
            final_temp = getattr(self.config, 'final_llm_temperature', 0.7)
            final_timeout = getattr(self.config, 'final_llm_timeout', 120)
            # Payload needs to be in the format expected by the adapter (e.g., Google format)
            final_call_payload_google_fmt = {"contents": final_llm_payload_contents}

            success, response_or_error = await self._async_llm_call_wrapper(
                api_url=final_url, api_key=final_key,
                payload=final_call_payload_google_fmt,
                temperature=final_temp, timeout=final_timeout,
                caller_info=f"Orch_FinalLLM_{session_id}"
            )

            intermediate_status = "Status: Final LLM Complete" + (" (Success)" if success else " (Failed)")
            await self._emit_status(event_emitter, session_id, intermediate_status, done=False)

            if success and isinstance(response_or_error, str):
                self.logger.info(f"[{session_id}] Final LLM call successful. Returning response string.")
                return response_or_error # Return the text response
            elif not success and isinstance(response_or_error, dict):
                self.logger.error(f"[{session_id}] Final LLM call failed. Returning error dict: {response_or_error}")
                return response_or_error # Return error dict
            else:
                # Handle unexpected return type from adapter
                self.logger.error(f"[{session_id}] Final LLM adapter returned unexpected format. Success={success}, Type={type(response_or_error)}")
                return {"error": "Final LLM adapter returned unexpected result format.", "status_code": 500}
        else:
            # Final LLM call disabled, return the constructed payload dictionary for OWI
            self.logger.info(f"[{session_id}] Final LLM Call disabled by config. Returning constructed payload dict.")
            # IMPORTANT: OWI expects the 'messages' key
            return {"messages": final_llm_payload_contents}


# === MAIN PROCESSING METHOD (MODIFIED to call context processor) ===
    async def process_turn(
        self,
        session_id: str,
        user_id: str,
        body: Dict,
        user_valves: Any, # Expects an object with user valve attributes
        event_emitter: Optional[Callable],
        embedding_func: Optional[Callable[[Sequence[str], str, Optional[Dict]], List[List[float]]]] = None, # OWI embedding func type hint
        chroma_embed_wrapper: Optional[Any] = None,
        is_regeneration_heuristic: bool = False
    ) -> OrchestratorResult:
        """
        Processes a single turn coordinating memory, state, context processing,
        inventory, hints, and final LLM calls.
        Delegates context prep/payload construction to context_processor module.
        """
        # [[ Implementation uses the restored helpers and calls context_processor ]]
        pipe_entry_time_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Orchestrator process_turn [{session_id}]: Started at {pipe_entry_time_iso} (Regen Flag: {is_regeneration_heuristic})")
        self.pipe_logger = getattr(self, 'pipe_logger', self.logger); self.pipe_debug_path_getter = self._orchestrator_get_debug_log_path # Use internal getter

        inventory_enabled = getattr(self.config, 'enable_inventory_management', False); event_hints_enabled = getattr(self.config, 'enable_event_hints', False)
        self.logger.info(f"[{session_id}] Inventory Mgmt Enabled: {inventory_enabled} (Module Avail: {_ORCH_INVENTORY_MODULE_AVAILABLE})")
        self.logger.info(f"[{session_id}] Event Hints Enabled: {event_hints_enabled} (Module Avail: {_EVENT_HINTS_AVAILABLE})")
        self.logger.info(f"[{session_id}] State Update Method: Full Turn Assessment LLM (Available: {_UNIFIED_STATE_ASSESSMENT_AVAILABLE})")
        self.logger.info(f"[{session_id}] Context Processor: Available={_CONTEXT_PROCESSOR_AVAILABLE}")

        session_period_setting = getattr(user_valves, 'period_setting', '').strip()
        if session_period_setting: self.logger.info(f"[{session_id}] Using Period Setting from User Valves: '{session_period_setting}'")
        else: self.logger.debug(f"[{session_id}] No Period Setting provided in User Valves.")

        # --- Initialize state variables ---
        summarization_performed = False; new_t1_summary_text = None; summarization_prompt_tokens = -1; summarization_output_tokens = -1; t1_retrieved_count = 0;
        t0_dialogue_tokens = -1; final_payload_tokens = -1; inventory_prompt_tokens = -1;
        final_result: Optional[OrchestratorResult] = None; final_llm_payload_contents: Optional[List[Dict]] = None;
        inventory_update_completed = False; inventory_update_success_flag = False;
        generated_event_hint_text: Optional[str] = None
        generated_weather_proposal: Dict[str, Optional[str]] = {}
        scene_changed_flag = False

        # World/Scene state variables
        default_season = "Summer"; default_weather = "Clear"; default_day = 1; default_time = "Morning";
        self.current_season: Optional[str] = default_season
        self.current_weather: Optional[str] = default_weather
        self.current_day: Optional[int] = default_day
        self.current_time_of_day: Optional[str] = default_time
        current_scene_state_dict: Dict = {"keywords": [], "description": ""}

        # --- Context processor status ---
        context_status_info: Dict[str, Any] = {}

        try:
            # --- History Sync ---
            await self._emit_status(event_emitter, session_id, "Status: Orchestrator syncing history...")
            incoming_messages = body.get("messages", []); stored_history = self.session_manager.get_active_history(session_id) or []
            if incoming_messages != stored_history: self.session_manager.set_active_history(session_id, incoming_messages.copy()); self.logger.debug(f"[{session_id}] Updating active history (Len: {len(incoming_messages)}).")
            else: self.logger.debug(f"[{session_id}] Incoming history matches stored.")
            current_active_history = self.session_manager.get_active_history(session_id) or []
            if not current_active_history: raise ValueError("Active history is empty after sync.")

            # --- Determine Query, History Slice ---
            latest_user_query_str, history_for_processing, previous_llm_response_str = await self._determine_effective_query( session_id, current_active_history, is_regeneration_heuristic )
            if latest_user_query_str is None: # Check specifically for None return from helper on error
                raise ValueError("Failed to determine effective query or history.")
            if not latest_user_query_str and not is_regeneration_heuristic:
                raise ValueError("Cannot proceed without an effective user query (and not regeneration).")
            safe_previous_llm_response_str = previous_llm_response_str if previous_llm_response_str is not None else ""

            # --- Fetch Initial World & Scene State ---
            await self._emit_status(event_emitter, session_id, "Status: Fetching initial world state...")
            initial_world_state_db_data = await self._get_world_state_db_func(self.sqlite_cursor, session_id) if self.sqlite_cursor and self._get_world_state_db_func else None
            if initial_world_state_db_data and isinstance(initial_world_state_db_data, dict):
                self.current_season = initial_world_state_db_data.get("season", default_season)
                self.current_weather = initial_world_state_db_data.get("weather", default_weather)
                self.current_day = initial_world_state_db_data.get("day", default_day)
                self.current_time_of_day = initial_world_state_db_data.get("time_of_day", default_time)
                self.logger.debug(f"[{session_id}] Fetched initial world state: D{self.current_day}, {self.current_time_of_day}, {self.current_weather}, {self.current_season}")
            else:
                self.logger.debug(f"[{session_id}] No world state found or error fetching. Using defaults.")
                self.current_season, self.current_weather, self.current_day, self.current_time_of_day = default_season, default_weather, default_day, default_time
            initial_world_state_dict = {"day": self.current_day, "time_of_day": self.current_time_of_day, "weather": self.current_weather, "season": self.current_season} # Store for context processor

            await self._emit_status(event_emitter, session_id, "Status: Fetching initial scene state...")
            initial_scene_state_db_data = await self._get_scene_state_db_func(self.sqlite_cursor, session_id) if self.sqlite_cursor and self._get_scene_state_db_func else None
            if initial_scene_state_db_data and isinstance(initial_scene_state_db_data, dict):
                kw_json = initial_scene_state_db_data.get("keywords_json"); desc = initial_scene_state_db_data.get("description", "")
                try: keywords = json.loads(kw_json) if isinstance(kw_json, str) else []
                except json.JSONDecodeError: keywords = []
                current_scene_state_dict = {"keywords": keywords if isinstance(keywords, list) else [], "description": desc if isinstance(desc, str) else ""}
                self.logger.debug(f"[{session_id}] Fetched initial scene state. Desc len: {len(current_scene_state_dict['description'])}")
            else:
                self.logger.debug(f"[{session_id}] No scene state found or error fetching. Using empty defaults.")
                current_scene_state_dict = {"keywords": [], "description": ""}

            # --- Memory Management (T1/T2) ---
            (summarization_performed, new_t1_summary_text, summarization_prompt_tokens, summarization_output_tokens) = await self._handle_tier1_summarization( session_id, user_id, current_active_history, is_regeneration_heuristic, event_emitter )
            await self._handle_tier2_transition( session_id, summarization_performed, chroma_embed_wrapper, event_emitter )
            recent_t1_summaries, t1_retrieved_count = await self._get_t1_summaries(session_id)

            # --- Generate Hint AND Weather Proposal ---
            hint_background_context = current_scene_state_dict.get("description", "")
            hint_background_context += f"\n(Day: {self.current_day}, Time: {self.current_time_of_day}, Weather: {self.current_weather})"
            generated_weather_proposal = {}

            if event_hints_enabled and self._generate_hint_func:
                self.logger.debug(f"[{session_id}] Attempting event hint generation (Period: '{session_period_setting}')...")
                await self._emit_status(event_emitter, session_id, "Status: Generating hint...")
                hint_llm_url = getattr(self.config, 'event_hint_llm_api_url', None)
                hint_llm_key = getattr(self.config, 'event_hint_llm_api_key', None)
                if not hint_llm_url or not hint_llm_key:
                     self.logger.warning(f"[{session_id}] Skipping hint: Config incomplete (event_hint_llm_...).")
                else:
                    try:
                        generated_event_hint_text, generated_weather_proposal = await self._generate_hint_func(
                            config=self.config, history_messages=current_active_history,
                            background_context=hint_background_context, current_season=self.current_season,
                            current_weather=self.current_weather, current_time_of_day=self.current_time_of_day,
                            llm_call_func=self._async_llm_call_wrapper, logger_instance=self.logger,
                            session_id=session_id, period_setting=session_period_setting
                        )
                        if generated_event_hint_text: self.logger.info(f"[{session_id}] Event Hint Generated: '{generated_event_hint_text[:80]}...'")
                        else: self.logger.info(f"[{session_id}] No event hint suggested.")
                        if generated_weather_proposal and generated_weather_proposal.get("new_weather"): self.logger.info(f"[{session_id}] Weather Proposal Received: From '{generated_weather_proposal.get('previous_weather')}' to '{generated_weather_proposal.get('new_weather')}'")
                        else: self.logger.debug(f"[{session_id}] No valid weather proposal received from hint system.")
                        await self._emit_status(event_emitter, session_id, "Status: Hint generation complete.")
                    except Exception as e_hint_gen:
                        self.logger.error(f"[{session_id}] Error during hint generation call: {e_hint_gen}", exc_info=True);
                        generated_event_hint_text = None; generated_weather_proposal = {}
            elif event_hints_enabled and not self._generate_hint_func: self.logger.error(f"[{session_id}] Skipping hint: Hint function unavailable.")
            else: self.logger.debug(f"[{session_id}] Event Hints disabled. Skipping.")

            # --- Select T0 History Slice ---
            # This gets the correctly token-limited slice
            t0_raw_history_slice, t0_dialogue_tokens = await self._select_t0_history_slice( session_id, history_for_processing )

            # --- Call Context Processor ---
            if not self._context_processor_func:
                raise RuntimeError("Context processor function is unavailable.")

            await self._emit_status(event_emitter, session_id, "Status: Processing background context...")
            # === MODIFIED CALL: Pass t0_raw_history_slice ===
            final_llm_payload_contents, context_status_info = await self._context_processor_func(
                session_id=session_id, body=body, user_valves=user_valves,
                current_active_history=current_active_history,
                history_for_processing=history_for_processing, # Still pass pre-query slice for T2/Refinement context
                t0_history_slice=t0_raw_history_slice,        # Pass the CORRECT slice for final payload
                latest_user_query_str=latest_user_query_str,
                recent_t1_summaries=recent_t1_summaries,
                current_scene_state_dict=current_scene_state_dict,
                current_world_state_dict=initial_world_state_dict,
                generated_event_hint_text=generated_event_hint_text,
                generated_weather_proposal=generated_weather_proposal, # Pass weather proposal
                config=self.config, logger=self.logger,
                sqlite_cursor=self.sqlite_cursor,
                chroma_client=self.chroma_client,
                chroma_embed_wrapper=chroma_embed_wrapper,
                embedding_func=embedding_func,
                llm_call_func=self._async_llm_call_wrapper,
                tokenizer=self._tokenizer,
                event_emitter=event_emitter,
                orchestrator_debug_path_getter=self._orchestrator_get_debug_log_path,
                dialogue_roles=self._dialogue_roles,
                session_period_setting=session_period_setting,
            )
            # === END MODIFIED CALL ===
            await self._emit_status(event_emitter, session_id, "Status: Context processing complete.")

            if context_status_info.get("error"):
                self.logger.error(f"[{session_id}] Error reported from context processor: {context_status_info['error']}")
            if final_llm_payload_contents is None:
                 self.logger.error(f"[{session_id}] Context processor failed to return payload contents.")
                 return {"error": "Failed to prepare LLM payload.", "status_code": 500}

            # --- Execute Final LLM Call or Prepare Output ---
            final_result = await self._execute_or_prepare_output(
                 session_id=session_id, body=body,
                 final_llm_payload_contents=final_llm_payload_contents,
                 event_emitter=event_emitter, status_message="Status: Core processing complete.",
                 final_payload_tokens=-1 # Tokens calculated later
                 )

            # --- Post-Turn Unified State Update ---
            new_state_dict = None
            if isinstance(final_result, str) and self._unified_state_func:
                narrative_response_text = final_result
                self.logger.info(f"[{session_id}] Performing post-turn unified state assessment...")
                await self._emit_status(event_emitter, session_id, "Status: Assessing state changes...", done=False)
                previous_world_state_for_assessment = {"day": self.current_day, "time_of_day": self.current_time_of_day, "weather": self.current_weather, "season": self.current_season }
                previous_scene_state_for_assessment = current_scene_state_dict
                try:
                    state_assess_llm_config = { "url": getattr(self.config, 'event_hint_llm_api_url', None), "key": getattr(self.config, 'event_hint_llm_api_key', None), "temp": getattr(self.config, 'state_assess_llm_temperature', 0.3), "prompt_template": DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT }
                    if not state_assess_llm_config["url"] or not state_assess_llm_config["key"]: self.logger.error(f"[{session_id}] State Assessment LLM URL/Key missing. Skipping state update."); new_state_dict = None
                    else:
                        new_state_dict = await self._unified_state_func( session_id=session_id, previous_world_state=previous_world_state_for_assessment, previous_scene_state=previous_scene_state_for_assessment, current_user_query=latest_user_query_str, assistant_response_text=narrative_response_text, history_messages=current_active_history, llm_call_func=self._async_llm_call_wrapper, state_assessment_llm_config=state_assess_llm_config, logger_instance=self.logger, event_emitter=event_emitter, weather_proposal=generated_weather_proposal )
                    if new_state_dict and isinstance(new_state_dict, dict):
                        old_day = self.current_day; old_time = self.current_time_of_day; old_weather = self.current_weather; old_season = self.current_season
                        self.current_day = new_state_dict.get("new_day", self.current_day)
                        self.current_time_of_day = new_state_dict.get("new_time_of_day", self.current_time_of_day)
                        self.current_weather = new_state_dict.get("new_weather", self.current_weather)
                        self.current_season = new_state_dict.get("new_season", self.current_season)
                        new_scene_keywords = new_state_dict.get("new_scene_keywords", current_scene_state_dict.get("keywords",[]))
                        new_scene_description = new_state_dict.get("new_scene_description", current_scene_state_dict.get("description",""))
                        scene_changed_flag = new_state_dict.get("scene_changed_flag", False)
                        current_scene_state_dict = {"keywords": new_scene_keywords, "description": new_scene_description }
                        world_state_changed = ( self.current_day != old_day or self.current_time_of_day != old_time or self.current_weather != old_weather or self.current_season != old_season )
                        scene_state_changed = scene_changed_flag
                        self.logger.debug(f"[{session_id}] Unified state assessment complete.")
                        if world_state_changed: self.logger.info(f"[{session_id}] New World State: D{self.current_day} {self.current_time_of_day} {self.current_weather} {self.current_season}")
                        else: self.logger.info(f"[{session_id}] No world state change detected.")
                        if scene_state_changed: self.logger.info(f"[{session_id}] New Scene State: Desc len {len(new_scene_description)}, Keywords {new_scene_keywords}")
                        else: self.logger.info(f"[{session_id}] No scene state change detected.")
                        # Save Updated World State
                        if world_state_changed and self.sqlite_cursor and self._set_world_state_db_func:
                             await self._emit_status(event_emitter, session_id, "Status: Saving world state...", done=False)
                             try:
                                 update_success = await self._set_world_state_db_func( self.sqlite_cursor, session_id, self.current_season, self.current_weather, self.current_day, self.current_time_of_day )
                                 if update_success: self.logger.info(f"[{session_id}] World state successfully saved.")
                                 else: self.logger.error(f"[{session_id}] Failed to save world state.")
                             except Exception as e_set_world: self.logger.error(f"[{session_id}] Error saving world state: {e_set_world}", exc_info=True)
                        # Save Updated Scene State
                        if scene_state_changed and self.sqlite_cursor and self._set_scene_state_db_func:
                             await self._emit_status(event_emitter, session_id, "Status: Saving scene state...", done=False)
                             try:
                                 kw_json_to_save = json.dumps(new_scene_keywords)
                                 update_success = await self._set_scene_state_db_func( self.sqlite_cursor, session_id, kw_json_to_save, new_scene_description )
                                 if update_success: self.logger.info(f"[{session_id}] Scene state successfully saved.")
                                 else: self.logger.error(f"[{session_id}] Failed to save scene state.")
                             except Exception as e_set_scene: self.logger.error(f"[{session_id}] Error saving scene state: {e_set_scene}", exc_info=True)
                    else: self.logger.error(f"[{session_id}] Unified state assessment returned invalid data."); scene_changed_flag = False
                except Exception as e_assess: self.logger.error(f"[{session_id}] Exception during unified state assessment call: {e_assess}", exc_info=True); scene_changed_flag = False
            elif not self._unified_state_func: self.logger.error(f"[{session_id}] Skipping state assessment: Unified state function unavailable."); scene_changed_flag = False
            elif not isinstance(final_result, str): self.logger.debug(f"[{session_id}] Skipping state assessment: Final result was not string."); scene_changed_flag = False


            # --- Post-Turn Inventory Update ---
            if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and self._update_inventories_func:
                inventory_update_completed = True
                if isinstance(final_result, str):
                    self.logger.debug(f"[{session_id}] Performing post-turn inventory update...")
                    await self._emit_status(event_emitter, session_id, "Status: Updating inventory state...", done=False)
                    inv_llm_url = getattr(self.config, 'inv_llm_api_url', None); inv_llm_key = getattr(self.config, 'inv_llm_api_key', None);
                    inv_llm_prompt_template = DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT
                    template_seems_valid = inv_llm_prompt_template != "[Default Inventory Prompt Load Failed]"

                    if not inv_llm_url or not inv_llm_key or not template_seems_valid: self.logger.error(f"[{session_id}] Inventory LLM config missing/invalid."); inventory_update_success_flag = False
                    else:
                        inv_llm_config = {"url": inv_llm_url, "key": inv_llm_key, "temp": getattr(self.config, 'inv_llm_temperature', 0.3), "prompt_template": inv_llm_prompt_template}
                        history_for_inv_update_list = self._get_recent_turns_func(current_active_history, 4, exclude_last=False); history_for_inv_update_str = self._format_history_func(history_for_inv_update_list)
                        inv_prompt_text = format_inventory_update_prompt( main_llm_response=final_result, user_query=latest_user_query_str, recent_history_str=history_for_inv_update_str, template=inv_llm_config['prompt_template'])
                        if inv_prompt_text and not inv_prompt_text.startswith("[Error") and self._count_tokens_func and self._tokenizer:
                            try: inventory_prompt_tokens = self._count_tokens_func(inv_prompt_text, self._tokenizer); self.logger.debug(f"[{session_id}] Calculated Inventory Prompt Tokens: {inventory_prompt_tokens}")
                            except Exception as e_inv_tok: inventory_prompt_tokens = -1
                        else: inventory_prompt_tokens = -1
                        if not self.sqlite_cursor or not self.sqlite_cursor.connection: self.logger.error(f"[{session_id}] Cannot update inventory: SQLite cursor invalid."); inventory_update_success_flag = False
                        else:
                             new_cursor = None
                             try:
                                 new_cursor = self.sqlite_cursor.connection.cursor()
                                 if getattr(self.config, 'debug_log_final_payload', False): self._orchestrator_log_debug_inventory_llm(session_id, inv_prompt_text, is_prompt=True)
                                 update_success = await self._update_inventories_func( cursor=new_cursor, session_id=session_id, main_llm_response=final_result, user_query=latest_user_query_str, recent_history_str=history_for_inv_update_str, llm_call_func=self._async_llm_call_wrapper, db_get_inventory_func=get_character_inventory_data, db_update_inventory_func=add_or_update_character_inventory, inventory_llm_config=inv_llm_config,)
                                 inventory_update_success_flag = update_success
                                 if update_success: self.logger.info(f"[{session_id}] Post-turn inventory update successful.")
                                 else: self.logger.warning(f"[{session_id}] Post-turn inventory update function returned False.")
                             except Exception as e_inv_update_inner: self.logger.error(f"[{session_id}] Error during inventory update call: {e_inv_update_inner}", exc_info=True); inventory_update_success_flag = False
                             finally:
                                  if new_cursor:
                                      try: new_cursor.close(); self.logger.debug(f"[{session_id}] Inventory update cursor closed.")
                                      except Exception as e_close_cursor: self.logger.error(f"[{session_id}] Error closing inventory update cursor: {e_close_cursor}")
                elif isinstance(final_result, dict) and "error" in final_result: self.logger.warning(f"[{session_id}] Skipping inventory update due to upstream error."); inventory_update_completed = False
                elif isinstance(final_result, dict) and "messages" in final_result: self.logger.debug(f"[{session_id}] Skipping inventory update: Final LLM call disabled."); inventory_update_completed = False
                else: self.logger.error(f"[{session_id}] Unexpected type for final_result. Skipping inventory update."); inventory_update_completed = False
            elif inventory_enabled and not _ORCH_INVENTORY_MODULE_AVAILABLE: self.logger.warning(f"[{session_id}] Skipping inventory update: Module import failed."); inventory_update_completed = False
            elif inventory_enabled and not self._update_inventories_func: self.logger.error(f"[{session_id}] Skipping inventory update: Update function alias None."); inventory_update_completed = False
            else: self.logger.debug(f"[{session_id}] Skipping inventory update: Disabled by global valve."); inventory_update_completed = False


            # --- Final Status Calculation and Emission ---
            world_state_status_str = f"| World: D{self.current_day} {self.current_time_of_day} {self.current_weather} {self.current_season}"
            inv_stat_indicator = "Inv=OFF";
            if inventory_enabled:
                 if not _ORCH_INVENTORY_MODULE_AVAILABLE: inv_stat_indicator = "Inv=MISSING"
                 else: inv_stat_indicator = "Inv=OK" if inventory_update_success_flag else ("Inv=FAIL" if inventory_update_completed else "Inv=SKIP")

            final_status_string, final_payload_tokens = await self._calculate_and_format_status(
                 session_id=session_id,
                 t1_retrieved_count=t1_retrieved_count,
                 summarization_prompt_tokens=summarization_prompt_tokens,
                 summarization_output_tokens=summarization_output_tokens,
                 t0_dialogue_tokens=t0_dialogue_tokens,
                 inventory_prompt_tokens=inventory_prompt_tokens,
                 final_llm_payload_contents=final_llm_payload_contents,
                 scene_changed_flag=scene_changed_flag,
                 world_state_status_str=world_state_status_str,
                 context_status_info=context_status_info, # Pass the dict here
                 session_process_owi_rag=bool(getattr(user_valves, 'process_owi_rag', True)),
            )

            final_status_string += f" | {inv_stat_indicator}" # Append inventory status
            self.logger.info(f"[{session_id}] Orchestrator FINAL STATUS: {final_status_string}") # Keep INFO
            await self._emit_status(event_emitter, session_id, final_status_string, done=True)

            pipe_end_time_iso = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"Orchestrator process_turn [{session_id}]: Finished at {pipe_end_time_iso}") # Keep INFO
            if final_result is None: raise RuntimeError("Internal processing error, final result was None.")
            return final_result

        # --- Exception Handling ---
        except asyncio.CancelledError:
            self.logger.info(f"[{session_id or 'unknown'}] Orchestrator process_turn cancelled.")
            await self._emit_status(event_emitter, session_id or 'unknown', "Status: Processing cancelled.", done=True)
            raise
        except ValueError as ve:
            session_id_for_log = session_id if 'session_id' in locals() else 'unknown'
            self.logger.error(f"[{session_id_for_log}] Orchestrator ValueError in process_turn: {ve}", exc_info=True) # Added exc_info
            try: await self._emit_status(event_emitter, session_id_for_log, f"ERROR: {ve}", done=True)
            except Exception: pass
            return {"error": f"Orchestrator failed: {ve}", "status_code": 500}
        except Exception as e_orch:
            session_id_for_log = session_id if 'session_id' in locals() else 'unknown'
            self.logger.critical(f"[{session_id_for_log}] Orchestrator UNHANDLED EXCEPTION in process_turn: {e_orch}", exc_info=True)
            try: await self._emit_status(event_emitter, session_id_for_log, f"ERROR: Orchestrator Failed ({type(e_orch).__name__})", done=True)
            except Exception: pass
            # Return a dict for OWI error handling
            return {"error": f"Orchestrator failed: {type(e_orch).__name__}", "status_code": 500}

# [[END CORRECTED orchestration.py - Restore Helper Logic]]
# === END OF FILE i4_llm_agent/orchestration.py ===