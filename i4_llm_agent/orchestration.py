
# === START OF FILE i4_llm_agent/orchestration.py ===
# i4_llm_agent/orchestration.py

import logging
import asyncio
import re
import sqlite3
import json
import uuid
import os # Added for path manipulation in helpers
from datetime import datetime, timezone
from typing import ( # Ensure Coroutine is imported
    Tuple, Union, List, Dict, Optional, Any, Callable, Coroutine, AsyncGenerator
)
import importlib.util # Added for import debugging

# --- Standard Library Imports ---
import urllib.parse

# --- i4_llm_agent Imports (Careful with internal package imports) ---
from .session import SessionManager
from .database import (
    add_tier1_summary, get_recent_tier1_summaries, get_tier1_summary_count,
    get_oldest_tier1_summary, delete_tier1_summary, get_max_t1_end_index,
    add_or_update_rag_cache, get_rag_cache,
    get_or_create_chroma_collection, add_to_chroma_collection,
    query_chroma_collection, get_chroma_collection_count,
    CHROMADB_AVAILABLE, ChromaEmbeddingFunction, ChromaCollectionType,
    InvalidDimensionException,
    get_all_inventories_for_session, get_character_inventory_data,
    add_or_update_character_inventory,
    get_world_state, # Added
    set_world_state, # Added (for initialization)
)
from .history import (
    format_history_for_llm, get_recent_turns, DIALOGUE_ROLES, select_turns_for_t0
)
from .memory import manage_tier1_summarization
from .prompting import (
    # format functions (used by aliases or directly)
    format_inventory_update_prompt,
    # standalone functions (used by aliases)
    construct_final_llm_payload as standalone_construct_payload, # Alias the standalone one
    clean_context_tags, generate_rag_query,
    combine_background_context, process_system_prompt,
    refine_external_context, format_stateless_refiner_prompt,
    format_cache_update_prompt, format_final_context_selection_prompt,
    # Default templates (used by aliases or config)
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
)
from .cache import update_rag_cache, select_final_context
# --- MODIFIED IMPORT: Only import the main dispatcher (now litellm adapter) ---
from .api_client import call_google_llm_api
# --- REMOVED IMPORT: No longer needed as adapter handles conversion ---
# from .api_client import _convert_google_to_openai_payload
from .utils import count_tokens, calculate_string_similarity, TIKTOKEN_AVAILABLE

# --- MODIFIED IMPORT: Import the new LLM-based parser ---
try:
    # <<< MODIFIED: Import the constant used in orchestration config checks >>>
    from .world_state_parser import parse_world_state_with_llm, DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT
    _WORLD_STATE_PARSER_LLM_AVAILABLE = True
except ImportError as e_world_parser:
    # Define a dummy async function if the parser module is not found
    async def parse_world_state_with_llm(*args, **kwargs) -> Dict:
        logging.getLogger(__name__).error("Executing FALLBACK parse_world_state_with_llm due to import error.")
        await asyncio.sleep(0) # Need await for async definition
        return {}
    # <<< MODIFIED: Ensure constant exists even if import fails >>>
    DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT = "[Default World State Parse Template Load Failed]"
    _WORLD_STATE_PARSER_LLM_AVAILABLE = False
    logging.getLogger(__name__).error(
        f"Failed to import world_state_parser (LLM version) in orchestration.py: {e_world_parser}. State change detection disabled.", exc_info=True
    )
# --- END MODIFIED WORLD STATE IMPORT ---


# --- MODIFIED: Import Event Hint logic with Debugging ---
_event_hints_import_success = False
_event_hints_import_path = "Unknown"
try:
    # Attempt the import
    from .event_hints import generate_event_hint, EVENT_HANDLING_GUIDELINE_TEXT, format_hint_for_query
    _event_hints_import_success = True
    # Try to find the spec to log the origin file path
    try:
        # Use the package name explicitly if known (it's i4_llm_agent)
        spec = importlib.util.find_spec(".event_hints", package="i4_llm_agent")
        if spec and spec.origin:
            _event_hints_import_path = spec.origin
            logging.getLogger(__name__).info(f"Successfully imported generate_event_hint from: {_event_hints_import_path}")
        else:
            logging.getLogger(__name__).warning("Successfully imported generate_event_hint, but couldn't determine file origin (spec missing?).")
    except Exception as e_spec:
        logging.getLogger(__name__).warning(f"Successfully imported generate_event_hint, but error finding spec: {e_spec}")

except ImportError as e_import:
    # Log the specific import error
    logging.getLogger(__name__).error(f"Failed to import from .event_hints: {e_import}", exc_info=True)
    # Define fallbacks if import fails
    async def generate_event_hint(*args, **kwargs):
        logging.getLogger(__name__).error("Executing FALLBACK generate_event_hint due to import error.")
        return None
    EVENT_HANDLING_GUIDELINE_TEXT = "[EVENT GUIDELINE LOAD FAILED]"
    def format_hint_for_query(hint): return f"[[Hint Load Failed: {hint}]]"

except Exception as e_other:
    # Catch any other unexpected errors during import
     logging.getLogger(__name__).error(f"Unexpected error during .event_hints import: {e_other}", exc_info=True)
     async def generate_event_hint(*args, **kwargs):
        logging.getLogger(__name__).error("Executing FALLBACK generate_event_hint due to unexpected import error.")
        return None
     EVENT_HANDLING_GUIDELINE_TEXT = "[EVENT GUIDELINE LOAD FAILED - UNEXPECTED]"
     def format_hint_for_query(hint): return f"[[Hint Load Failed: {hint}]]"
# --- END MODIFIED EVENT HINTS IMPORT ---


# --- Inventory Module Import (LOCAL TO ORCHESTRATION) ---
try:
    from .inventory import (
        format_inventory_for_prompt as _real_format_inventory_func,
        update_inventories_from_llm as _real_update_inventories_func,
    )
    _ORCH_INVENTORY_MODULE_AVAILABLE = True
    _dummy_format_inventory = None
    _dummy_update_inventories = None
except ImportError:
    _ORCH_INVENTORY_MODULE_AVAILABLE = False
    _real_format_inventory_func = None
    _real_update_inventories_func = None
    def _dummy_format_inventory(*args, **kwargs): return "[Inventory Module Unavailable]"
    async def _dummy_update_inventories(*args, **kwargs): await asyncio.sleep(0); return False
    logging.getLogger(__name__).warning(
        "Orchestration: Inventory module not found. Inventory features disabled within orchestrator."
        )
# --- END LOCAL Inventory Import ---


# --- REMOVED Optional httpx Import ---
# No longer needed for streaming

logger = logging.getLogger(__name__) # i4_llm_agent.orchestration

# --- MODIFIED Type Alias: Removed AsyncGenerator ---
OrchestratorResult = Union[Dict, str]

# ==============================================================================
# === Session Pipe Orchestrator Class (Modularized)                          ===
# ==============================================================================

class SessionPipeOrchestrator:
    """
    Orchestrates the core processing logic of the Session Memory Pipe.
    Encapsulates logic in helper methods for clarity and maintainability.
    Includes inventory management and event hint features.
    Handles final payload debug logging.
    Uses LiteLLM adapter via api_client for LLM calls. NO STREAMING.
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
        self.logger = logger_instance or logger # Use passed logger or default
        self.pipe_logger = logger_instance or logger # Explicitly store pipe logger if passed
        self.pipe_debug_path_getter = None # Placeholder for path getter from Pipe

        self._tokenizer = None
        if TIKTOKEN_AVAILABLE and hasattr(self.config, 'tokenizer_encoding_name'):
            try:
                self.logger.info(f"Orchestrator: Initializing tokenizer '{self.config.tokenizer_encoding_name}'...")
                import tiktoken # Import locally if not already top-level
                self._tokenizer = tiktoken.get_encoding(self.config.tokenizer_encoding_name)
                self.logger.info("Orchestrator: Tokenizer initialized.")
            except Exception as e:
                self.logger.error(f"Orchestrator: Tokenizer init failed: {e}. Token counting disabled.", exc_info=True)
        elif not TIKTOKEN_AVAILABLE:
             self.logger.warning("Orchestrator: tiktoken unavailable. Token counting disabled.")

        # --- Function Aliases ---
        # Core
        self._llm_call_func = call_google_llm_api
        self._format_history_func = format_history_for_llm
        self._get_recent_turns_func = get_recent_turns
        self._manage_memory_func = manage_tier1_summarization
        self._count_tokens_func = count_tokens
        self._calculate_similarity_func = calculate_string_similarity
        self._dialogue_roles = DIALOGUE_ROLES
        # Prompting/Context
        self._clean_context_tags_func = clean_context_tags
        self._generate_rag_query_func = generate_rag_query
        self._combine_context_func = combine_background_context # Will be modified later
        self._process_system_prompt_func = process_system_prompt
        # Refinement/Cache Orchestrators
        self._stateless_refine_func = refine_external_context
        self._cache_update_func = update_rag_cache
        self._cache_select_func = select_final_context
        # DB Getters/Setters (Specific)
        self._get_rag_cache_db_func = get_rag_cache
        self._get_all_inventories_db_func = get_all_inventories_for_session
        self._get_char_inventory_db_func = get_character_inventory_data
        self._update_char_inventory_db_func = add_or_update_character_inventory
        self._get_world_state_db_func = get_world_state  # Already added
        self._set_world_state_db_func = set_world_state  # Already added

        # --- MODIFIED: Alias the new LLM-based parser ---
        if _WORLD_STATE_PARSER_LLM_AVAILABLE:
            self._parse_world_state_func = parse_world_state_with_llm
        else:
            # Assign the async fallback if import failed
            self._parse_world_state_func = parse_world_state_with_llm # Which is the dummy async func
        # --- END MODIFICATION ---

        # Inventory Module Functions
        if _ORCH_INVENTORY_MODULE_AVAILABLE:
            self._format_inventory_func = _real_format_inventory_func
            self._update_inventories_func = _real_update_inventories_func
        else:
            self._format_inventory_func = _dummy_format_inventory
            self._update_inventories_func = _dummy_update_inventories

        self.logger.info("SessionPipeOrchestrator initialized (Non-Streaming).")

    # --- Internal Helper: Status Emitter ---
    async def _emit_status(
        self,
        event_emitter: Optional[Callable],
        session_id: str,
        description: str,
        done: bool = False
    ):
        """Emits status updates via the provided callable if configured."""
        if event_emitter and callable(event_emitter) and getattr(self.config, 'emit_status_updates', True):
            try:
                status_data = { "type": "status", "data": {"description": str(description), "done": bool(done)} }
                # Check if emitter itself is async (Open WebUI's usually is)
                if asyncio.iscoroutinefunction(event_emitter):
                    await event_emitter(status_data)
                else:
                    # If somehow it's sync, call it directly (less likely for OWI)
                    event_emitter(status_data)
            except Exception as e_emit:
                self.logger.warning(f"[{session_id}] Orchestrator failed to emit status '{description}': {e_emit}")
        else:
             self.logger.debug(f"[{session_id}] Orchestrator status update (not emitted): '{description}' (Done: {done})")


    # --- MODIFIED Internal Helper: Async LLM Call Wrapper (Simplified) ---
    async def _async_llm_call_wrapper(
        self,
        api_url: str,
        api_key: str,
        payload: Dict[str, Any], # Expects Google 'contents' format
        temperature: float,
        timeout: int = 90,
        caller_info: str = "Orchestrator_LLM",
    ) -> Tuple[bool, Union[str, Dict]]:
        """
        Wraps the library's LLM call function (now the litellm adapter) for error handling.
        Directly awaits the underlying async function.
        """
        if not self._llm_call_func:
            self.logger.error(f"[{caller_info}] LLM func alias unavailable in orchestrator.")
            return False, {"error_type": "SetupError", "message": "LLM func alias unavailable"}

        if not asyncio.iscoroutinefunction(self._llm_call_func):
             self.logger.critical(f"[{caller_info}] LLM func alias is NOT async! Expected async adapter. Cannot proceed.")
             return False, {"error_type": "SetupError", "message": "LLM func alias is not async"}

        try:
            # Directly await the function (which is now the async litellm adapter)
            self.logger.debug(f"[{caller_info}] Awaiting result from LLM adapter function.")
            success, result_or_error = await self._llm_call_func(
                api_url=api_url, api_key=api_key, payload=payload,
                temperature=temperature, timeout=timeout, caller_info=caller_info
            )
            self.logger.debug(f"[{caller_info}] LLM adapter returned (Success: {success}).")
            return success, result_or_error

        except asyncio.CancelledError:
             self.logger.info(f"[{caller_info}] LLM call cancelled.")
             raise # Re-raise cancellation
        except Exception as e:
            self.logger.error(f"Orchestrator LLM Wrapper Error [{caller_info}]: Uncaught exception during await: {e}", exc_info=True)
            return False, {"error_type": "AsyncWrapperError", "message": f"{type(e).__name__}: {str(e)}"}


    # --- REMOVED Internal Helper: Final LLM Stream Call Wrapper ---
    # Streaming is removed


    # --- START DEBUG LOGGING HELPERS (adapted from Pipe) ---
    def _orchestrator_get_debug_log_path(self, suffix: str) -> Optional[str]:
        """Gets the debug log path using config and logger passed from Pipe."""
        func_logger = getattr(self, 'pipe_logger', self.logger) # Use pipe logger if available
        func_logger.debug(f"_orchestrator_get_debug_log_path called with suffix: '{suffix}'")
        try:
            base_log_path = getattr(self.config, "log_file_path", None)
            if not base_log_path:
                func_logger.error("Orch Debug Path: Main log_file_path config is empty.")
                return None
            log_dir = os.path.dirname(base_log_path)
            func_logger.debug(f"Orch Debug Path: Target log directory: '{log_dir}'")
            try:
                os.makedirs(log_dir, exist_ok=True)
            except PermissionError as pe:
                func_logger.error(f"Orch Debug Path: PERMISSION ERROR creating log directory '{log_dir}': {pe}")
                return None
            except Exception as e_mkdir:
                func_logger.error(f"Orch Debug Path: Error creating log directory '{log_dir}': {e_mkdir}", exc_info=True)
                return None

            base_name, _ = os.path.splitext(os.path.basename(base_log_path))
            debug_filename = f"{base_name}{suffix}.log"
            final_path = os.path.join(log_dir, debug_filename)
            func_logger.info(f"Orch Debug Path: Constructed debug log path: '{final_path}'")
            return final_path
        except AttributeError as ae:
            func_logger.error(f"Orch Debug Path: Config object missing attribute ('log_file_path'?): {ae}")
            return None
        except Exception as e:
            func_logger.error(f"Orch Debug Path: Failed get debug path '{suffix}': {e}", exc_info=True)
            return None

    def _orchestrator_log_debug_payload(self, session_id: str, payload_body: Dict):
        """Logs the final payload dictionary to the designated debug file."""
        debug_log_path = self._orchestrator_get_debug_log_path(".DEBUG_PAYLOAD")
        if not debug_log_path:
            self.logger.error(f"[{session_id}] Orch: Cannot log final payload: No path determined.")
            return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_entry = {
                "ts": ts,
                "pipe_version": getattr(self.config, "version", "unknown"),
                "sid": session_id,
                "payload": payload_body,
            }
            self.logger.info(f"[{session_id}] Orch: Attempting write FINAL PAYLOAD debug log to: {debug_log_path}")
            with open(debug_log_path, "a", encoding="utf-8") as f:
                 f.write(f"--- [{ts}] SESSION: {session_id} - FINAL ORCHESTRATOR PAYLOAD --- START ---\n")
                 if 'contents' in payload_body:
                     log_entry_payload = payload_body.copy()
                     log_entry['payload']['contents'] = log_entry_payload.pop('contents', None)
                     log_entry['payload_other_keys'] = log_entry_payload
                 json.dump(log_entry, f, indent=2)
                 f.write(f"\n--- [{ts}] SESSION: {session_id} - FINAL ORCHESTRATOR PAYLOAD --- END ---\n\n")
            self.logger.info(f"[{session_id}] Orch: Successfully wrote FINAL PAYLOAD debug log.")
        except Exception as e:
            self.logger.error(f"[{session_id}] Orch: Failed write debug final payload log: {e}", exc_info=True)

    # --- NEW: Debug Logger for Inventory LLM ---
    def _orchestrator_log_debug_inventory_llm(self, session_id: str, text: str, is_prompt: bool):
        """Logs the Inventory LLM prompt or response text to a debug file."""
        # Log to the main payload file if enabled
        debug_log_path = self._orchestrator_get_debug_log_path(".DEBUG_PAYLOAD")
        if not debug_log_path:
            self.logger.error(f"[{session_id}] Orch: Cannot log inventory LLM text: No path determined.")
            return
        if not getattr(self.config, 'debug_log_final_payload', False):
            return # Only log if main debug payload is enabled

        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_type = "PROMPT" if is_prompt else "RESPONSE"
            self.logger.info(f"[{session_id}] Orch: Attempting write INVENTORY LLM {log_type} debug log to: {debug_log_path}")
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- [{ts}] SESSION: {session_id} - INVENTORY LLM {log_type} --- START ---\n")
                f.write(str(text)) # Ensure text is string
                f.write(f"\n--- [{ts}] SESSION: {session_id} - INVENTORY LLM {log_type} --- END ---\n\n")
            self.logger.info(f"[{session_id}] Orch: Successfully wrote INVENTORY LLM {log_type} debug log.")
        except Exception as e:
            self.logger.error(f"[{session_id}] Orch: Failed write debug inventory LLM {log_type} log: {e}", exc_info=True)
    # --- END DEBUG LOGGING HELPERS ---

    # --- Helper Methods for process_turn ---

    async def _determine_effective_query(
        self, session_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool
    ) -> Tuple[str, List[Dict]]:
        """ Determines the effective user query and the history slice preceding it. """
        effective_user_message_index = -1
        user_message_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") == "user"]
        if not user_message_indices: self.logger.error(f"[{session_id}] No user messages found."); return "", []
        if is_regeneration_heuristic:
            effective_user_message_index = user_message_indices[-2] if len(user_message_indices) >= 2 else user_message_indices[-1]
            log_level = self.logger.info if len(user_message_indices) >= 2 else self.logger.warning
            log_level(f"[{session_id}] Regen: Using user message at index {effective_user_message_index} as query base.")
        else:
            effective_user_message_index = user_message_indices[-1]
            self.logger.debug(f"[{session_id}] Normal: Using user message at index {effective_user_message_index} as query base.")
        if effective_user_message_index < 0 or effective_user_message_index >= len(current_active_history): self.logger.error(f"[{session_id}] Effective user index {effective_user_message_index} out of bounds."); return "", []
        effective_user_message = current_active_history[effective_user_message_index]
        history_for_processing = current_active_history[:effective_user_message_index]
        latest_user_query_str = effective_user_message.get("content", "") if isinstance(effective_user_message, dict) else ""
        self.logger.debug(f"[{session_id}] Effective query set (len: {len(latest_user_query_str)}). History slice for processing len: {len(history_for_processing)}.")
        return latest_user_query_str, history_for_processing


    async def _handle_tier1_summarization(
        self, session_id: str, user_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool, event_emitter: Optional[Callable]
    ) -> Tuple[bool, Optional[str], int, int]:
        """ Checks and performs T1 summarization. """
        await self._emit_status(event_emitter, session_id, "Status: Checking summarization...")
        summarization_performed_successfully = False; generated_summary = None; summarization_prompt_tokens = -1; summarization_output_tokens = -1
        can_summarize = all([ self._manage_memory_func, self._tokenizer, self._count_tokens_func, self.sqlite_cursor, self._async_llm_call_wrapper, hasattr(self.config, 'summarizer_api_url') and self.config.summarizer_api_url, hasattr(self.config, 'summarizer_api_key') and self.config.summarizer_api_key, current_active_history,])
        if not can_summarize:
             missing_prereqs = [p for p, v in {"manage_func": self._manage_memory_func, "tokenizer": self._tokenizer, "count_func": self._count_tokens_func, "db_cursor": self.sqlite_cursor, "llm_wrapper": self._async_llm_call_wrapper, "summ_url": getattr(self.config, 'summarizer_api_url', None), "summ_key": getattr(self.config, 'summarizer_api_key', None), "history": bool(current_active_history)}.items() if not v]
             self.logger.warning(f"[{session_id}] Skipping T1 check: Missing prerequisites: {', '.join(missing_prereqs)}."); return False, None, -1, -1

        summarizer_llm_config = { "url": self.config.summarizer_api_url, "key": self.config.summarizer_api_key, "temp": getattr(self.config, 'summarizer_temperature', 0.5), "sys_prompt": getattr(self.config, 'summarizer_system_prompt', "Summarize this dialogue."), }
        new_last_summary_idx = -1; prompt_tokens = -1; t0_end_idx = -1; db_max_index = None; current_last_summary_index_for_memory = -1
        try:
            db_max_index = await get_max_t1_end_index(self.sqlite_cursor, session_id)
            if isinstance(db_max_index, int) and db_max_index >= 0: current_last_summary_index_for_memory = db_max_index; self.logger.info(f"[{session_id}] T1: Start Index from DB: {current_last_summary_index_for_memory}")
            else: self.logger.info(f"[{session_id}] T1: No valid start index in DB. Starting from -1.")
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
                self.logger.info(f"[{session_id}] T1 summary generated/saved. New Idx: {new_last_summary_idx}, PromptTok: {summarization_prompt_tokens}, OutTok: {summarization_output_tokens}.")
                await self._emit_status(event_emitter, session_id, "Status: Summary generated.", done=False)
            else: self.logger.debug(f"[{session_id}] T1 summarization skipped or criteria not met (Returned Index: {new_last_summary_idx}).")
        except TypeError as e_type: self.logger.error(f"[{session_id}] Orchestrator TYPE ERROR calling T1 manage func: {e_type}. Signature mismatch?", exc_info=True)
        except Exception as e_manage: self.logger.error(f"[{session_id}] Orchestrator EXCEPTION during T1 manage call: {e_manage}", exc_info=True)
        return summarization_performed_successfully, generated_summary, summarization_prompt_tokens, summarization_output_tokens


    async def _handle_tier2_transition(
        self, session_id: str, t1_success: bool, chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> None:
        """ Handles the transition of the oldest T1 summary to T2 if needed. """
        await self._emit_status(event_emitter, session_id, "Status: Checking long-term memory capacity...")
        tier2_collection = None
        max_t1_blocks = getattr(self.config, 'max_stored_summary_blocks', 0)
        can_transition = all([ t1_success, self.chroma_client is not None, chroma_embed_wrapper is not None, self.sqlite_cursor is not None, max_t1_blocks > 0])
        if not can_transition: self.logger.debug(f"[{session_id}] Skipping T1->T2 transition check: {'(T1 did not run)' if not t1_success else '(Prerequisites not met)'}."); return

        try:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_'); safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]; tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
            tier2_collection = await get_or_create_chroma_collection(self.chroma_client, tier2_collection_name, chroma_embed_wrapper)
            if not tier2_collection: self.logger.error(f"[{session_id}] Failed to get/create T2 collection '{tier2_collection_name}'. Skipping transition."); return
        except Exception as e_get_coll: self.logger.error(f"[{session_id}] Error getting T2 collection: {e_get_coll}. Skipping transition.", exc_info=True); return

        try:
            current_tier1_count = await get_tier1_summary_count(self.sqlite_cursor, session_id)
            if current_tier1_count == -1: self.logger.error(f"[{session_id}] Failed get T1 count. Skipping T1->T2 check.")
            elif current_tier1_count > max_t1_blocks:
                self.logger.info(f"[{session_id}] T1 limit ({max_t1_blocks}) exceeded ({current_tier1_count}). Transitioning oldest...")
                await self._emit_status(event_emitter, session_id, "Status: Archiving oldest summary...")
                oldest_summary_data = await get_oldest_tier1_summary(self.sqlite_cursor, session_id)
                if not oldest_summary_data: self.logger.warning(f"[{session_id}] T1 count exceeded limit, but couldn't retrieve oldest summary."); return
                oldest_id, oldest_text, oldest_metadata = oldest_summary_data
                embedding_vector = None; embedding_successful = False
                try:
                    embedding_list = await asyncio.to_thread(chroma_embed_wrapper, [oldest_text])
                    if isinstance(embedding_list, list) and len(embedding_list) == 1 and isinstance(embedding_list[0], list) and len(embedding_list[0]) > 0: embedding_vector = embedding_list[0]; embedding_successful = True
                    else: self.logger.error(f"[{session_id}] T1->T2 Embed: Invalid structure returned: {type(embedding_list)}")
                except Exception as embed_e: self.logger.error(f"[{session_id}] EXCEPTION embedding T1->T2 summary {oldest_id}: {embed_e}", exc_info=True)
                if embedding_successful and embedding_vector:
                    added_to_t2 = False; deleted_from_t1 = False; chroma_metadata = oldest_metadata.copy(); chroma_metadata["transitioned_from_t1"] = True; chroma_metadata["original_t1_id"] = oldest_id
                    sanitized_chroma_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in chroma_metadata.items() if v is not None}
                    tier2_id = f"t2_{oldest_id}"
                    self.logger.info(f"[{session_id}] Adding summary {tier2_id} to T2 collection '{tier2_collection.name}'...")
                    added_to_t2 = await add_to_chroma_collection(tier2_collection, ids=[tier2_id], embeddings=[embedding_vector], metadatas=[sanitized_chroma_metadata], documents=[oldest_text])
                    if added_to_t2:
                         self.logger.info(f"[{session_id}] Added {tier2_id} to T2. Deleting T1 summary {oldest_id}...")
                         deleted_from_t1 = await delete_tier1_summary(self.sqlite_cursor, oldest_id)
                         if deleted_from_t1: self.logger.info(f"[{session_id}] Successfully deleted T1 summary {oldest_id}."); await self._emit_status(event_emitter, session_id, "Status: Summary archive complete.", done=False)
                         else: self.logger.critical(f"[{session_id}] Added {tier2_id} to T2, but FAILED TO DELETE T1 {oldest_id}!")
                    else: self.logger.error(f"[{session_id}] Failed to add summary {tier2_id} to T2 collection.")
                else: self.logger.error(f"[{session_id}] Skipping T2 addition for T1 summary {oldest_id}: Embedding failed.")
            else: self.logger.debug(f"[{session_id}] T1 count ({current_tier1_count}) within limit ({max_t1_blocks}). No transition needed.")
        except Exception as e_t2_trans: self.logger.error(f"[{session_id}] Unexpected error during T1->T2 transition: {e_t2_trans}", exc_info=True)


    async def _get_t1_summaries(self, session_id: str) -> Tuple[List[str], int]:
        """ Fetches recent T1 summaries from the database. """
        recent_t1_summaries = []; t1_retrieved_count = 0; max_blocks_t1 = getattr(self.config, 'max_stored_summary_blocks', 0)
        if self.sqlite_cursor and max_blocks_t1 > 0:
             try: recent_t1_summaries = await get_recent_tier1_summaries(self.sqlite_cursor, session_id, max_blocks_t1); t1_retrieved_count = len(recent_t1_summaries)
             except Exception as e_get_t1: self.logger.error(f"[{session_id}] Error retrieving T1 summaries: {e_get_t1}", exc_info=True); recent_t1_summaries = []; t1_retrieved_count = 0
        elif not self.sqlite_cursor: self.logger.warning(f"[{session_id}] Cannot get T1 summaries: SQLite cursor unavailable.")
        elif max_blocks_t1 <= 0: self.logger.debug(f"[{session_id}] Skipping T1 summary retrieval: max_stored_summary_blocks is {max_blocks_t1}.")
        if t1_retrieved_count > 0: self.logger.info(f"[{session_id}] Retrieved {t1_retrieved_count} T1 summaries for context.")
        return recent_t1_summaries, t1_retrieved_count


    async def _get_t2_rag_results(
        self, session_id: str, history_for_processing: List[Dict], latest_user_query_str: str,
        embedding_func: Optional[Callable], chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> Tuple[List[str], int]:
        """ Performs T2 RAG lookup based on a generated query. """
        await self._emit_status(event_emitter, session_id, "Status: Searching long-term memory...")
        retrieved_rag_summaries = []; t2_retrieved_count = 0; tier2_collection = None; n_results_t2 = getattr(self.config, 'rag_summary_results_count', 0)
        can_rag = all([ self.chroma_client is not None, chroma_embed_wrapper is not None, latest_user_query_str, embedding_func is not None, self._generate_rag_query_func is not None, self._async_llm_call_wrapper is not None, getattr(self.config, 'ragq_llm_api_url', None), getattr(self.config, 'ragq_llm_api_key', None), getattr(self.config, 'ragq_llm_prompt', None), n_results_t2 > 0 ])
        if not can_rag: self.logger.info(f"[{session_id}] Skipping T2 RAG check: Prerequisites not met (RAG Results Count: {n_results_t2})."); return [], 0

        try: base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_'); safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]; tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
        except Exception as e_name: self.logger.error(f"[{session_id}] Error creating T2 collection name: {e_name}"); return [], 0
        try: tier2_collection = await get_or_create_chroma_collection(self.chroma_client, tier2_collection_name, chroma_embed_wrapper)
        except Exception as e_get_coll_rag: self.logger.error(f"[{session_id}] Error getting T2 collection for RAG: {e_get_coll_rag}. Skipping.", exc_info=True); return [], 0
        if not tier2_collection: self.logger.error(f"[{session_id}] Failed get/create T2 collection '{tier2_collection_name}'. Skipping RAG."); return [], 0
        try: t2_doc_count = await get_chroma_collection_count(tier2_collection)
        except Exception as e_count: self.logger.error(f"[{session_id}] Error checking T2 collection count: {e_count}. Skipping RAG.", exc_info=True); return [], 0
        if t2_doc_count <= 0: self.logger.info(f"[{session_id}] Skipping T2 RAG: Collection '{tier2_collection.name}' is empty ({t2_doc_count})."); return [], 0

        try:
            await self._emit_status(event_emitter, session_id, "Status: Generating search query...")
            context_messages_for_ragq = self._get_recent_turns_func( history_for_processing, count=getattr(self.config, 'refiner_history_count', 6), exclude_last=False, roles=self._dialogue_roles)
            dialogue_context_str = self._format_history_func(context_messages_for_ragq) if context_messages_for_ragq else "[No recent history]"
            ragq_llm_config = { "url": self.config.ragq_llm_api_url, "key": self.config.ragq_llm_api_key, "temp": getattr(self.config, 'ragq_llm_temperature', 0.3), "prompt": self.config.ragq_llm_prompt, }
            rag_query = await self._generate_rag_query_func( latest_message_str=latest_user_query_str, dialogue_context_str=dialogue_context_str, llm_call_func=self._async_llm_call_wrapper, llm_config=ragq_llm_config, caller_info=f"Orch_RAGQ_{session_id}",)
            if not (rag_query and isinstance(rag_query, str) and not rag_query.startswith("[Error:") and rag_query.strip()): self.logger.error(f"[{session_id}] RAG Query Generation failed: '{rag_query}'. Skipping RAG."); return [], 0
            self.logger.info(f"[{session_id}] Generated RAG Query: '{rag_query[:100]}...'")

            await self._emit_status(event_emitter, session_id, "Status: Embedding search query...")
            query_embedding = None; query_embedding_successful = False
            try:
                if not callable(embedding_func): self.logger.error(f"[{session_id}] Cannot embed RAG query: OWI Embedding function invalid."); return [], 0
                from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX
                query_embedding_list = await asyncio.to_thread(embedding_func, [rag_query], prefix=RAG_EMBEDDING_QUERY_PREFIX)
                if isinstance(query_embedding_list, list) and len(query_embedding_list) == 1 and isinstance(query_embedding_list[0], list) and len(query_embedding_list[0]) > 0: query_embedding = query_embedding_list[0]; query_embedding_successful = True; self.logger.debug(f"[{session_id}] RAG query embedding successful (dim: {len(query_embedding)}).")
                else: self.logger.error(f"[{session_id}] RAG query embed invalid structure: {type(query_embedding_list)}.")
            except Exception as embed_e: self.logger.error(f"[{session_id}] EXCEPTION during RAG query embedding: {embed_e}", exc_info=True)
            if not (query_embedding_successful and query_embedding): self.logger.error(f"[{session_id}] Skipping T2 ChromaDB query: RAG query embedding failed."); return [], 0

            await self._emit_status(event_emitter, session_id, f"Status: Searching vector store (top {n_results_t2})...")
            rag_results_dict = await query_chroma_collection( tier2_collection, query_embeddings=[query_embedding], n_results=n_results_t2, include=["documents", "distances", "metadatas"])
            if rag_results_dict and isinstance(rag_results_dict.get("documents"), list) and rag_results_dict["documents"] and isinstance(rag_results_dict["documents"][0], list):
                  retrieved_docs = rag_results_dict["documents"][0]
                  if retrieved_docs:
                       retrieved_rag_summaries = retrieved_docs; t2_retrieved_count = len(retrieved_docs)
                       distances = rag_results_dict.get("distances", [[None]])[0]; ids = rag_results_dict.get("ids", [["N/A"]])[0]; dist_str = [f"{d:.4f}" for d in distances if d is not None]
                       self.logger.info(f"[{session_id}] Retrieved {t2_retrieved_count} docs from T2 RAG. IDs: {ids}, Dist: {dist_str}")
                  else: self.logger.info(f"[{session_id}] T2 RAG query executed but returned no documents.")
            else: self.logger.info(f"[{session_id}] T2 RAG query returned no matches or unexpected structure: {type(rag_results_dict)}")
        except Exception as e_rag_outer: self.logger.error(f"[{session_id}] Unexpected error during outer T2 RAG processing: {e_rag_outer}", exc_info=True); retrieved_rag_summaries = []; t2_retrieved_count = 0
        return retrieved_rag_summaries, t2_retrieved_count


    async def _prepare_and_refine_background(
        self, session_id: str, body: Dict, user_valves: Any,
        retrieved_t1_summaries: List[str], retrieved_rag_summaries: List[str],
        current_active_history: List[Dict], latest_user_query_str: str,
        event_emitter: Optional[Callable]
    ) -> Tuple[str, str, int, int, bool, bool, bool, bool, str]: # Return includes formatted inventory string (for status)
        """
        Processes system prompt, manages context refinement (RAG Cache or Stateless),
        fetches and formats inventory, and combines background information.
        MODIFIED: Always passes valid formatted inventory to the context combiner,
                  bypassing the Step 2 LLM for inventory inclusion guarantee.
                  Logs inventory tracing steps to a dedicated file.
        """
        func_logger = self.logger
        inventory_trace_log_path = None
        debug_inventory_trace_enabled = getattr(self.config, 'debug_log_final_payload', False)

        def _log_inventory_trace(message: str):
            nonlocal inventory_trace_log_path
            if not debug_inventory_trace_enabled: return
            try:
                if inventory_trace_log_path is None:
                    # Log inventory trace to the payload file for consolidation
                    inventory_trace_log_path = self._orchestrator_get_debug_log_path(".DEBUG_PAYLOAD")
                if inventory_trace_log_path:
                    ts = datetime.now(timezone.utc).isoformat()
                    log_line = f"\n--- [{ts}] SESSION: {session_id} - INV_TRACE --- {message} ---\n"
                    with open(inventory_trace_log_path, "a", encoding="utf-8") as f: f.write(log_line)
                else:
                    if inventory_trace_log_path is not False: # Only log error once
                        func_logger.error(f"[{session_id}] Inventory Trace: Cannot log, failed to determine debug path.")
                        inventory_trace_log_path = False # Prevent further attempts
            except Exception as e_log:
                 if inventory_trace_log_path is not False:
                    func_logger.error(f"[{session_id}] Inventory Trace: Error writing to log file '{inventory_trace_log_path}': {e_log}", exc_info=False)
                    inventory_trace_log_path = False

        _log_inventory_trace("--- START _prepare_and_refine_background (Bypass Logic Active) ---")

        await self._emit_status(event_emitter, session_id, "Status: Preparing context...")
        base_system_prompt_text = "You are helpful."; extracted_owi_context = None; initial_owi_context_tokens = -1; current_output_messages = body.get("messages", [])

        _log_inventory_trace("Step 1: Processing system prompt...")
        if self._process_system_prompt_func:
             try: base_system_prompt_text, extracted_owi_context = self._process_system_prompt_func(current_output_messages)
             except Exception as e_proc_sys: func_logger.error(f"[{session_id}] Error process_system_prompt: {e_proc_sys}.", exc_info=True); extracted_owi_context = None
        else: func_logger.error(f"[{session_id}] process_system_prompt unavailable."); base_system_prompt_text = "You are helpful."

        if extracted_owi_context and self._count_tokens_func and self._tokenizer:
             try: initial_owi_context_tokens = self._count_tokens_func(extracted_owi_context, self._tokenizer)
             except Exception: initial_owi_context_tokens = -1
        elif not extracted_owi_context: func_logger.debug(f"[{session_id}] No OWI <context> tag found.")
        _log_inventory_trace(f"Extracted OWI Context Length: {len(extracted_owi_context) if extracted_owi_context else 0}, Tokens: {initial_owi_context_tokens}")

        if not base_system_prompt_text: base_system_prompt_text = "You are helpful."; func_logger.warning(f"[{session_id}] System prompt empty after clean. Using default.")

        session_text_block_to_remove = getattr(user_valves, 'text_block_to_remove', '') if user_valves else ''
        if session_text_block_to_remove:
            original_len = len(base_system_prompt_text); temp_prompt = base_system_prompt_text.replace(session_text_block_to_remove, "")
            if len(temp_prompt) < original_len: base_system_prompt_text = temp_prompt; _log_inventory_trace(f"Removed text block from system prompt ({original_len - len(temp_prompt)} chars).")
            else: _log_inventory_trace(f"Text block for removal NOT FOUND: '{session_text_block_to_remove[:50]}...'")

        session_process_owi_rag = bool(getattr(user_valves, 'process_owi_rag', True))
        if not session_process_owi_rag:
            _log_inventory_trace("Session valve 'process_owi_rag=False'. Discarding OWI context.")
            extracted_owi_context = None; initial_owi_context_tokens = 0

        _log_inventory_trace("Step 2: Fetching and formatting inventory...")
        formatted_inventory_string = "[Inventory Management Disabled]"; raw_session_inventories = {}; inventory_enabled = getattr(self.config, 'enable_inventory_management', False)
        if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and self._get_all_inventories_db_func and self._format_inventory_func and self.sqlite_cursor:
            _log_inventory_trace("Inventory enabled, fetching data...")
            try:
                raw_session_inventories = await self._get_all_inventories_db_func(self.sqlite_cursor, session_id)
                if raw_session_inventories:
                    _log_inventory_trace(f"Retrieved inventory data for {len(raw_session_inventories)} characters.")
                    try:
                        formatted_inventory_string = self._format_inventory_func(raw_session_inventories);
                        _log_inventory_trace(f"Formatted inventory string generated (len: {len(formatted_inventory_string)}). Content:\n------\n{formatted_inventory_string}\n------")
                    except Exception as e_fmt_inv:
                        func_logger.error(f"[{session_id}] Failed to format inventory string: {e_fmt_inv}", exc_info=True);
                        formatted_inventory_string = "[Error Formatting Inventory]"
                        _log_inventory_trace(f"ERROR formatting inventory string: {e_fmt_inv}")
                else:
                    _log_inventory_trace("No inventory data found in DB for this session.")
                    formatted_inventory_string = "[No Inventory Data Available]"
            except Exception as e_get_inv:
                func_logger.error(f"[{session_id}] Error retrieving inventory data from DB: {e_get_inv}", exc_info=True);
                formatted_inventory_string = "[Error Retrieving Inventory]"
                _log_inventory_trace(f"ERROR retrieving inventory data from DB: {e_get_inv}")
        elif not inventory_enabled: _log_inventory_trace("Skipping inventory fetch: Feature disabled by global valve.")
        elif inventory_enabled and not _ORCH_INVENTORY_MODULE_AVAILABLE: _log_inventory_trace("Skipping inventory fetch: Module unavailable (Import failed).")
        else:
             missing_inv_funcs = [f for f, fn in {"db_get": self._get_all_inventories_db_func, "formatter": self._format_inventory_func, "cursor": self.sqlite_cursor}.items() if not fn];
             _log_inventory_trace(f"Skipping inventory fetch: Missing prerequisites: {missing_inv_funcs}")
             formatted_inventory_string = "[Inventory Init/Config Error]"

        _log_inventory_trace("Step 3: Context Refinement Logic...")
        context_for_prompt = extracted_owi_context; refined_context_tokens = -1; cache_update_performed = False; cache_update_skipped = False; final_context_selection_performed = False; stateless_refinement_performed = False; updated_cache_text_intermediate = "[Cache not initialized or updated]"
        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False); enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False)
        _log_inventory_trace(f"RAG Cache Enabled: {enable_rag_cache_global}, Stateless Refinement Enabled: {enable_stateless_refin_global}")

        if enable_rag_cache_global and self._cache_update_func and self._cache_select_func and self._get_rag_cache_db_func and self.sqlite_cursor:
            _log_inventory_trace("RAG Cache Path Selected.")
            run_step1 = False; previous_cache_text = "";
            try: cache_result = await self._get_rag_cache_db_func(self.sqlite_cursor, session_id); previous_cache_text = cache_result if cache_result is not None else ""
            except Exception as e_get_cache: func_logger.error(f"[{session_id}] Error retrieving previous cache: {e_get_cache}", exc_info=True)
            _log_inventory_trace(f"Previous cache length: {len(previous_cache_text)}")

            if not session_process_owi_rag:
                _log_inventory_trace("Skipping RAG Cache Step 1 (session valve 'process_owi_rag=False').")
                cache_update_skipped = True; run_step1 = False; updated_cache_text_intermediate = previous_cache_text
            else:
                 skip_len = False; skip_sim = False; owi_content_for_check = extracted_owi_context or ""; len_thresh = getattr(self.config, 'CACHE_UPDATE_SKIP_OWI_THRESHOLD', 50)
                 if len(owi_content_for_check.strip()) < len_thresh: skip_len = True; _log_inventory_trace(f"Cache S1 Skip: OWI len ({len(owi_content_for_check.strip())}) < {len_thresh}.")
                 elif self._calculate_similarity_func and previous_cache_text:
                      sim_thresh = getattr(self.config, 'CACHE_UPDATE_SIMILARITY_THRESHOLD', 0.9)
                      try: sim_score = self._calculate_similarity_func(owi_content_for_check, previous_cache_text)
                      except Exception as e_sim: func_logger.error(f"[{session_id}] Error calculating similarity: {e_sim}")
                      else:
                          if sim_score > sim_thresh: skip_sim = True; _log_inventory_trace(f"Cache S1 Skip: Sim ({sim_score:.2f}) > {sim_thresh:.2f}.")
                 cache_update_skipped = skip_len or skip_sim; run_step1 = not cache_update_skipped
                 if cache_update_skipped:
                     await self._emit_status(event_emitter, session_id, "Status: Skipping cache update (redundant OWI).");
                     updated_cache_text_intermediate = previous_cache_text; _log_inventory_trace("Cache Step 1 SKIPPED.")

            cache_update_llm_config = { "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None), "temp": getattr(self.config, 'refiner_llm_temperature', 0.3), "prompt_template": getattr(self.config, 'cache_update_prompt_template', DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT),}
            final_select_llm_config = { "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None), "temp": getattr(self.config, 'refiner_llm_temperature', 0.3), "prompt_template": getattr(self.config, 'final_context_selection_prompt_template', DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT),}
            configs_ok_step1 = all([cache_update_llm_config["url"], cache_update_llm_config["key"], cache_update_llm_config["prompt_template"]])
            configs_ok_step2 = all([final_select_llm_config["url"], final_select_llm_config["key"], final_select_llm_config["prompt_template"]])

            if not (configs_ok_step1 and configs_ok_step2):
                 _log_inventory_trace("ERROR: RAG Cache Refiner config incomplete. Cannot proceed.")
                 await self._emit_status(event_emitter, session_id, "ERROR: RAG Cache Refiner config incomplete.", done=False);
                 updated_cache_text_intermediate = previous_cache_text; run_step1 = False
            else:
                 if run_step1:
                      await self._emit_status(event_emitter, session_id, "Status: Updating background cache...");
                      _log_inventory_trace("Executing RAG Cache Step 1 (Update)...")
                      try:
                          updated_cache_text_intermediate = await self._cache_update_func( session_id=session_id, current_owi_context=extracted_owi_context, history_messages=current_active_history, latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper, sqlite_cursor=self.sqlite_cursor, cache_update_llm_config=cache_update_llm_config, history_count=getattr(self.config, 'refiner_history_count', 6), dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_CacheUpdate_{session_id}",)
                          cache_update_performed = True;
                          _log_inventory_trace(f"RAG Cache Step 1 (Update) completed. Updated cache length: {len(updated_cache_text_intermediate)}")
                      except Exception as e_cache_update:
                           func_logger.error(f"[{session_id}] EXCEPTION during RAG Cache Step 1 (Update): {e_cache_update}", exc_info=True);
                           updated_cache_text_intermediate = previous_cache_text
                           _log_inventory_trace(f"EXCEPTION during RAG Cache Step 1: {e_cache_update}")

                 await self._emit_status(event_emitter, session_id, "Status: Selecting relevant context...")
                 base_owi_context_for_selection = extracted_owi_context or ""
                 _log_inventory_trace("Executing RAG Cache Step 2 (Select) with OWI only (Inventory bypassed)...")
                 _log_inventory_trace(f"Step 2 Input Cache Length: {len(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else '')}")
                 _log_inventory_trace(f"Step 2 Input OWI Length: {len(base_owi_context_for_selection)}")

                 final_selected_context = await self._cache_select_func(
                     updated_cache_text=(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else ""),
                     current_owi_context=base_owi_context_for_selection,
                     history_messages=current_active_history,
                     latest_user_query=latest_user_query_str,
                     llm_call_func=self._async_llm_call_wrapper,
                     context_selection_llm_config=final_select_llm_config,
                     history_count=getattr(self.config, 'refiner_history_count', 6),
                     dialogue_only_roles=self._dialogue_roles,
                     caller_info=f"Orch_CtxSelect_{session_id}",
                 )
                 final_context_selection_performed = True
                 context_for_prompt = final_selected_context
                 log_step1_status = "Performed" if cache_update_performed else ("Skipped" if cache_update_skipped else "Not Run")
                 _log_inventory_trace(f"RAG Cache Step 2 complete. Selected context length: {len(context_for_prompt)}. Step 1: {log_step1_status}\n------\n{context_for_prompt}\n------")
                 await self._emit_status(event_emitter, session_id, "Status: Context selection complete.", done=False)

        elif enable_stateless_refin_global and self._stateless_refine_func:
            _log_inventory_trace("Stateless Refinement Path Selected.")
            await self._emit_status(event_emitter, session_id, "Status: Refining OWI context (stateless)...")
            if not extracted_owi_context: _log_inventory_trace("Skipping stateless refinement: No OWI context.")
            elif not latest_user_query_str: _log_inventory_trace("Skipping stateless refinement: Query empty.")
            else:
                 stateless_refiner_config = { "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None), "temp": getattr(self.config, 'refiner_llm_temperature', 0.3), "prompt_template": getattr(self.config, 'stateless_refiner_prompt_template', None),}
                 if not stateless_refiner_config["url"] or not stateless_refiner_config["key"]:
                     _log_inventory_trace("Skipping stateless refinement: Refiner URL/Key missing.")
                     await self._emit_status(event_emitter, session_id, "ERROR: Stateless Refiner config incomplete.", done=False)
                 else:
                      try:
                          refined_stateless_context = await self._stateless_refine_func( external_context=extracted_owi_context, history_messages=current_active_history, latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper, refiner_llm_config=stateless_refiner_config, skip_threshold=getattr(self.config, 'stateless_refiner_skip_threshold', 500), history_count=getattr(self.config, 'refiner_history_count', 6), dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_StatelessRef_{session_id}",)
                          if refined_stateless_context != extracted_owi_context:
                              context_for_prompt = refined_stateless_context; stateless_refinement_performed = True;
                              _log_inventory_trace(f"Stateless refinement successful. Refined length: {len(context_for_prompt)}.")
                              await self._emit_status(event_emitter, session_id, "Status: OWI context refined (stateless).", done=False)
                          else: _log_inventory_trace("Stateless refinement resulted in no change or was skipped by length.")
                      except Exception as e_refine_stateless:
                           func_logger.error(f"[{session_id}] EXCEPTION during stateless refinement: {e_refine_stateless}", exc_info=True)
                           _log_inventory_trace(f"EXCEPTION during stateless refinement: {e_refine_stateless}")
        else:
            _log_inventory_trace("No context refinement feature (RAG Cache or Stateless) is enabled.")

        _log_inventory_trace("Step 4: Calculating refined context tokens...")
        if self._count_tokens_func and self._tokenizer:
            try: token_source = context_for_prompt; refined_context_tokens = self._count_tokens_func(token_source, self._tokenizer) if token_source and isinstance(token_source, str) else 0
            except Exception as e_tok_ref: refined_context_tokens = -1; func_logger.error(f"[{session_id}] Error calculating refined tokens: {e_tok_ref}")
        else: refined_context_tokens = -1
        _log_inventory_trace(f"Final context_for_prompt tokens: {refined_context_tokens}")

        _log_inventory_trace("Step 5: Combining final background context...")
        combined_context_string = "[No background context generated]"
        if self._combine_context_func:
            try:
                inventory_to_pass_to_combiner = None
                is_valid_inventory_for_combine = ( inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and formatted_inventory_string and isinstance(formatted_inventory_string, str) and "[Error" not in formatted_inventory_string and "[Disabled]" not in formatted_inventory_string and "[No Inventory Data Available]" not in formatted_inventory_string and "[Inventory Init/Config Error]" not in formatted_inventory_string )
                if is_valid_inventory_for_combine:
                    inventory_to_pass_to_combiner = formatted_inventory_string
                    _log_inventory_trace("Inventory is valid, preparing to pass to combiner.")
                else: _log_inventory_trace("Inventory is invalid or unavailable, not passing to combiner.")

                _log_inventory_trace(f"Combiner Input - final_selected_context Length: {len(context_for_prompt if isinstance(context_for_prompt, str) else '')}")
                _log_inventory_trace(f"Combiner Input - inventory_context Length: {len(inventory_to_pass_to_combiner if inventory_to_pass_to_combiner else '')}")

                # Pass world state attributes stored on 'self'
                combined_context_string = self._combine_context_func(
                    final_selected_context=(context_for_prompt if isinstance(context_for_prompt, str) else None),
                    t1_summaries=retrieved_t1_summaries,
                    t2_rag_results=retrieved_rag_summaries,
                    inventory_context=inventory_to_pass_to_combiner,
                    current_day=self.current_day,
                    current_time_of_day=self.current_time_of_day,
                    current_season=self.current_season,
                    current_weather=self.current_weather
                )
                _log_inventory_trace(f"Combiner Output Length: {len(combined_context_string)}\n------\n{combined_context_string}\n------")
            except Exception as e_combine:
                func_logger.error(f"[{session_id}] Error combining context: {e_combine}", exc_info=True);
                combined_context_string = "[Error combining context]"
                _log_inventory_trace(f"ERROR combining context: {e_combine}")
        else:
            func_logger.error(f"[{session_id}] Cannot combine context: Function unavailable.")
            _log_inventory_trace("ERROR: combine_context function unavailable.")
        _log_inventory_trace("--- END _prepare_and_refine_background ---")

        return ( combined_context_string, base_system_prompt_text, initial_owi_context_tokens, refined_context_tokens, cache_update_performed, cache_update_skipped, final_context_selection_performed, stateless_refinement_performed, formatted_inventory_string )


    async def _select_t0_history_slice(self, session_id: str, history_for_processing: List[Dict]) -> Tuple[List[Dict], int]:
        """ Selects the T0 history slice based on token limit and dialogue roles. """
        t0_raw_history_slice = []; t0_dialogue_tokens = -1; t0_token_limit = getattr(self.config, 't0_active_history_token_limit', 4000)
        try:
             if self._tokenizer:
                  t0_raw_history_slice = select_turns_for_t0( full_history=history_for_processing, target_tokens=t0_token_limit, tokenizer=self._tokenizer, dialogue_only_roles=self._dialogue_roles)
                  self.logger.info(f"[{session_id}] T0 Slice: Selected {len(t0_raw_history_slice)} dialogue msgs using select_turns_for_t0.")
             else:
                 self.logger.warning(f"[{session_id}] Tokenizer unavailable. Using simple turn count fallback for T0.")
                 fallback_turns = 10; dialogue_history = [msg for msg in history_for_processing if isinstance(msg, dict) and msg.get("role") in self._dialogue_roles]; start_idx = max(0, len(dialogue_history) - fallback_turns); t0_raw_history_slice = dialogue_history[start_idx:]
             if t0_raw_history_slice and self._count_tokens_func and self._tokenizer:
                 try: t0_dialogue_tokens = sum(self._count_tokens_func(msg["content"], self._tokenizer) for msg in t0_raw_history_slice if isinstance(msg, dict) and isinstance(msg.get("content"), str))
                 except Exception as e_tok_t0: t0_dialogue_tokens = -1; self.logger.error(f"[{session_id}] Error calculating T0 tokens: {e_tok_t0}")
             elif not t0_raw_history_slice: t0_dialogue_tokens = 0
             else: t0_dialogue_tokens = -1
        except Exception as e_select_t0: self.logger.error(f"[{session_id}] Error during T0 slice selection: {e_select_t0}", exc_info=True); t0_raw_history_slice = []; t0_dialogue_tokens = -1
        return t0_raw_history_slice, t0_dialogue_tokens


    async def _calculate_and_format_status(
        self, session_id: str, t1_retrieved_count: int, t2_retrieved_count: int,
        session_process_owi_rag: bool, final_context_selection_performed: bool,
        cache_update_skipped: bool, stateless_refinement_performed: bool,
        initial_owi_context_tokens: int, refined_context_tokens: int,
        summarization_prompt_tokens: int, summarization_output_tokens: int,
        t0_dialogue_tokens: int, inventory_prompt_tokens: int,
        final_llm_payload_contents: Optional[List[Dict]]
    ) -> Tuple[str, int]:
        """ Calculates final payload tokens and formats the status message string. """
        final_payload_tokens = -1
        if final_llm_payload_contents and self._count_tokens_func and self._tokenizer:
            try: final_payload_tokens = sum( self._count_tokens_func(part["text"], self._tokenizer) for turn in final_llm_payload_contents if isinstance(turn, dict) for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str))
            except Exception as e_tok_final: final_payload_tokens = -1; self.logger.error(f"[{session_id}] Error calculating final payload tokens: {e_tok_final}")
        elif not final_llm_payload_contents: final_payload_tokens = 0
        status_parts = []; status_parts.append(f"T1={t1_retrieved_count}"); status_parts.append(f"T2={t2_retrieved_count}")
        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False); enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False); refinement_indicator = None
        if enable_rag_cache_global and final_context_selection_performed: refinement_indicator = f"Cache(S1Skip={'Y' if cache_update_skipped else 'N'})"
        elif enable_stateless_refin_global and stateless_refinement_performed: refinement_indicator = "StatelessRef"
        if refinement_indicator: status_parts.append(refinement_indicator)
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
        status_message = ", ".join(status_parts) + token_string
        return status_message, final_payload_tokens


    async def _execute_or_prepare_output(
        self, session_id: str, body: Dict, final_llm_payload_contents: Optional[List[Dict]],
        event_emitter: Optional[Callable], status_message: str, final_payload_tokens: int
    ) -> OrchestratorResult: # Return type is now str | Dict
        """
        Executes the final LLM call using the LiteLLM adapter OR returns the
        constructed payload dictionary if the call is disabled by config.
        NO STREAMING.
        """
        output_body = body.copy() if isinstance(body, dict) else {}
        if not final_llm_payload_contents:
            self.logger.error(f"[{session_id}] Final payload construction failed.");
            await self._emit_status(event_emitter, session_id, "ERROR: Final payload preparation failed.", done=True);
            return {"error": "Orchestrator: Final payload construction failed.", "status_code": 500}

        output_body["messages"] = final_llm_payload_contents
        preserved_keys = ["model", "stream", "options", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"];
        keys_preserved = [k for k in preserved_keys if k in body];
        for k in keys_preserved: output_body[k] = body[k]
        self.logger.info(f"[{session_id}] Output body constructed/updated. Preserved keys: {keys_preserved}.")

        if getattr(self.config, 'debug_log_final_payload', False):
            self.logger.info(f"[{session_id}] Logging final constructed payload dict due to debug valve.")
            self._orchestrator_log_debug_payload(session_id, {"contents": final_llm_payload_contents})
        else:
            self.logger.debug(f"[{session_id}] Skipping final payload log: Debug valve is OFF.")

        final_url = getattr(self.config, 'final_llm_api_url', None)
        final_key = getattr(self.config, 'final_llm_api_key', None)
        url_present = bool(final_url and isinstance(final_url, str) and final_url.strip())
        key_present = bool(final_key and isinstance(final_key, str) and final_key.strip())
        self.logger.debug(f"[{session_id}] Checking Final LLM Trigger. URL Present:{url_present}, Key Present:{key_present}")
        final_llm_triggered = url_present and key_present

        if final_llm_triggered:
            self.logger.info(f"[{session_id}] Final LLM Call via Pipe TRIGGERED (Non-Streaming, using LiteLLM Adapter).")
            await self._emit_status(event_emitter, session_id, "Status: Executing final LLM Call...", done=False)
            final_temp = getattr(self.config, 'final_llm_temperature', 0.7);
            final_timeout = getattr(self.config, 'final_llm_timeout', 120);
            final_call_payload_google_fmt = {"contents": final_llm_payload_contents}

            success, response_or_error = await self._async_llm_call_wrapper(
                api_url=final_url, api_key=final_key, payload=final_call_payload_google_fmt,
                temperature=final_temp, timeout=final_timeout, caller_info=f"Orch_FinalLLM_{session_id}"
            )
            intermediate_status = "Status: Final LLM Complete" + (" (Success)" if success else " (Failed)")
            await self._emit_status(event_emitter, session_id, intermediate_status, done=False)

            if success and isinstance(response_or_error, str):
                self.logger.info(f"[{session_id}] Final LLM call successful. Returning response string.")
                return response_or_error
            elif not success and isinstance(response_or_error, dict):
                self.logger.error(f"[{session_id}] Final LLM call failed. Returning error dict: {response_or_error}")
                return response_or_error
            else:
                self.logger.error(f"[{session_id}] Final LLM adapter returned unexpected format. Success={success}, Type={type(response_or_error)}")
                return {"error": "Final LLM adapter returned unexpected result format.", "status_code": 500}
        else:
            self.logger.info(f"[{session_id}] Final LLM Call disabled by config. Returning constructed payload dict.")
            return {"messages": final_llm_payload_contents}


    async def process_turn(
        self,
        session_id: str,
        user_id: str,
        body: Dict,
        user_valves: Any,
        event_emitter: Optional[Callable],
        embedding_func: Optional[Callable] = None,
        chroma_embed_wrapper: Optional[Any] = None,
        is_regeneration_heuristic: bool = False
    ) -> OrchestratorResult:
        """ Processes a single turn by calling helper methods in sequence. """
        pipe_entry_time_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Orchestrator process_turn [{session_id}]: Started at {pipe_entry_time_iso} (Regen Flag: {is_regeneration_heuristic})")

        self.pipe_logger = getattr(self, 'pipe_logger', self.logger)
        self.pipe_debug_path_getter = getattr(self, 'pipe_debug_path_getter', None)

        inventory_enabled = getattr(self.config, 'enable_inventory_management', False)
        event_hints_enabled = getattr(self.config, 'enable_event_hints', False)

        self.logger.info(f"[{session_id}] Inventory Management Enabled (Global Valve): {inventory_enabled}")
        self.logger.info(f"[{session_id}] Event Hints Enabled (Global Valve): {event_hints_enabled}")
        self.logger.info(f"[{session_id}] Inventory Module Available (Local Import Check): {_ORCH_INVENTORY_MODULE_AVAILABLE}")
        self.logger.info(f"[{session_id}] World State Parser Available (Local Import Check): {_WORLD_STATE_PARSER_LLM_AVAILABLE}") # Check LLM parser flag

        summarization_performed = False; new_t1_summary_text = None; summarization_prompt_tokens = -1; summarization_output_tokens = -1; t1_retrieved_count = 0; t2_retrieved_count = 0; retrieved_rag_summaries = []; cache_update_performed = False; cache_update_skipped = False; final_context_selection_performed = False; stateless_refinement_performed = False; initial_owi_context_tokens = -1; refined_context_tokens = -1; t0_dialogue_tokens = -1; final_payload_tokens = -1; inventory_prompt_tokens = -1; formatted_inventory_string_for_status = ""; final_result: Optional[OrchestratorResult] = None; final_llm_payload_contents: Optional[List[Dict]] = None; inventory_update_completed = False; inventory_update_success_flag = False
        generated_event_hint: Optional[str] = None
        self.current_season: Optional[str] = "Unknown"
        self.current_weather: Optional[str] = "Unknown"
        self.current_day: Optional[int] = 1
        self.current_time_of_day: Optional[str] = "Unknown"

        try:
            await self._emit_status(event_emitter, session_id, "Status: Orchestrator syncing history...")
            incoming_messages = body.get("messages", [])
            stored_history = self.session_manager.get_active_history(session_id) or []
            if incoming_messages != stored_history: self.session_manager.set_active_history(session_id, incoming_messages.copy()); self.logger.debug(f"[{session_id}] Updating active history (Len: {len(incoming_messages)}).")
            else: self.logger.debug(f"[{session_id}] Incoming history matches stored.")
            current_active_history = self.session_manager.get_active_history(session_id) or []
            if not current_active_history: raise ValueError("Active history is empty after sync.")

            latest_user_query_str, history_for_processing = await self._determine_effective_query( session_id, current_active_history, is_regeneration_heuristic )
            if not latest_user_query_str and not is_regeneration_heuristic: raise ValueError("Cannot proceed without an effective user query (and not regeneration).")

            (summarization_performed, new_t1_summary_text, summarization_prompt_tokens, summarization_output_tokens) = await self._handle_tier1_summarization( session_id, user_id, current_active_history, is_regeneration_heuristic, event_emitter )

            await self._handle_tier2_transition( session_id, summarization_performed, chroma_embed_wrapper, event_emitter )

            recent_t1_summaries, t1_retrieved_count = await self._get_t1_summaries(session_id)
            retrieved_rag_summaries, t2_retrieved_count = await self._get_t2_rag_results( session_id, history_for_processing, latest_user_query_str, embedding_func, chroma_embed_wrapper, event_emitter )

            await self._emit_status(event_emitter, session_id, "Status: Fetching world state...")
            world_state_db_data = None
            default_season = "Summer"; default_weather = "Clear"; default_day = 1; default_time = "Morning"
            self.current_season = default_season; self.current_weather = default_weather; self.current_day = default_day; self.current_time_of_day = default_time

            if self.sqlite_cursor and self._get_world_state_db_func and self._set_world_state_db_func:
                try:
                    world_state_db_data = await self._get_world_state_db_func(self.sqlite_cursor, session_id)
                    if world_state_db_data and isinstance(world_state_db_data, dict):
                        self.current_season = world_state_db_data.get("season") or default_season
                        self.current_weather = world_state_db_data.get("weather") or default_weather
                        self.current_day = world_state_db_data.get("day") or default_day
                        self.current_time_of_day = world_state_db_data.get("time_of_day") or default_time
                        self.logger.info(f"[{session_id}] Successfully fetched world state: Season='{self.current_season}', Weather='{self.current_weather}', Day='{self.current_day}', Time='{self.current_time_of_day}'")
                    else:
                        self.logger.info(f"[{session_id}] No world state found. Initializing with defaults.")
                        set_success = await self._set_world_state_db_func( self.sqlite_cursor, session_id, self.current_season, self.current_weather, self.current_day, self.current_time_of_day )
                        if set_success: self.logger.info(f"[{session_id}] Successfully initialized and saved default world state.")
                        else: self.logger.error(f"[{session_id}] Failed to initialize/save world state in DB.")
                except Exception as e_world_state:
                    self.logger.error(f"[{session_id}] Error fetching/setting world state: {e_world_state}", exc_info=True)
                    self.current_season = "Unknown (Error)"; self.current_weather = "Unknown (Error)"; self.current_day = 1; self.current_time_of_day = "Unknown (Error)"
            else:
                missing_world_funcs = [f for f, fn in {"cursor": self.sqlite_cursor, "getter": self._get_world_state_db_func, "setter": self._set_world_state_db_func}.items() if not fn]
                self.logger.error(f"[{session_id}] Cannot fetch/set world state: Missing prerequisites: {missing_world_funcs}. Using defaults.")

            (combined_context_string, base_system_prompt_text, initial_owi_context_tokens, refined_context_tokens, cache_update_performed, cache_update_skipped, final_context_selection_performed, stateless_refinement_performed, formatted_inventory_string_for_status ) = await self._prepare_and_refine_background( session_id, body, user_valves, recent_t1_summaries, retrieved_rag_summaries, current_active_history, latest_user_query_str, event_emitter )

            t0_raw_history_slice, t0_dialogue_tokens = await self._select_t0_history_slice( session_id, history_for_processing )

            if event_hints_enabled:
                self.logger.info(f"[{session_id}] Attempting event hint generation. Import Success: {_event_hints_import_success}, Import Path: '{_event_hints_import_path}'")
                self.logger.info(f"[{session_id}] Event Hints feature enabled. Attempting generation...")
                await self._emit_status(event_emitter, session_id, "Status: Checking for dynamic event hints...")
                try:
                    generated_event_hint = await generate_event_hint(
                        config=self.config, history_messages=current_active_history, background_context=combined_context_string,
                        current_season=self.current_season, current_weather=self.current_weather, current_time_of_day=self.current_time_of_day,
                        llm_call_func=self._async_llm_call_wrapper, logger_instance=self.logger, session_id=session_id
                    )
                    if generated_event_hint:
                         self.logger.info(f"[{session_id}] Event Hint Generated: '{generated_event_hint[:80]}...'")
                         await self._emit_status(event_emitter, session_id, "Status: Event hint generated.")
                    else:
                         self.logger.info(f"[{session_id}] No event hint suggested by Hint LLM.")
                         await self._emit_status(event_emitter, session_id, "Status: No event hint suggested.")
                except Exception as e_hint_gen:
                    self.logger.error(f"[{session_id}] Error during event hint generation call: {e_hint_gen}", exc_info=True)
                    generated_event_hint = None
            else:
                self.logger.debug(f"[{session_id}] Event Hints feature disabled. Skipping hint generation.")

            await self._emit_status(event_emitter, session_id, "Status: Constructing final request...")
            payload_dict_or_error = standalone_construct_payload(
                system_prompt=base_system_prompt_text, history=t0_raw_history_slice, context=combined_context_string,
                query=latest_user_query_str, long_term_goal=getattr(user_valves, 'long_term_goal', ''), event_hint=generated_event_hint,
                strategy="standard", include_ack_turns=getattr(self.config, 'include_ack_turns', True),
            )
            if isinstance(payload_dict_or_error, dict) and "contents" in payload_dict_or_error:
                final_llm_payload_contents = payload_dict_or_error["contents"]
                self.logger.info(f"[{session_id}] Constructed final payload using standalone function ({len(final_llm_payload_contents)} turns).")
            else:
                error_msg = payload_dict_or_error.get("error", "Unknown payload construction error") if isinstance(payload_dict_or_error, dict) else "Invalid return type"
                self.logger.error(f"[{session_id}] Standalone payload constructor failed: {error_msg}")
                final_llm_payload_contents = None

            final_result = await self._execute_or_prepare_output(
                 session_id=session_id, body=body, final_llm_payload_contents=final_llm_payload_contents,
                 event_emitter=event_emitter, status_message="Status: Core processing complete.", final_payload_tokens=-1
            )

            # --- MODIFIED WORLD STATE PARSING & UPDATE BLOCK ---
            if isinstance(final_result, str) and self._parse_world_state_func and self.sqlite_cursor and self._set_world_state_db_func and _WORLD_STATE_PARSER_LLM_AVAILABLE:
                 self.logger.info(f"[{session_id}] Analyzing final response for world state changes (using LLM)...")
                 detected_changes = {}
                 try:
                     # --- Prepare Config for World State Parse LLM ---
                     ws_parse_llm_config = {
                         "url": getattr(self.config, 'event_hint_llm_api_url', None), # Use Event Hint URL
                         "key": getattr(self.config, 'event_hint_llm_api_key', None), # Use Event Hint Key
                         "temp": getattr(self.config, 'event_hint_llm_temperature', 0.3), # Use Event Hint Temp (or default 0.3)
                         "prompt_template": getattr(self.config, 'world_state_parse_prompt_template', DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT) # NEW Valve
                     }
                     # --- Prepare Current State Dict ---
                     current_state_dict = {
                         "day": self.current_day,
                         "time_of_day": self.current_time_of_day,
                         "weather": self.current_weather,
                         "season": self.current_season
                     }
                     # --- Call the new async parser function ---
                     # <<< MODIFIED CALL: Pass debug path getter >>>
                     detected_changes = await self._parse_world_state_func(
                         llm_response_text=final_result,
                         history_messages=current_active_history,
                         current_state=current_state_dict,
                         llm_call_func=self._async_llm_call_wrapper,
                         ws_parse_llm_config=ws_parse_llm_config,
                         logger_instance=self.logger,
                         session_id=session_id,
                         debug_path_getter=self._orchestrator_get_debug_log_path # <<< PASSING GETTER >>>
                     )
                     # <<< END MODIFIED CALL >>>
                 except Exception as e_parse:
                     self.logger.error(f"[{session_id}] Error calling world state parser LLM function: {e_parse}", exc_info=True)

                 if detected_changes:
                     self.logger.debug(f"[{session_id}] LLM Parser detected changes: {detected_changes}")
                     # Apply changes to current state for comparison and potential update
                     new_day = self.current_day + detected_changes.get('day_increment', 0)
                     new_time = detected_changes.get('time_of_day', self.current_time_of_day)
                     new_weather = detected_changes.get('weather', self.current_weather)
                     new_season = detected_changes.get('season', self.current_season)

                     # Check if any state actually changed compared to current state
                     state_changed = not (
                         new_day == self.current_day and
                         new_time == self.current_time_of_day and
                         new_weather == self.current_weather and
                         new_season == self.current_season
                     )

                     if state_changed:
                         self.logger.info(f"[{session_id}] World state change detected by LLM Parser. Attempting DB update.")
                         await self._emit_status(event_emitter, session_id, "Status: Updating world state...", done=False)
                         try:
                             self.logger.debug(f"[{session_id}] Attempting DB update with: Day={new_day}, Time='{new_time}', Weather='{new_weather}', Season='{new_season}'")
                             update_success = await self._set_world_state_db_func(
                                 self.sqlite_cursor, session_id, new_season, new_weather, new_day, new_time
                             )
                             self.logger.debug(f"[{session_id}] DB update success status: {update_success}")

                             if update_success:
                                 self.logger.info(f"[{session_id}] World state successfully updated in DB.")
                                 # Update the orchestrator's internal state to match DB
                                 self.current_day = new_day
                                 self.current_time_of_day = new_time
                                 self.current_weather = new_weather
                                 self.current_season = new_season
                             else:
                                 self.logger.error(f"[{session_id}] Failed to save updated world state to DB (set function returned False).")
                         except Exception as e_set_world:
                             self.logger.error(f"[{session_id}] Error saving updated world state: {e_set_world}", exc_info=True)
                     else:
                         self.logger.info(f"[{session_id}] LLM Parser ran, but no net change detected compared to current world state.")
                 else:
                     self.logger.info(f"[{session_id}] No world state changes detected by LLM parser (empty dict returned).")

            elif not isinstance(final_result, str):
                self.logger.debug(f"[{session_id}] Skipping world state analysis: Final result was not a string (Type: {type(final_result).__name__}).")
            elif not _WORLD_STATE_PARSER_LLM_AVAILABLE:
                self.logger.warning(f"[{session_id}] Skipping world state analysis: LLM Parser function unavailable (import failed).")
            elif not self.sqlite_cursor or not self._set_world_state_db_func:
                 missing_ws_update = [f for f, fn in {"cursor": self.sqlite_cursor, "setter": self._set_world_state_db_func}.items() if not fn]
                 self.logger.warning(f"[{session_id}] Skipping world state update: Missing prerequisites: {missing_ws_update}")
            # <<< --- END WORLD STATE PARSING & UPDATE BLOCK --- >>>


            if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and self._update_inventories_func:
                inventory_update_completed = True
                if isinstance(final_result, str):
                    self.logger.info(f"[{session_id}] Performing post-turn inventory update...")
                    await self._emit_status(event_emitter, session_id, "Status: Updating inventory state...", done=False)
                    try:
                        inv_llm_url = getattr(self.config, 'inv_llm_api_url', None); inv_llm_key = getattr(self.config, 'inv_llm_api_key', None); inv_llm_temp = getattr(self.config, 'inv_llm_temperature', 0.3); inv_llm_prompt_template = getattr(self.config, 'inv_llm_prompt_template', None); template_seems_valid = inv_llm_prompt_template and isinstance(inv_llm_prompt_template, str) and len(inv_llm_prompt_template) > 50
                        if not inv_llm_url or not inv_llm_key or not template_seems_valid: self.logger.error(f"[{session_id}] Inventory LLM config missing/invalid."); inventory_update_success_flag = False
                        else:
                            inv_llm_config = {"url": inv_llm_url, "key": inv_llm_key, "temp": inv_llm_temp, "prompt_template": inv_llm_prompt_template}
                            history_for_inv_update_list = self._get_recent_turns_func(current_active_history, 4, exclude_last=False); history_for_inv_update_str = self._format_history_func(history_for_inv_update_list)
                            inv_prompt_text = format_inventory_update_prompt( main_llm_response=final_result, user_query=latest_user_query_str, recent_history_str=history_for_inv_update_str, template=inv_llm_config['prompt_template'])
                            if inv_prompt_text and not inv_prompt_text.startswith("[Error") and self._count_tokens_func and self._tokenizer:
                                try: inventory_prompt_tokens = self._count_tokens_func(inv_prompt_text, self._tokenizer); self.logger.debug(f"[{session_id}] Calculated Inventory Prompt Tokens: {inventory_prompt_tokens}")
                                except Exception as e_inv_tok: self.logger.error(f"[{session_id}] Failed to calculate inventory prompt tokens: {e_inv_tok}"); inventory_prompt_tokens = -1
                            else: inventory_prompt_tokens = -1
                            if not self.sqlite_cursor or not self.sqlite_cursor.connection: self.logger.error(f"[{session_id}] Cannot update inventory: SQLite cursor invalid."); inventory_update_success_flag = False
                            else:
                                 new_cursor = None
                                 try:
                                     new_cursor = self.sqlite_cursor.connection.cursor()
                                     # --- MODIFIED CALL for Debug Logging ---
                                     if getattr(self.config, 'debug_log_final_payload', False):
                                         self.logger.debug(f"[{session_id}] Calling Orch inventory logger for PROMPT.")
                                         self._orchestrator_log_debug_inventory_llm(session_id, inv_prompt_text, is_prompt=True)
                                     # --- END MODIFIED CALL ---

                                     update_success = await self._update_inventories_func(
                                         cursor=new_cursor, session_id=session_id, main_llm_response=final_result, user_query=latest_user_query_str, recent_history_str=history_for_inv_update_str,
                                         llm_call_func=self._async_llm_call_wrapper,
                                         db_get_inventory_func=get_character_inventory_data,
                                         db_update_inventory_func=add_or_update_character_inventory,
                                         inventory_llm_config=inv_llm_config,
                                     )
                                     inventory_update_success_flag = update_success
                                     if update_success: self.logger.info(f"[{session_id}] Post-turn inventory update successful.")
                                     else: self.logger.warning(f"[{session_id}] Post-turn inventory update function returned False.")
                                 except Exception as e_inv_update_inner: self.logger.error(f"[{session_id}] Error during inventory update call: {e_inv_update_inner}", exc_info=True); inventory_update_success_flag = False
                                 finally:
                                      if new_cursor:
                                          try: new_cursor.close(); self.logger.debug(f"[{session_id}] Inventory update cursor closed.")
                                          except Exception as e_close_cursor: self.logger.error(f"[{session_id}] Error closing inventory update cursor: {e_close_cursor}")
                    except Exception as e_inv_update_outer: self.logger.error(f"[{session_id}] Outer error during inventory update setup: {e_inv_update_outer}", exc_info=True); inventory_update_success_flag = False
                elif isinstance(final_result, dict) and "error" in final_result: self.logger.warning(f"[{session_id}] Skipping inventory update due to upstream error: {final_result.get('error')}"); inventory_update_completed = False
                elif isinstance(final_result, dict) and "messages" in final_result: self.logger.info(f"[{session_id}] Skipping inventory update: Final LLM call was disabled or skipped."); inventory_update_completed = False
                else: self.logger.error(f"[{session_id}] Unexpected type for final_result: {type(final_result)}. Skipping inventory update."); inventory_update_completed = False
            elif inventory_enabled and not _ORCH_INVENTORY_MODULE_AVAILABLE: self.logger.warning(f"[{session_id}] Skipping inventory update: Module import failed."); inventory_update_completed = False; inventory_update_success_flag = False;
            elif inventory_enabled and not self._update_inventories_func: self.logger.error(f"[{session_id}] Skipping inventory update: Update function alias is None."); inventory_update_completed = False; inventory_update_success_flag = False;
            else: self.logger.debug(f"[{session_id}] Skipping inventory update: Disabled by global valve."); inventory_update_completed = False; inventory_update_success_flag = False;

            if final_llm_payload_contents and self._count_tokens_func and self._tokenizer:
                try: final_payload_tokens = sum(self._count_tokens_func(part["text"], self._tokenizer) for turn in final_llm_payload_contents if isinstance(turn, dict) for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str))
                except Exception: final_payload_tokens = -1
            elif not final_llm_payload_contents: final_payload_tokens = 0
            else: final_payload_tokens = -1

            final_status_string_base, _ = await self._calculate_and_format_status( session_id=session_id, t1_retrieved_count=t1_retrieved_count, t2_retrieved_count=t2_retrieved_count, session_process_owi_rag=bool(getattr(user_valves, 'process_owi_rag', True)), final_context_selection_performed=final_context_selection_performed, cache_update_skipped=cache_update_skipped, stateless_refinement_performed=stateless_refinement_performed, initial_owi_context_tokens=initial_owi_context_tokens, refined_context_tokens=refined_context_tokens, summarization_prompt_tokens=summarization_prompt_tokens, summarization_output_tokens=summarization_output_tokens, t0_dialogue_tokens=t0_dialogue_tokens, inventory_prompt_tokens=inventory_prompt_tokens, final_llm_payload_contents=final_llm_payload_contents)

            world_state_status = f"| World: D{self.current_day} {self.current_time_of_day} {self.current_weather} {self.current_season}"
            inv_stat_indicator = "Inv=OFF";
            if inventory_enabled:
                 if not _ORCH_INVENTORY_MODULE_AVAILABLE: inv_stat_indicator = "Inv=MISSING"
                 else:
                     if not inventory_update_completed: inv_stat_indicator = "Inv=SKIP"
                     elif inventory_update_success_flag: inv_stat_indicator = "Inv=OK"
                     else: inv_stat_indicator = "Inv=FAIL"
            final_status_string = final_status_string_base + world_state_status + f" | {inv_stat_indicator}"
            self.logger.debug(f"[{session_id}] Orchestrator: About to emit FINAL status: '{final_status_string}'")
            await self._emit_status(event_emitter, session_id, final_status_string, done=True)

            pipe_end_time_iso = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"Orchestrator process_turn [{session_id}]: Finished at {pipe_end_time_iso}")
            if final_result is None: raise RuntimeError("Internal processing error, final result was None.")
            return final_result

        except asyncio.CancelledError:
            self.logger.info(f"[{session_id or 'unknown'}] Orchestrator process_turn cancelled."); await self._emit_status(event_emitter, session_id or 'unknown', "Status: Processing cancelled.", done=True); raise
        except ValueError as ve:
             session_id_for_log = session_id if 'session_id' in locals() else 'unknown'
             self.logger.error(f"[{session_id_for_log}] Orchestrator ValueError in process_turn: {ve}")
             try: await self._emit_status(event_emitter, session_id_for_log, f"ERROR: {ve}", done=True)
             except: pass
             return {"error": f"Orchestrator failed: {ve}", "status_code": 500}
        except Exception as e_orch:
            session_id_for_log = session_id if 'session_id' in locals() else 'unknown'
            self.logger.critical(f"[{session_id_for_log}] Orchestrator UNHANDLED EXCEPTION in process_turn: {e_orch}", exc_info=True)
            try: await self._emit_status(event_emitter, session_id_for_log, f"ERROR: Orchestrator Failed ({type(e_orch).__name__})", done=True)
            except: pass
            return {"error": f"Orchestrator failed: {type(e_orch).__name__}", "status_code": 500}


# === END OF FILE i4_llm_agent/orchestration.py ===