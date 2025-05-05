# [[START MODIFIED orchestration.py - Regen Skip Logic v0.2.3]]
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
import functools # Added for lambda binding clarity

# --- Standard Library Imports ---
import urllib.parse

# --- i4_llm_agent Imports ---
from .session import SessionManager
from .database import (
    add_tier1_summary, get_recent_tier1_summaries, get_tier1_summary_count,
    get_oldest_tier1_summary, delete_tier1_summary, get_max_t1_end_index,
    get_oldest_t1_batch, delete_t1_batch,
    add_aged_summary, get_recent_aged_summaries,
    get_or_create_chroma_collection, add_to_chroma_collection,
    CHROMADB_AVAILABLE, ChromaEmbeddingFunction, ChromaCollectionType,
    InvalidDimensionException,
    get_character_inventory_data, add_or_update_character_inventory,
    get_world_state, set_world_state,
    get_scene_state, set_scene_state,
)
from .history import (
    format_history_for_llm, get_recent_turns, DIALOGUE_ROLES, select_turns_for_t0
)
from .memory import manage_tier1_summarization
from .api_client import call_google_llm_api

# === Utility Imports (Token counting, Similarity, AND NEW Logging funcs) ===
try:
    from .utils import (
        TIKTOKEN_AVAILABLE,
        count_tokens,
        calculate_string_similarity,
        get_debug_log_path,
        log_debug_payload,
        awaitable_log_inventory_debug,
    )
    _UTILS_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported utils module.")
except ImportError as e_utils:
    TIKTOKEN_AVAILABLE = False
    def count_tokens(*args, **kwargs): return 0
    def calculate_string_similarity(*args, **kwargs): return 0.0
    def get_debug_log_path(*args, **kwargs): return None
    def log_debug_payload(*args, **kwargs): pass
    async def awaitable_log_inventory_debug(*args, **kwargs): await asyncio.sleep(0); pass
    _UTILS_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Orchestration: Failed to import utils: {e_utils}. Some features may be disabled.")

# --- Prompting Imports ---
from .prompting import (
    format_inventory_update_prompt,
    format_memory_aging_prompt,
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
    DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE,
)

# === Event Hint Import ===
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
    async def generate_event_hint(*args, **kwargs):
        logging.getLogger(__name__).error("Executing FALLBACK generate_event_hint due to import error.")
        await asyncio.sleep(0)
        return None, {"previous_weather": None, "new_weather": None}
    EVENT_HANDLING_GUIDELINE_TEXT = "[EVENT GUIDELINE LOAD FAILED]"
    def format_hint_for_query(hint): return f"[[Hint Load Failed: {hint}]]"
    DEFAULT_EVENT_HINT_TEMPLATE_TEXT = "[Default Event Hint Template Load Failed]"

# === Unified State Assessment Import ===
try:
    from .state_assessment import update_state_via_full_turn_assessment, DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT
    _UNIFIED_STATE_ASSESSMENT_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported state_assessment module.")
except ImportError as e_state_assess:
    logging.getLogger(__name__).error(f"Failed to import state_assessment module: {e_state_assess}", exc_info=True)
    _UNIFIED_STATE_ASSESSMENT_AVAILABLE = False
    DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT = "[Default Unified State Assessment Template Load Failed]"
    async def update_state_via_full_turn_assessment(
        session_id: str, previous_world_state: Dict[str, Any], previous_scene_state: Dict[str, Any],
        current_user_query: str, assistant_response_text: str, history_messages: List[Dict],
        llm_call_func: Callable, state_assessment_llm_config: Dict[str, Any],
        logger_instance: Optional[logging.Logger] = None, event_emitter: Optional[Callable] = None,
        weather_proposal: Optional[Dict[str, Optional[str]]] = None
    ) -> Dict[str, Any]:
        lg = logger_instance or logging.getLogger(__name__)
        lg.error(f"[{session_id}] Executing FALLBACK update_state_via_full_turn_assessment due to import error.")
        await asyncio.sleep(0)
        return {
            "new_day": previous_world_state.get("day", 1),
            "new_time_of_day": previous_world_state.get("time_of_day", "Morning"),
            "new_weather": previous_world_state.get("weather", "Clear"),
            "new_season": previous_world_state.get("season", "Summer"),
            "new_scene_keywords": previous_scene_state.get("keywords", []),
            "new_scene_description": previous_scene_state.get("description", ""),
            "scene_changed_flag": False
        }

# === Inventory Module Import ===
try:
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

# === Context Processor Import ===
try:
    from .context_processor import process_context_and_prepare_payload
    _CONTEXT_PROCESSOR_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported context_processor module.")
except ImportError as e_ctx_proc:
    _CONTEXT_PROCESSOR_AVAILABLE = False
    logging.getLogger(__name__).error(f"Failed to import context_processor: {e_ctx_proc}", exc_info=True)
    async def process_context_and_prepare_payload(*args, **kwargs) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
        lg = kwargs.get('logger') or logging.getLogger(__name__)
        session_id = kwargs.get('session_id', 'unknown')
        lg.error(f"[{session_id}] Executing FALLBACK process_context_and_prepare_payload due to import error.")
        await asyncio.sleep(0)
        return None, {"error": "Context processor unavailable"}


logger = logging.getLogger(__name__) # i4_llm_agent.orchestration

OrchestratorResult = Union[Dict, str]

# ==============================================================================
# === Session Pipe Orchestrator Class (Regen Skip Logic Integrated v0.2.3) ===
# ==============================================================================

class SessionPipeOrchestrator:
    """
    Orchestrates the core processing logic of the Session Memory Pipe.
    Implements two-stage state assessment and memory aging.
    Uses utility functions for debug logging.
    Includes logic to skip certain steps during regeneration.
    Version: 0.2.3
    """
    version = "0.2.3"

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

        # --- Log Config ---
        self.logger.debug(f"Orchestrator v{self.version} __init__: Received config object.")
        try:
            config_dump = {}
            if hasattr(config, 'model_dump'): config_dump = config.model_dump()
            elif hasattr(config, 'dict'): config_dump = config.dict()
            elif hasattr(config, '__dict__'): config_dump = config.__dict__
            safe_config_log = {
                k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                for k, v in config_dump.items()
                if "api_key" not in k.lower() and "prompt" not in k.lower()
            }
            self.logger.debug(f"Orchestrator __init__: Received config values (filtered): {safe_config_log}")
            self.aging_trigger_threshold = getattr(self.config, 'aging_trigger_threshold', 15)
            self.aging_batch_size = getattr(self.config, 'aging_batch_size', 5)
            self.logger.info(f"Memory Aging Config: Trigger Threshold={self.aging_trigger_threshold}, Batch Size={self.aging_batch_size}")
            if self.aging_trigger_threshold <= self.aging_batch_size:
                 self.logger.warning(f"Memory Aging Config: Trigger threshold ({self.aging_trigger_threshold}) should ideally be greater than batch size ({self.aging_batch_size}) to ensure batches are available.")
            max_t1 = getattr(self.config, 'max_stored_summary_blocks', 20)
            if self.aging_trigger_threshold >= max_t1:
                 self.logger.warning(f"Memory Aging Config: Trigger threshold ({self.aging_trigger_threshold}) >= Max T1 Blocks ({max_t1}). Aging may not run before T2 push.")
        except Exception as e_dump:
            self.logger.error(f"Orchestrator __init__: Error processing config: {e_dump}")
            self.aging_trigger_threshold = 15
            self.aging_batch_size = 5

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

        # --- Function Aliases ---
        self._llm_call_func = call_google_llm_api
        self._format_history_func = format_history_for_llm
        self._get_recent_turns_func = get_recent_turns
        self._manage_memory_func = manage_tier1_summarization
        self._count_tokens_func = count_tokens
        self._calculate_similarity_func = calculate_string_similarity
        self._dialogue_roles = DIALOGUE_ROLES
        self._get_world_state_db_func = get_world_state; self._set_world_state_db_func = set_world_state
        self._get_scene_state_db_func = get_scene_state; self._set_scene_state_db_func = set_scene_state
        self._get_t1_count_db_func = get_tier1_summary_count
        self._get_oldest_t1_batch_db_func = get_oldest_t1_batch
        self._add_aged_summary_db_func = add_aged_summary
        self._delete_t1_batch_db_func = delete_t1_batch
        self._generate_hint_func = generate_event_hint if _EVENT_HINTS_AVAILABLE else None
        self._unified_state_func = update_state_via_full_turn_assessment if _UNIFIED_STATE_ASSESSMENT_AVAILABLE else None
        if _ORCH_INVENTORY_MODULE_AVAILABLE: self._update_inventories_func = _real_update_inventories_func
        else: self._update_inventories_func = _dummy_update_inventories
        self._context_processor_func = process_context_and_prepare_payload if _CONTEXT_PROCESSOR_AVAILABLE else None
        self._format_aging_prompt_func = format_memory_aging_prompt

        self.logger.info(f"SessionPipeOrchestrator v{self.version} initialized (Regen Skip Logic).")
        self.logger.info(f"Utils Available Check (Init): {_UTILS_AVAILABLE}")
        self.logger.info(f"Unified State Assessment Status Check (Init): Available={_UNIFIED_STATE_ASSESSMENT_AVAILABLE}")
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
        if event_emitter and callable(event_emitter) and getattr(self.config, 'emit_status_updates', True):
            try:
                status_data = { "type": "status", "data": {"description": str(description), "done": bool(done)} }
                if asyncio.iscoroutinefunction(event_emitter): await event_emitter(status_data)
                else: event_emitter(status_data)
            except Exception as e_emit: self.logger.warning(f"[{session_id}] Orchestrator failed to emit status '{description}': {e_emit}")
        else: self.logger.debug(f"[{session_id}] Orchestrator status update (not emitted): '{description}' (Done: {done})")

    # --- Internal Helper: Async LLM Call Wrapper ---
    async def _async_llm_call_wrapper(
        self, api_url: str, api_key: str, payload: Dict[str, Any], temperature: float,
        timeout: int = 90, caller_info: str = "Orchestrator_LLM",
    ) -> Tuple[bool, Union[str, Dict]]:
        """ Wraps the library's LLM call function for error handling. """
        if not self._llm_call_func: self.logger.error(f"[{caller_info}] LLM func alias unavailable."); return False, {"error_type": "SetupError", "message": "LLM func alias unavailable"}
        if not asyncio.iscoroutinefunction(self._llm_call_func): self.logger.critical(f"[{caller_info}] LLM func alias is NOT async!"); return False, {"error_type": "SetupError", "message": "LLM func alias is not async"}
        try:
            self.logger.debug(f"[{caller_info}] Awaiting result from LLM adapter function.")
            success, result_or_error = await self._llm_call_func(api_url=api_url, api_key=api_key, payload=payload, temperature=temperature, timeout=timeout, caller_info=caller_info)
            self.logger.debug(f"[{caller_info}] LLM adapter returned (Success: {success}).")
            return success, result_or_error
        except asyncio.CancelledError: self.logger.info(f"[{caller_info}] LLM call cancelled."); raise
        except Exception as e: self.logger.error(f"Orchestrator LLM Wrapper Error [{caller_info}]: Uncaught exception: {e}", exc_info=True); return False, {"error_type": "AsyncWrapperError", "message": f"{type(e).__name__}: {str(e)}"}

    # --- Helper Methods for process_turn ---

    # === _determine_effective_query (Unchanged) ===
    async def _determine_effective_query(
        self, session_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool
    ) -> Tuple[str, List[Dict], Optional[str]]:
        """ Determines the effective user query, history slice, and last assistant response. """
        effective_user_message_index = -1; last_assistant_message_str: Optional[str] = None
        history_for_processing: List[Dict] = []; latest_user_query_str: str = ""
        if not current_active_history: self.logger.error(f"[{session_id}] Cannot determine query: Active history is empty."); return "", [], None
        user_message_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") == "user"]
        if not user_message_indices:
            self.logger.error(f"[{session_id}] No user messages found in history."); history_for_processing = current_active_history
            assistant_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") in ("assistant", "model")]
            if assistant_indices: last_assistant_msg = current_active_history[assistant_indices[-1]]; last_assistant_message_str = last_assistant_msg.get("content") if isinstance(last_assistant_msg, dict) else None
            return "", history_for_processing, last_assistant_message_str
        if is_regeneration_heuristic:
            effective_user_message_index = user_message_indices[-2] if len(user_message_indices) >= 2 else user_message_indices[-1]
            log_level = self.logger.info if len(user_message_indices) >= 2 else self.logger.warning
            log_level(f"[{session_id}] Regen: Using user message at index {effective_user_message_index} as query base.")
        else: effective_user_message_index = user_message_indices[-1]; self.logger.debug(f"[{session_id}] Normal: Using user message at index {effective_user_message_index} as query base.")
        if effective_user_message_index < 0 or effective_user_message_index >= len(current_active_history): self.logger.error(f"[{session_id}] Effective user index {effective_user_message_index} out of bounds for history len {len(current_active_history)}."); return "", [], None
        effective_user_message = current_active_history[effective_user_message_index]; history_for_processing = current_active_history[:effective_user_message_index]
        latest_user_query_str = effective_user_message.get("content", "") if isinstance(effective_user_message, dict) else ""
        assistant_indices_in_slice = [i for i, msg in enumerate(history_for_processing) if isinstance(msg, dict) and msg.get("role") in ("assistant", "model")]
        if assistant_indices_in_slice: last_assistant_msg_in_slice = history_for_processing[assistant_indices_in_slice[-1]]; last_assistant_message_str = last_assistant_msg_in_slice.get("content") if isinstance(last_assistant_msg_in_slice, dict) else None; self.logger.debug(f"[{session_id}] Found last assistant message at index {assistant_indices_in_slice[-1]} before query.")
        else: self.logger.debug(f"[{session_id}] No assistant message found before the effective user query."); last_assistant_message_str = None
        self.logger.debug(f"[{session_id}] Effective query set (len: {len(latest_user_query_str)}). History slice len: {len(history_for_processing)}. Last assistant msg len: {len(last_assistant_message_str or '')}.")
        return latest_user_query_str, history_for_processing, last_assistant_message_str

    # === _handle_tier1_summarization (Unchanged - already handles regen flag) ===
    async def _handle_tier1_summarization(
        self, session_id: str, user_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool, event_emitter: Optional[Callable]
    ) -> Tuple[bool, Optional[str], int, int]:
        """ Handles checking for and executing Tier 1 summarization. """
        await self._emit_status(event_emitter, session_id, "Status: Checking summarization...")
        summarization_performed_successfully = False; generated_summary = None; summarization_prompt_tokens = -1; summarization_output_tokens = -1
        summarizer_url = getattr(self.config, 'summarizer_api_url', None); summarizer_key = getattr(self.config, 'summarizer_api_key', None)
        can_summarize = all([ self._manage_memory_func, self._tokenizer, self._count_tokens_func, self.sqlite_cursor, self._async_llm_call_wrapper, summarizer_url, summarizer_key, current_active_history ])
        if not can_summarize:
             missing_prereqs = [p for p, v in {"manage_func": self._manage_memory_func, "tokenizer": self._tokenizer, "count_func": self._count_tokens_func, "db_cursor": self.sqlite_cursor, "llm_wrapper": self._async_llm_call_wrapper, "summ_url": summarizer_url, "summ_key": summarizer_key, "history": bool(current_active_history)}.items() if not v]
             self.logger.warning(f"[{session_id}] Skipping T1 check: Missing prerequisites: {', '.join(missing_prereqs)}."); return False, None, -1, -1
        summarizer_llm_config = { "url": summarizer_url, "key": summarizer_key, "temp": getattr(self.config, 'summarizer_temperature', 0.5), }
        self.logger.debug(f"[{session_id}] Orchestrator: Passing summarizer config (URL/Key/Temp) to memory manager.")
        new_last_summary_idx = -1; prompt_tokens = -1; t0_end_idx = -1; db_max_index = None; current_last_summary_index_for_memory = -1
        try:
            db_max_index = await get_max_t1_end_index(self.sqlite_cursor, session_id)
            if isinstance(db_max_index, int) and db_max_index >= 0: current_last_summary_index_for_memory = db_max_index; self.logger.debug(f"[{session_id}] T1: Start Index from DB: {current_last_summary_index_for_memory}")
            else: self.logger.debug(f"[{session_id}] T1: No valid start index in DB. Starting from -1.")
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

    # === _handle_memory_aging (Unchanged) ===
    async def _handle_memory_aging(
        self, session_id: str, user_id: str, event_emitter: Optional[Callable]
    ) -> bool:
        """ Handles checking for and executing Memory Aging (Condensing T1 batch). """
        await self._emit_status(event_emitter, session_id, "Status: Checking memory aging...")
        trigger_threshold = self.aging_trigger_threshold
        batch_size = self.aging_batch_size
        if trigger_threshold <= 0 or batch_size <= 0: self.logger.debug(f"[{session_id}] Skipping Aging: Trigger/Batch invalid."); return False

        aging_llm_url = getattr(self.config, 'summarizer_api_url', None); aging_llm_key = getattr(self.config, 'summarizer_api_key', None)
        aging_temp = getattr(self.config, 'summarizer_temperature', 0.5); aging_prompt_template = DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE
        can_age = all([ self.sqlite_cursor, self._get_t1_count_db_func, self._get_oldest_t1_batch_db_func, self._add_aged_summary_db_func, self._delete_t1_batch_db_func, self._async_llm_call_wrapper, aging_llm_url, aging_llm_key, self._format_aging_prompt_func, aging_prompt_template != "[Default Memory Aging Prompt Load Failed]" ])
        if not can_age: self.logger.warning(f"[{session_id}] Skipping Aging check: Missing prerequisites."); return False

        try:
            current_t1_count = await self._get_t1_count_db_func(self.sqlite_cursor, session_id)
            if current_t1_count == -1: self.logger.error(f"[{session_id}] Aging: Failed to get T1 count."); return False
        except Exception as e_count: self.logger.error(f"[{session_id}] Aging: Exception getting T1 count: {e_count}", exc_info=True); return False

        should_age = current_t1_count >= trigger_threshold
        self.logger.debug(f"[{session_id}] Aging Trigger Check: T1 Count={current_t1_count}, Threshold={trigger_threshold}, Triggered={should_age}")
        if not should_age: self.logger.debug(f"[{session_id}] Aging not triggered."); return False

        await self._emit_status(event_emitter, session_id, "Status: Performing memory aging...")
        self.logger.info(f"[{session_id}] Aging triggered. Fetching oldest {batch_size} T1 summaries.")
        t1_batch_data = []
        try: t1_batch_data = await self._get_oldest_t1_batch_db_func(self.sqlite_cursor, session_id, batch_size)
        except Exception as e_fetch: self.logger.error(f"[{session_id}] Aging: Exception fetching T1 batch: {e_fetch}", exc_info=True); return False
        if not t1_batch_data or len(t1_batch_data) < batch_size: self.logger.warning(f"[{session_id}] Aging: Fetched {len(t1_batch_data)}/{batch_size} summaries. Skipping."); return False

        combined_text = ""; original_t1_ids = []; min_start_index: Optional[int] = None; max_end_index: Optional[int] = None; separator = "\n---\n"
        for item in t1_batch_data:
            if isinstance(item, dict):
                text = item.get('summary_text', '').strip(); t1_id = item.get('id'); start_idx = item.get('turn_start_index'); end_idx = item.get('turn_end_index')
                if text: combined_text += (text + separator)
                if t1_id: original_t1_ids.append(t1_id)
                if isinstance(start_idx, int): min_start_index = min(min_start_index, start_idx) if min_start_index is not None else start_idx
                if isinstance(end_idx, int): max_end_index = max(max_end_index, end_idx) if max_end_index is not None else end_idx
            else: self.logger.warning(f"[{session_id}] Aging: Invalid item type in batch: {type(item)}")

        combined_text = combined_text.strip()
        if not combined_text: self.logger.error(f"[{session_id}] Aging: Combined text empty. Aborting."); return False
        if min_start_index is None or max_end_index is None: self.logger.error(f"[{session_id}] Aging: Could not determine span. Aborting."); return False

        self.logger.debug(f"[{session_id}] Aging: Combined len: {len(combined_text)}. Span: {min_start_index}-{max_end_index}. IDs: {original_t1_ids}")
        aging_prompt = self._format_aging_prompt_func(combined_text, aging_prompt_template)
        if not aging_prompt or aging_prompt.startswith("[Error"): self.logger.error(f"[{session_id}] Aging: Failed format prompt: {aging_prompt}. Aborting."); return False

        aging_payload = {"contents": [{"parts": [{"text": aging_prompt}]}]}
        self.logger.info(f"[{session_id}] Aging: Calling LLM to condense T1 batch...")
        success, response_or_error = await self._async_llm_call_wrapper( api_url=aging_llm_url, api_key=aging_llm_key, payload=aging_payload, temperature=aging_temp, timeout=120, caller_info=f"Orch_MemAging_{session_id}" )

        if success and isinstance(response_or_error, str) and response_or_error.strip():
            aged_summary_text = response_or_error.strip()
            self.logger.info(f"[{session_id}] Aging: Condensation successful (Len: {len(aged_summary_text)}).")
            aged_summary_id = f"aged_{uuid.uuid4()}"; original_t1_count = len(original_t1_ids)
            add_success = await self._add_aged_summary_db_func( cursor=self.sqlite_cursor, aged_summary_id=aged_summary_id, session_id=session_id, aged_summary_text=aged_summary_text, original_batch_start_index=min_start_index, original_batch_end_index=max_end_index, original_t1_count=original_t1_count, original_t1_ids=original_t1_ids )
            if add_success:
                self.logger.info(f"[{session_id}] Aging: Added {aged_summary_id}. Deleting original {original_t1_count} T1 summaries...")
                delete_success = await self._delete_t1_batch_db_func(self.sqlite_cursor, session_id, original_t1_ids)
                if delete_success: self.logger.info(f"[{session_id}] Aging: Deleted original T1 batch."); await self._emit_status(event_emitter, session_id, "Status: Memory aging complete.", done=False); return True
                else: self.logger.critical(f"[{session_id}] Aging CRITICAL: Added {aged_summary_id} BUT failed delete original T1 batch!"); return False
            else: self.logger.error(f"[{session_id}] Aging: Failed save aged summary {aged_summary_id}. Aborting."); return False
        else:
            error_details = str(response_or_error);
            if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
            self.logger.error(f"[{session_id}] Aging: LLM condensation failed. Error: '{error_details}'."); return False

    # === _handle_tier2_transition (Unchanged) ===
    async def _handle_tier2_transition(
        self, session_id: str, t1_success: bool, chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> None:
        """ Handles checking and transitioning oldest T1 summary to T2 if limits exceeded. """
        await self._emit_status(event_emitter, session_id, "Status: Checking long-term memory capacity...")
        tier2_collection = None
        max_t1_blocks = getattr(self.config, 'max_stored_summary_blocks', 0)
        can_transition = all([ t1_success, self.chroma_client is not None, chroma_embed_wrapper is not None, self.sqlite_cursor is not None, get_tier1_summary_count is not None, get_oldest_tier1_summary is not None, add_to_chroma_collection is not None, delete_tier1_summary is not None, max_t1_blocks > 0 ])
        if not can_transition: self.logger.debug(f"[{session_id}] Skipping T1->T2 check: {'(T1 no run)' if not t1_success else '(Prereq fail)'}."); return
        try:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_'); safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]; tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
            if get_or_create_chroma_collection is None: raise ImportError("get_or_create_chroma_collection NA")
            tier2_collection = await get_or_create_chroma_collection(self.chroma_client, tier2_collection_name, chroma_embed_wrapper)
            if not tier2_collection: self.logger.error(f"[{session_id}] Failed get/create T2 collection '{tier2_collection_name}'. Skip."); return
        except Exception as e_get_coll: self.logger.error(f"[{session_id}] Error getting T2 collection: {e_get_coll}. Skip.", exc_info=True); return
        try:
            current_tier1_count = await get_tier1_summary_count(self.sqlite_cursor, session_id)
            if current_tier1_count == -1: self.logger.error(f"[{session_id}] Failed get T1 count. Skip T2 check."); return
            elif current_tier1_count > max_t1_blocks:
                self.logger.debug(f"[{session_id}] T1 limit ({max_t1_blocks}) exceeded ({current_tier1_count}). Transitioning...")
                await self._emit_status(event_emitter, session_id, "Status: Archiving oldest summary...")
                oldest_summary_data = await get_oldest_tier1_summary(self.sqlite_cursor, session_id)
                if not oldest_summary_data: self.logger.warning(f"[{session_id}] T1 count exceeded, but couldn't retrieve oldest."); return
                oldest_id, oldest_text, oldest_metadata = oldest_summary_data
                embedding_vector = None; embedding_successful = False
                try:
                    embedding_list = await asyncio.to_thread(chroma_embed_wrapper, [oldest_text])
                    if isinstance(embedding_list, list) and len(embedding_list) == 1 and isinstance(embedding_list[0], list) and len(embedding_list[0]) > 0: embedding_vector = embedding_list[0]; embedding_successful = True
                    else: self.logger.error(f"[{session_id}] T1->T2 Embed: Invalid structure: {type(embedding_list)}")
                except Exception as embed_e: self.logger.error(f"[{session_id}] EXCEPTION embedding T1->T2 {oldest_id}: {embed_e}", exc_info=True)
                if embedding_successful and embedding_vector:
                    added_to_t2 = False; deleted_from_t1 = False
                    chroma_metadata = oldest_metadata.copy(); chroma_metadata["transitioned_from_t1"] = True; chroma_metadata["original_t1_id"] = oldest_id
                    sanitized_chroma_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in chroma_metadata.items() if v is not None}
                    tier2_id = f"t2_{oldest_id}"
                    self.logger.debug(f"[{session_id}] Adding {tier2_id} to T2 '{tier2_collection.name}'...")
                    added_to_t2 = await add_to_chroma_collection(tier2_collection, ids=[tier2_id], embeddings=[embedding_vector], metadatas=[sanitized_chroma_metadata], documents=[oldest_text])
                    if added_to_t2:
                         self.logger.debug(f"[{session_id}] Added {tier2_id} to T2. Deleting T1 {oldest_id}...")
                         deleted_from_t1 = await delete_tier1_summary(self.sqlite_cursor, oldest_id)
                         if deleted_from_t1: self.logger.info(f"[{session_id}] Archived T1 {oldest_id} to T2."); await self._emit_status(event_emitter, session_id, "Status: Summary archive complete.", done=False)
                         else: self.logger.critical(f"[{session_id}] Added {tier2_id} to T2, FAILED DELETE T1 {oldest_id}!")
                    else: self.logger.error(f"[{session_id}] Failed add {tier2_id} to T2 collection.")
                else: self.logger.error(f"[{session_id}] Skip T2 add for T1 {oldest_id}: Embedding failed.")
            else: self.logger.debug(f"[{session_id}] T1 count ({current_tier1_count}) <= limit ({max_t1_blocks}). No T2 transition.")
        except Exception as e_t2_trans: self.logger.error(f"[{session_id}] Unexpected error during T1->T2 transition: {e_t2_trans}", exc_info=True)

    # === _calculate_and_format_status (Unchanged) ===
    async def _calculate_and_format_status(
        self, session_id: str, summarization_prompt_tokens: int, summarization_output_tokens: int,
        inventory_prompt_tokens: int, final_llm_payload_contents: Optional[List[Dict]],
        pre_scene_changed_flag: bool, final_confirmed_world_state: Dict[str, Any],
        final_confirmed_scene_state: Dict[str, Any], final_scene_changed_flag: bool,
        context_status_info: Dict[str, Any], session_process_owi_rag: bool,
        aging_performed_flag: bool, inventory_update_success_flag: bool,
        inventory_update_completed: bool,
    ) -> Tuple[str, int]:
        """ Calculates final status string including Aging status. """
        final_payload_tokens = -1
        if final_llm_payload_contents and self._count_tokens_func and self._tokenizer:
            try: final_payload_tokens = sum( self._count_tokens_func(part["text"], self._tokenizer) for turn in final_llm_payload_contents if isinstance(turn, dict) for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str) )
            except Exception as e_tok_final: final_payload_tokens = -1; self.logger.error(f"[{session_id}] Error calc final payload tokens: {e_tok_final}")
        elif not final_llm_payload_contents: final_payload_tokens = 0

        t1_retrieved_count = context_status_info.get("t1_retrieved_count", 0); aged_retrieved_count = context_status_info.get("aged_retrieved_count", 0); t2_retrieved_count = context_status_info.get("t2_retrieved_count", 0)
        initial_owi_context_tokens = context_status_info.get("initial_owi_context_tokens", -1); t0_dialogue_tokens = context_status_info.get("t0_dialogue_tokens", -1); cache_maintenance_performed = context_status_info.get("cache_maintenance_performed", False)

        final_day = final_confirmed_world_state.get("day", "?"); final_time = final_confirmed_world_state.get("time_of_day", "?"); final_weather = final_confirmed_world_state.get("weather", "?"); final_season = final_confirmed_world_state.get("season", "?")
        world_state_status_str = f"World: D{final_day} {final_time} {final_weather} {final_season}"
        t1_aged_status_str = f"T1={t1_retrieved_count}/{aged_retrieved_count}"; t2_status_str = f"T2={t2_retrieved_count}"
        pre_status_str = "NEW" if pre_scene_changed_flag else "OK"; post_status_str = "NEW" if final_scene_changed_flag else "OK"
        scene_status_str = f"Scene={pre_status_str}/{post_status_str}"

        token_parts = []
        if initial_owi_context_tokens >= 0: token_parts.append(f"OWI={initial_owi_context_tokens}")
        if t0_dialogue_tokens >= 0: token_parts.append(f"Hist={t0_dialogue_tokens}")
        if inventory_prompt_tokens >= 0: token_parts.append(f"Inv={inventory_prompt_tokens}")
        if final_payload_tokens >= 0: token_parts.append(f"Final={final_payload_tokens}")
        token_string = f"Tok: {' '.join(token_parts)}" if token_parts else ""

        inventory_enabled = getattr(self.config, 'enable_inventory_management', False)
        inv_stat_indicator = "Inv=OFF";
        if inventory_enabled: inv_stat_indicator = "Inv=MISSING" if not _ORCH_INVENTORY_MODULE_AVAILABLE else ("Inv=OK" if inventory_update_success_flag else ("Inv=FAIL" if inventory_update_completed else "Inv=SKIP"))
        aging_status_indicator = "Age=Y" if aging_performed_flag else "Age=N"; cache_status_indicator = "Cache=Y" if cache_maintenance_performed else "Cache=N"
        feature_status_string = f"{inv_stat_indicator} | {aging_status_indicator} | {cache_status_indicator}"

        final_status_string = f"{world_state_status_str} | {t1_aged_status_str}, {t2_status_str}, {scene_status_str} | {token_string} | {feature_status_string}"
        return final_status_string, final_payload_tokens

    # === _execute_or_prepare_output (Unchanged) ===
    async def _execute_or_prepare_output(
        self, session_id: str, body: Dict, final_llm_payload_contents: Optional[List[Dict]],
        event_emitter: Optional[Callable], status_message: str, final_payload_tokens: int
    ) -> OrchestratorResult:
        """ Executes final LLM call if configured, otherwise returns constructed payload. Uses utils.log_debug_payload. """
        if not final_llm_payload_contents:
            self.logger.error(f"[{session_id}] Final payload construction failed (input was None).")
            await self._emit_status(event_emitter, session_id, "ERROR: Final payload preparation failed.", done=True)
            return {"error": "Orchestrator: Final payload construction failed.", "status_code": 500}

        payload_for_log_and_call = {"contents": final_llm_payload_contents}
        preserved_keys = ["model", "stream", "options", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"]
        keys_preserved = [k for k in preserved_keys if k in body];
        for k in keys_preserved: payload_for_log_and_call[k] = body[k]
        self.logger.debug(f"[{session_id}] Output body constructed/updated. Preserved keys: {keys_preserved}.")

        if getattr(self.config, 'debug_log_final_payload', False) and _UTILS_AVAILABLE:
            self.logger.debug(f"[{session_id}] Logging final payload using utils.log_debug_payload.")
            try: log_debug_payload(session_id=session_id, payload_body=payload_for_log_and_call, config=self.config, logger_instance=self.logger)
            except Exception as e_log: self.logger.error(f"[{session_id}] Error calling log_debug_payload: {e_log}", exc_info=True)
        elif not _UTILS_AVAILABLE: self.logger.warning(f"[{session_id}] Skipping final payload log: Utils unavailable.")
        else: self.logger.debug(f"[{session_id}] Skipping final payload log: Debug valve OFF.")

        final_url = getattr(self.config, 'final_llm_api_url', None); final_key = getattr(self.config, 'final_llm_api_key', None)
        url_present = bool(final_url and isinstance(final_url, str) and final_url.strip()); key_present = bool(final_key and isinstance(final_key, str) and final_key.strip())
        self.logger.debug(f"[{session_id}] Checking Final LLM Trigger. URL:{url_present}, Key:{key_present}")
        final_llm_triggered = url_present and key_present

        if final_llm_triggered:
            self.logger.info(f"[{session_id}] Final LLM Call via Pipe TRIGGERED."); await self._emit_status(event_emitter, session_id, "Status: Executing final LLM Call...", done=False)
            final_temp = getattr(self.config, 'final_llm_temperature', 0.7); final_timeout = getattr(self.config, 'final_llm_timeout', 120)
            success, response_or_error = await self._async_llm_call_wrapper( api_url=final_url, api_key=final_key, payload=payload_for_log_and_call, temperature=final_temp, timeout=final_timeout, caller_info=f"Orch_FinalLLM_{session_id}" )
            intermediate_status = "Status: Final LLM Complete" + (" (Success)" if success else " (Failed)"); await self._emit_status(event_emitter, session_id, intermediate_status, done=False)
            if success and isinstance(response_or_error, str): self.logger.info(f"[{session_id}] Final LLM successful. Returning response string."); return response_or_error
            elif not success and isinstance(response_or_error, dict): self.logger.error(f"[{session_id}] Final LLM failed. Returning error dict: {response_or_error}"); return response_or_error
            else: self.logger.error(f"[{session_id}] Final LLM unexpected format. Success={success}, Type={type(response_or_error)}"); return {"error": "Final LLM adapter unexpected result format.", "status_code": 500}
        else: self.logger.info(f"[{session_id}] Final LLM Call disabled. Returning constructed payload dict."); return {"messages": final_llm_payload_contents}


    # === MAIN PROCESSING METHOD (Regen Skip Logic Added v0.2.3) ===
    async def process_turn(
        self,
        session_id: str,
        user_id: str,
        body: Dict,
        user_valves: Any, # Expects an object with user valve attributes
        event_emitter: Optional[Callable],
        embedding_func: Optional[Callable[[Sequence[str], str, Optional[Dict]], List[List[float]]]] = None,
        chroma_embed_wrapper: Optional[Any] = None,
        is_regeneration_heuristic: bool = False # <<< FLAG RECEIVED HERE
    ) -> OrchestratorResult:
        """
        Processes a single turn coordinating memory, state, context, inventory, hints,
        and final LLM calls. Skips certain steps if is_regeneration_heuristic is True.
        """
        pipe_entry_time_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Orchestrator process_turn v{self.version} [{session_id}]: Started at {pipe_entry_time_iso} (Regen Flag: {is_regeneration_heuristic})")

        # --- Define Getter/Logger Lambdas/Helpers using Utils ---
        debug_path_getter = None
        inventory_log_func = None # Default to None

        if _UTILS_AVAILABLE:
            debug_path_getter = functools.partial(
                get_debug_log_path,
                config=self.config,
                logger_instance=self.logger
            )
            async def inventory_logger_wrapper(sid: str, msg: str, log_type: str):
                if getattr(self.config, 'debug_log_final_payload', False):
                    await awaitable_log_inventory_debug(
                        session_id=sid,
                        message=msg,
                        log_type=log_type,
                        config=self.config,
                        logger_instance=self.logger
                    )
            inventory_log_func = inventory_logger_wrapper
        else:
             self.logger.warning(f"[{session_id}] Utils not available, debug logging features disabled.")
        # --- End Defining Lambdas/Helpers ---


        # --- Log feature status ---
        inventory_enabled = getattr(self.config, 'enable_inventory_management', False)
        event_hints_enabled = getattr(self.config, 'enable_event_hints', False)
        memory_aging_enabled = self.aging_trigger_threshold > 0 and self.aging_batch_size > 0
        self.logger.info(f"[{session_id}] Inventory Mgmt Enabled: {inventory_enabled} (Module Avail: {_ORCH_INVENTORY_MODULE_AVAILABLE})")
        self.logger.info(f"[{session_id}] Event Hints Enabled: {event_hints_enabled} (Module Avail: {_EVENT_HINTS_AVAILABLE})")
        self.logger.info(f"[{session_id}] Memory Aging Enabled: {memory_aging_enabled} (Trigger={self.aging_trigger_threshold}, Batch={self.aging_batch_size})")
        self.logger.info(f"[{session_id}] State Update Method: Two-Stage Unified Assessment (Available: {_UNIFIED_STATE_ASSESSMENT_AVAILABLE})")
        self.logger.info(f"[{session_id}] Context Processor: Available={_CONTEXT_PROCESSOR_AVAILABLE}")

        session_period_setting = getattr(user_valves, 'period_setting', '').strip()
        if session_period_setting: self.logger.info(f"[{session_id}] Using Period Setting from User Valves: '{session_period_setting}'")
        else: self.logger.debug(f"[{session_id}] No Period Setting provided in User Valves.")

        # --- Initialize state variables ---
        summarization_performed = False; new_t1_summary_text = None; summarization_prompt_tokens = -1; summarization_output_tokens = -1; t1_retrieved_count = 0;
        final_result: Optional[OrchestratorResult] = None; final_llm_payload_contents: Optional[List[Dict]] = None;
        inventory_update_completed = False; inventory_update_success_flag = False; inventory_prompt_tokens = -1;
        generated_event_hint_text: Optional[str] = None; generated_weather_proposal: Dict[str, Optional[str]] = {}
        aging_performed = False
        initial_world_state_dict: Dict = {}; initial_scene_state_dict: Dict = {"keywords": [], "description": ""}
        pre_assessed_state_dict: Optional[Dict] = None; pre_assessed_world_state_for_context: Dict = {}; pre_assessed_scene_state_for_context: Dict = {"keywords": [], "description": ""}
        final_confirmed_state_dict: Optional[Dict] = None; final_confirmed_world_state: Dict = {}; final_confirmed_scene_state: Dict = {"keywords": [], "description": ""}
        final_scene_changed_flag = False
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
            if latest_user_query_str is None: raise ValueError("Failed to determine effective query or history.")
            # Allow empty query *only* if regenerating
            if not latest_user_query_str and not is_regeneration_heuristic: raise ValueError("Cannot proceed without an effective user query (and not regeneration).")
            safe_previous_llm_response_str = previous_llm_response_str if previous_llm_response_str is not None else ""

            # --- Fetch Initial World & Scene State ---
            await self._emit_status(event_emitter, session_id, "Status: Fetching initial world state...")
            default_season = "Summer"; default_weather = "Clear"; default_day = 1; default_time = "Morning";
            db_world_state = await self._get_world_state_db_func(self.sqlite_cursor, session_id) if self.sqlite_cursor and self._get_world_state_db_func else None
            initial_world_state_dict = { "day": db_world_state.get("day", default_day) if db_world_state else default_day, "time_of_day": db_world_state.get("time_of_day", default_time) if db_world_state else default_time, "weather": db_world_state.get("weather", default_weather) if db_world_state else default_weather, "season": db_world_state.get("season", default_season) if db_world_state else default_season, }
            self.logger.debug(f"[{session_id}] Fetched initial world state: {initial_world_state_dict}")
            await self._emit_status(event_emitter, session_id, "Status: Fetching initial scene state...")
            db_scene_state = await self._get_scene_state_db_func(self.sqlite_cursor, session_id) if self.sqlite_cursor and self._get_scene_state_db_func else None
            kw_json = db_scene_state.get("keywords_json") if db_scene_state else None; desc = db_scene_state.get("description", "") if db_scene_state else ""
            try: keywords = json.loads(kw_json) if isinstance(kw_json, str) else []
            except json.JSONDecodeError: keywords = []
            initial_scene_state_dict = {"keywords": keywords if isinstance(keywords, list) else [], "description": desc if isinstance(desc, str) else ""}
            self.logger.debug(f"[{session_id}] Fetched initial scene state. Desc len: {len(initial_scene_state_dict['description'])}")

            # --- Memory Management Sequence (T1 already skips on regen, Aging/T2 check naturally follow) ---
            (summarization_performed, new_t1_summary_text, summarization_prompt_tokens, summarization_output_tokens) = await self._handle_tier1_summarization( session_id, user_id, current_active_history, is_regeneration_heuristic, event_emitter )
            if not is_regeneration_heuristic: # Only run aging/T2 transition if NOT regenerating
                 if summarization_performed: aging_performed = await self._handle_memory_aging(session_id, user_id, event_emitter)
                 else: aging_performed = await self._handle_memory_aging(session_id, user_id, event_emitter) # Check aging even if T1 didn't run THIS turn
                 await self._handle_tier2_transition( session_id, summarization_performed, chroma_embed_wrapper, event_emitter )
            else:
                 self.logger.info(f"[{session_id}] Skipping Aging and T2 Transition checks due to Regeneration flag.")
                 aging_performed = False # Ensure flag is false if skipped

            # --- Generate Hint AND Weather Proposal (Skip if regenerating) ---
            generated_weather_proposal = {"previous_weather": initial_world_state_dict.get("weather"), "new_weather": None} # Default
            if not is_regeneration_heuristic:
                hint_background_context = initial_scene_state_dict.get("description", "") + f"\n(Day: {initial_world_state_dict.get('day')}, Time: {initial_world_state_dict.get('time_of_day')}, Weather: {initial_world_state_dict.get('weather')})"
                if event_hints_enabled and self._generate_hint_func:
                    self.logger.debug(f"[{session_id}] Attempting event hint generation (Period: '{session_period_setting}')...")
                    await self._emit_status(event_emitter, session_id, "Status: Generating hint...")
                    hint_llm_url = getattr(self.config, 'event_hint_llm_api_url', None); hint_llm_key = getattr(self.config, 'event_hint_llm_api_key', None)
                    if not hint_llm_url or not hint_llm_key: self.logger.warning(f"[{session_id}] Skipping hint: Config incomplete.")
                    else:
                        try:
                            generated_event_hint_text, temp_weather_proposal = await self._generate_hint_func( config=self.config, history_messages=current_active_history, background_context=hint_background_context, current_season=initial_world_state_dict.get('season'), current_weather=initial_world_state_dict.get('weather'), current_time_of_day=initial_world_state_dict.get('time_of_day'), llm_call_func=self._async_llm_call_wrapper, logger_instance=self.logger, session_id=session_id, period_setting=session_period_setting )
                            generated_weather_proposal = temp_weather_proposal # Update proposal
                            if generated_event_hint_text: self.logger.info(f"[{session_id}] Hint Generated: '{generated_event_hint_text[:80]}...'")
                            else: self.logger.info(f"[{session_id}] No hint suggested.")
                            if generated_weather_proposal and generated_weather_proposal.get("new_weather"): self.logger.info(f"[{session_id}] Weather Proposal: '{generated_weather_proposal.get('previous_weather')}' -> '{generated_weather_proposal.get('new_weather')}'")
                            else: self.logger.debug(f"[{session_id}] No valid weather proposal.")
                            await self._emit_status(event_emitter, session_id, "Status: Hint generation complete.")
                        except Exception as e_hint_gen: self.logger.error(f"[{session_id}] Error during hint generation: {e_hint_gen}", exc_info=True); generated_event_hint_text = None; generated_weather_proposal = {"previous_weather": initial_world_state_dict.get("weather"), "new_weather": None} # Reset proposal on error
                elif event_hints_enabled and not self._generate_hint_func: self.logger.error(f"[{session_id}] Skipping hint: Function unavailable.")
                else: self.logger.debug(f"[{session_id}] Skipping hint: Disabled by global valve."); # Keep default proposal
            else:
                 self.logger.info(f"[{session_id}] Skipping Hint Generation due to Regeneration flag.")
                 generated_event_hint_text = None # Ensure hint is None if skipped
                 # Keep default weather proposal if skipped

            # --- STAGE 1: Pre-emptive State Assessment (Skip if regenerating) ---
            pre_assessed_state_dict = None
            pre_assessed_world_state_for_context = initial_world_state_dict.copy() # Default to initial
            pre_assessed_scene_state_for_context = initial_scene_state_dict.copy() # Default to initial

            if not is_regeneration_heuristic:
                if self._unified_state_func:
                    self.logger.info(f"[{session_id}] Performing pre-emptive state assessment...")
                    await self._emit_status(event_emitter, session_id, "Status: Assessing pre-emptive state...", done=False)
                    try:
                        state_assess_llm_config = { "url": getattr(self.config, 'event_hint_llm_api_url', None), "key": getattr(self.config, 'event_hint_llm_api_key', None), "temp": getattr(self.config, 'state_assess_llm_temperature', 0.3), "prompt_template": DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT }
                        if not state_assess_llm_config["url"] or not state_assess_llm_config["key"]: self.logger.error(f"[{session_id}] Pre-State LLM URL/Key missing. Skip."); pre_assessed_state_dict = None
                        else:
                            pre_assessed_state_dict = await self._unified_state_func( session_id=session_id, previous_world_state=initial_world_state_dict, previous_scene_state=initial_scene_state_dict, current_user_query=latest_user_query_str, assistant_response_text=safe_previous_llm_response_str, history_messages=current_active_history, llm_call_func=self._async_llm_call_wrapper, state_assessment_llm_config=state_assess_llm_config, logger_instance=self.logger, event_emitter=event_emitter, weather_proposal=generated_weather_proposal )
                            if pre_assessed_state_dict and isinstance(pre_assessed_state_dict, dict):
                                 self.logger.info(f"[{session_id}] Pre-state assessment completed.")
                                 # Update context states ONLY if assessment succeeded
                                 pre_assessed_world_state_for_context = {k: pre_assessed_state_dict.get(f"new_{k}", initial_world_state_dict[k]) for k in ["day", "time_of_day", "weather", "season"]}
                                 pre_assessed_scene_state_for_context = { "keywords": pre_assessed_state_dict.get("new_scene_keywords", initial_scene_state_dict["keywords"]), "description": pre_assessed_state_dict.get("new_scene_description", initial_scene_state_dict["description"]) }
                                 self.logger.debug(f"[{session_id}] Pre-assessed World (for context): {pre_assessed_world_state_for_context}")
                                 self.logger.debug(f"[{session_id}] Pre-assessed Scene Desc len (for context): {len(pre_assessed_scene_state_for_context['description'])}")
                            else: self.logger.error(f"[{session_id}] Pre-state assessment invalid data. Using initial for context."); pre_assessed_state_dict = None; # Keep defaults set above
                    except Exception as e_pre_assess: self.logger.error(f"[{session_id}] Exception during pre-state assessment: {e_pre_assess}", exc_info=True); pre_assessed_state_dict = None; # Keep defaults set above
                else: self.logger.error(f"[{session_id}] Skipping pre-state assessment: Function unavailable."); # Keep defaults set above
            else:
                 self.logger.info(f"[{session_id}] Skipping Pre-State Assessment due to Regeneration flag.")
                 # Ensure context states remain the initial ones if skipped
                 pre_assessed_world_state_for_context = initial_world_state_dict.copy()
                 pre_assessed_scene_state_for_context = initial_scene_state_dict.copy()

            # --- Call Context Processor (PASSING REGEN FLAG) ---
            if not self._context_processor_func: raise RuntimeError("Context processor unavailable.")
            await self._emit_status(event_emitter, session_id, "Status: Processing background context...")
            final_llm_payload_contents, context_status_info = await self._context_processor_func(
                session_id=session_id, body=body, user_valves=user_valves,
                current_active_history=current_active_history, history_for_processing=history_for_processing,
                latest_user_query_str=latest_user_query_str,
                current_scene_state_dict=pre_assessed_scene_state_for_context, # Use the determined state (initial or pre-assessed)
                current_world_state_dict=pre_assessed_world_state_for_context, # Use the determined state (initial or pre-assessed)
                generated_event_hint_text=generated_event_hint_text, generated_weather_proposal=generated_weather_proposal,
                config=self.config, logger=self.logger, sqlite_cursor=self.sqlite_cursor,
                chroma_client=self.chroma_client, chroma_embed_wrapper=chroma_embed_wrapper,
                embedding_func=embedding_func, llm_call_func=self._async_llm_call_wrapper,
                tokenizer=self._tokenizer, event_emitter=event_emitter,
                orchestrator_debug_path_getter=debug_path_getter,
                dialogue_roles=self._dialogue_roles, session_period_setting=session_period_setting,
                db_get_recent_t1_summaries_func=get_recent_tier1_summaries,
                db_get_recent_aged_summaries_func=get_recent_aged_summaries,
                is_regeneration_heuristic=is_regeneration_heuristic # <<< PASS FLAG HERE
            )
            await self._emit_status(event_emitter, session_id, "Status: Context processing complete.")
            if context_status_info.get("error"): self.logger.error(f"[{session_id}] Error from context processor: {context_status_info['error']}")
            if final_llm_payload_contents is None: raise ValueError("Context processor failed return payload.")
            t1_retrieved_count = context_status_info.get("t1_retrieved_count", 0)

            # --- Execute Final LLM Call or Prepare Output ---
            final_result = await self._execute_or_prepare_output( session_id=session_id, body=body, final_llm_payload_contents=final_llm_payload_contents, event_emitter=event_emitter, status_message="Status: Core processing complete.", final_payload_tokens=-1 )

            # --- STAGE 2: Post-Turn State Assessment (Finalization - Skip if regenerating) ---
            final_confirmed_state_dict = None
            narrative_response_text = final_result if isinstance(final_result, str) else None

            if not is_regeneration_heuristic:
                if narrative_response_text and self._unified_state_func:
                    self.logger.info(f"[{session_id}] Performing post-turn state finalization...")
                    await self._emit_status(event_emitter, session_id, "Status: Finalizing state assessment...", done=False)
                    try:
                        state_assess_llm_config = { "url": getattr(self.config, 'event_hint_llm_api_url', None), "key": getattr(self.config, 'event_hint_llm_api_key', None), "temp": getattr(self.config, 'state_assess_llm_temperature', 0.3), "prompt_template": DEFAULT_UNIFIED_STATE_ASSESSMENT_PROMPT_TEXT }
                        if not state_assess_llm_config["url"] or not state_assess_llm_config["key"]: self.logger.error(f"[{session_id}] Post-State LLM URL/Key missing. Skip."); final_confirmed_state_dict = None
                        else:
                            final_confirmed_state_dict = await self._unified_state_func( session_id=session_id, previous_world_state=initial_world_state_dict, previous_scene_state=initial_scene_state_dict, current_user_query=latest_user_query_str, assistant_response_text=narrative_response_text, history_messages=current_active_history, llm_call_func=self._async_llm_call_wrapper, state_assessment_llm_config=state_assess_llm_config, logger_instance=self.logger, event_emitter=event_emitter, weather_proposal=generated_weather_proposal )
                            if final_confirmed_state_dict and isinstance(final_confirmed_state_dict, dict): self.logger.info(f"[{session_id}] Post-state finalization completed.")
                            else: self.logger.error(f"[{session_id}] Post-state finalization invalid data."); final_confirmed_state_dict = None
                    except Exception as e_post_assess: self.logger.error(f"[{session_id}] Exception during post-state finalization: {e_post_assess}", exc_info=True); final_confirmed_state_dict = None
                elif not self._unified_state_func: self.logger.error(f"[{session_id}] Skipping post-state finalization: Function unavailable.")
                elif not narrative_response_text: self.logger.debug(f"[{session_id}] Skipping post-state finalization: Final result not string.")
            else:
                 self.logger.info(f"[{session_id}] Skipping Post-State Finalization due to Regeneration flag.")
                 final_confirmed_state_dict = None # Ensure it's None if skipped

            # --- Update Orchestrator State & Save FINAL Confirmed State ---
            # Logic remains the same, will use initial or pre-assessed state if final is None
            if final_confirmed_state_dict:
                self.logger.debug(f"[{session_id}] Using final confirmed state.")
                final_world_state = {k: final_confirmed_state_dict.get(f"new_{k}", initial_world_state_dict[k]) for k in ["day", "time_of_day", "weather", "season"]}
                final_scene_state = { "keywords": final_confirmed_state_dict.get("new_scene_keywords", initial_scene_state_dict["keywords"]), "description": final_confirmed_state_dict.get("new_scene_description", initial_scene_state_dict["description"]) }
                final_scene_changed_flag = final_confirmed_state_dict.get("scene_changed_flag", False)
            elif pre_assessed_state_dict:
                self.logger.warning(f"[{session_id}] Post-finalization failed/skipped. Using pre-assessed state.")
                final_world_state = pre_assessed_world_state_for_context; final_scene_state = pre_assessed_scene_state_for_context
                final_scene_changed_flag = pre_assessed_state_dict.get("scene_changed_flag", False)
            else:
                self.logger.warning(f"[{session_id}] Both state assessments failed/skipped. Using initial state.")
                final_world_state = initial_world_state_dict; final_scene_state = initial_scene_state_dict; final_scene_changed_flag = False

            # Save state to DB ONLY IF NOT REGENERATING (state shouldn't change on regen)
            if not is_regeneration_heuristic:
                world_state_changed_final = final_world_state != initial_world_state_dict; scene_state_changed_final = final_scene_changed_flag
                if world_state_changed_final and self.sqlite_cursor and self._set_world_state_db_func:
                    await self._emit_status(event_emitter, session_id, "Status: Saving final world state...", done=False)
                    try:
                        update_success = await self._set_world_state_db_func( self.sqlite_cursor, session_id, final_world_state["season"], final_world_state["weather"], final_world_state["day"], final_world_state["time_of_day"] )
                        if update_success: self.logger.info(f"[{session_id}] Final world state saved: {final_world_state}")
                        else: self.logger.error(f"[{session_id}] Failed save final world state.")
                    except Exception as e_set_world: self.logger.error(f"[{session_id}] Error saving final world state: {e_set_world}", exc_info=True)
                elif not world_state_changed_final: self.logger.debug(f"[{session_id}] No final world state change. Skip save.")
                if scene_state_changed_final and self.sqlite_cursor and self._set_scene_state_db_func:
                    await self._emit_status(event_emitter, session_id, "Status: Saving final scene state...", done=False)
                    try:
                        kw_json_to_save = json.dumps(final_scene_state["keywords"])
                        update_success = await self._set_scene_state_db_func( self.sqlite_cursor, session_id, kw_json_to_save, final_scene_state["description"] )
                        if update_success: self.logger.info(f"[{session_id}] Final scene state saved. Desc len: {len(final_scene_state['description'])}")
                        else: self.logger.error(f"[{session_id}] Failed save final scene state.")
                    except Exception as e_set_scene: self.logger.error(f"[{session_id}] Error saving final scene state: {e_set_scene}", exc_info=True)
                elif not scene_state_changed_final: self.logger.debug(f"[{session_id}] No final scene state change. Skip save.")
            else:
                self.logger.info(f"[{session_id}] Skipping World/Scene state saving due to Regeneration flag.")
                # Ensure the state used for status reflects the initial state if regen
                final_world_state = initial_world_state_dict
                final_scene_state = initial_scene_state_dict
                final_scene_changed_flag = False


            # --- Post-Turn Inventory Update (Skip if regenerating) ---
            if not is_regeneration_heuristic:
                if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and self._update_inventories_func:
                    inventory_update_completed = True
                    if narrative_response_text:
                        self.logger.debug(f"[{session_id}] Performing post-turn inventory update...")
                        await self._emit_status(event_emitter, session_id, "Status: Updating inventory state...", done=False)
                        inv_llm_url = getattr(self.config, 'inv_llm_api_url', None); inv_llm_key = getattr(self.config, 'inv_llm_api_key', None);
                        inv_llm_config_from_main = getattr(self.config,'inventory_llm_config', {})
                        inv_llm_prompt_template = inv_llm_config_from_main.get("prompt_template", DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT) if isinstance(inv_llm_config_from_main, dict) else DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT
                        inv_llm_config_for_call = { "url": inv_llm_url, "key": inv_llm_key, "temp": getattr(self.config, 'inv_llm_temperature', 0.3), "prompt_template": inv_llm_prompt_template }
                        template_seems_valid = inv_llm_prompt_template != "[Default Inventory Prompt Load Failed]" and inv_llm_prompt_template is not None

                        if not inv_llm_url or not inv_llm_key or not template_seems_valid: self.logger.error(f"[{session_id}] Inventory LLM config missing/invalid. URL:{bool(inv_llm_url)}, Key:{bool(inv_llm_key)}, Template Valid:{template_seems_valid}."); inventory_update_success_flag = False
                        else:
                            history_for_inv_update_list = self._get_recent_turns_func(current_active_history, 4, exclude_last=False); history_for_inv_update_str = self._format_history_func(history_for_inv_update_list)
                            inv_prompt_text = format_inventory_update_prompt( main_llm_response=narrative_response_text, user_query=latest_user_query_str, recent_history_str=history_for_inv_update_str, template=inv_llm_config_for_call['prompt_template'])
                            if inv_prompt_text and not inv_prompt_text.startswith("[Error") and self._count_tokens_func and self._tokenizer:
                                try: inventory_prompt_tokens = self._count_tokens_func(inv_prompt_text, self._tokenizer); self.logger.debug(f"[{session_id}] Inv Prompt Tokens: {inventory_prompt_tokens}")
                                except Exception as e_inv_tok: inventory_prompt_tokens = -1
                            else: inventory_prompt_tokens = -1
                            if not self.sqlite_cursor or not self.sqlite_cursor.connection: self.logger.error(f"[{session_id}] Cannot update inventory: SQLite cursor invalid."); inventory_update_success_flag = False
                            else:
                                try:
                                    update_success = await self._update_inventories_func(
                                        cursor=self.sqlite_cursor, session_id=session_id,
                                        main_llm_response=narrative_response_text, user_query=latest_user_query_str,
                                        recent_history_str=history_for_inv_update_str, llm_call_func=self._async_llm_call_wrapper,
                                        db_get_inventory_func=get_character_inventory_data, db_update_inventory_func=add_or_update_character_inventory,
                                        inventory_llm_config=inv_llm_config_for_call,
                                        inventory_log_func=inventory_log_func # Pass conditional wrapper/None
                                    )
                                    inventory_update_success_flag = update_success
                                    if update_success: self.logger.info(f"[{session_id}] Post-turn inventory update successful.")
                                    else: self.logger.warning(f"[{session_id}] Post-turn inventory update function returned False.")
                                except Exception as e_inv_update_inner: self.logger.error(f"[{session_id}] Error during inventory update call: {e_inv_update_inner}", exc_info=True); inventory_update_success_flag = False
                    elif isinstance(final_result, dict) and "error" in final_result: self.logger.warning(f"[{session_id}] Skipping inventory update due to upstream error."); inventory_update_completed = False
                    elif isinstance(final_result, dict) and "messages" in final_result: self.logger.debug(f"[{session_id}] Skipping inventory update: Final LLM call disabled."); inventory_update_completed = False
                    else: self.logger.error(f"[{session_id}] Unexpected type for final_result. Skipping inventory update."); inventory_update_completed = False
                elif inventory_enabled and not _ORCH_INVENTORY_MODULE_AVAILABLE: self.logger.warning(f"[{session_id}] Skipping inventory update: Module import failed."); inventory_update_completed = False
                elif inventory_enabled and not self._update_inventories_func: self.logger.error(f"[{session_id}] Skipping inventory update: Update function alias None."); inventory_update_completed = False
                else: self.logger.debug(f"[{session_id}] Skipping inventory update: Disabled by global valve."); inventory_update_completed = False
            else:
                 self.logger.info(f"[{session_id}] Skipping Inventory Update due to Regeneration flag.")
                 inventory_update_completed = False # Ensure flag is false if skipped
                 inventory_update_success_flag = False # Ensure flag is false if skipped

            # --- Final Status Calculation and Emission ---
            pre_scene_changed_flag = False # Default for regen or if pre-assess failed
            if pre_assessed_state_dict and isinstance(pre_assessed_state_dict, dict):
                pre_scene_changed_flag = pre_assessed_state_dict.get("scene_changed_flag", False)

            final_status_string, final_payload_tokens = await self._calculate_and_format_status(
                 session_id=session_id, summarization_prompt_tokens=summarization_prompt_tokens,
                 summarization_output_tokens=summarization_output_tokens, inventory_prompt_tokens=inventory_prompt_tokens,
                 final_llm_payload_contents=final_llm_payload_contents, pre_scene_changed_flag=pre_scene_changed_flag,
                 final_confirmed_world_state=final_world_state, # Will be initial state if regen
                 final_confirmed_scene_state=final_scene_state, # Will be initial state if regen
                 final_scene_changed_flag=final_scene_changed_flag, # Will be false if regen
                 context_status_info=context_status_info,
                 session_process_owi_rag=bool(getattr(user_valves, 'process_owi_rag', True)),
                 aging_performed_flag=aging_performed, # Will be false if regen
                 inventory_update_success_flag=inventory_update_success_flag, # Will be false if regen
                 inventory_update_completed=inventory_update_completed # Will be false if regen
            )

            self.logger.info(f"[{session_id}] Orchestrator FINAL STATUS: {final_status_string}")
            await self._emit_status(event_emitter, session_id, final_status_string, done=True)

            pipe_end_time_iso = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"Orchestrator process_turn v{self.version} [{session_id}]: Finished at {pipe_end_time_iso}")
            if final_result is None: raise RuntimeError("Internal processing error, final result was None.")
            return final_result

        # --- Exception Handling ---
        except asyncio.CancelledError:
            self.logger.info(f"[{session_id or 'unknown'}] Orchestrator process_turn cancelled.")
            await self._emit_status(event_emitter, session_id or 'unknown', "Status: Processing cancelled.", done=True)
            raise
        except ValueError as ve:
            session_id_for_log = session_id if 'session_id' in locals() else 'unknown'
            self.logger.error(f"[{session_id_for_log}] Orchestrator ValueError in process_turn: {ve}", exc_info=True)
            try: await self._emit_status(event_emitter, session_id_for_log, f"ERROR: {ve}", done=True)
            except Exception: pass
            return {"error": f"Orchestrator failed: {ve}", "status_code": 500}
        except Exception as e_orch:
            session_id_for_log = session_id if 'session_id' in locals() else 'unknown'
            self.logger.critical(f"[{session_id_for_log}] Orchestrator UNHANDLED EXCEPTION in process_turn: {e_orch}", exc_info=True)
            try: await self._emit_status(event_emitter, session_id_for_log, f"ERROR: Orchestrator Failed ({type(e_orch).__name__})", done=True)
            except Exception: pass
            return {"error": f"Orchestrator failed: {type(e_orch).__name__}", "status_code": 500}

# [[END MODIFIED orchestration.py - Regen Skip Logic v0.2.3]]