# --- START OF FILE orchestration.py ---

# [[START MODIFIED orchestration.py - FULL V2 - Fix Circular Import]]
# i4_llm_agent/orchestration.py

import logging
import asyncio
import re
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import (
    Tuple, Union, List, Dict, Optional, Any, Callable, Coroutine, AsyncGenerator
)

# --- Standard Library Imports ---
# (Add others as needed)

# --- i4_llm_agent Imports (Careful with internal package imports) ---
# Import specific modules directly to avoid __init__.py dependency during load
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
)
from .history import (
    format_history_for_llm, get_recent_turns, DIALOGUE_ROLES, select_turns_for_t0
)
from .memory import manage_tier1_summarization
from .prompting import (
    construct_final_llm_payload, clean_context_tags, generate_rag_query,
    combine_background_context, process_system_prompt,
    refine_external_context, format_stateless_refiner_prompt,
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
    format_cache_update_prompt, format_final_context_selection_prompt,
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
    format_inventory_update_prompt, DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
)
from .cache import update_rag_cache, select_final_context
from .api_client import call_google_llm_api, _convert_google_to_openai_payload # Import converter too
from .utils import count_tokens, calculate_string_similarity, TIKTOKEN_AVAILABLE


# --- Inventory Module Import (LOCAL TO ORCHESTRATION) ---
# This block determines inventory availability *within this module*
# without relying on the top-level __init__.py during initial load.
try:
    from .inventory import (
        format_inventory_for_prompt as _real_format_inventory_func,
        update_inventories_from_llm as _real_update_inventories_func,
    )
    _ORCH_INVENTORY_MODULE_AVAILABLE = True
    _dummy_format_inventory = None # Not needed if import succeeds
    _dummy_update_inventories = None # Not needed if import succeeds
except ImportError:
    _ORCH_INVENTORY_MODULE_AVAILABLE = False
    _real_format_inventory_func = None # Not available
    _real_update_inventories_func = None # Not available
    # Define dummy functions *here* if module not available
    def _dummy_format_inventory(*args, **kwargs): return "[Inventory Module Unavailable]"
    async def _dummy_update_inventories(*args, **kwargs): await asyncio.sleep(0); return False
    logging.getLogger(__name__).warning(
        "Orchestration: Inventory module not found. Inventory features disabled within orchestrator."
        )
# --- END LOCAL Inventory Import ---


# --- Optional Imports ---
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False
    # Logging handled by Pipe startup

import urllib.parse


logger = logging.getLogger(__name__) # i4_llm_agent.orchestration

# Type Alias for the complex return type
OrchestratorResult = Union[Dict, AsyncGenerator[str, None], str]

# ==============================================================================
# === Session Pipe Orchestrator Class (Modularized)                          ===
# ==============================================================================

class SessionPipeOrchestrator:
    """
    Orchestrates the core processing logic of the Session Memory Pipe.
    Encapsulates logic in helper methods for clarity and maintainability.
    Includes inventory management features.
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
        self._llm_call_func = call_google_llm_api # Use dispatcher
        self._format_history_func = format_history_for_llm
        self._get_recent_turns_func = get_recent_turns
        self._manage_memory_func = manage_tier1_summarization
        self._construct_payload_func = construct_final_llm_payload
        self._count_tokens_func = count_tokens
        self._calculate_similarity_func = calculate_string_similarity
        self._dialogue_roles = DIALOGUE_ROLES
        # Prompting/Context
        self._clean_context_tags_func = clean_context_tags
        self._generate_rag_query_func = generate_rag_query
        self._combine_context_func = combine_background_context
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

        # Inventory Module Functions
        # Assign functions based on the *local* flag determined above
        if _ORCH_INVENTORY_MODULE_AVAILABLE:
            self._format_inventory_func = _real_format_inventory_func
            self._update_inventories_func = _real_update_inventories_func
        else:
            self._format_inventory_func = _dummy_format_inventory
            self._update_inventories_func = _dummy_update_inventories

        self.logger.info("SessionPipeOrchestrator initialized.")

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
                if asyncio.iscoroutinefunction(event_emitter):
                    await event_emitter(status_data)
                else:
                    event_emitter(status_data) # Call sync directly
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
        # (Code remains the same as previous version)
        if not self._llm_call_func:
            self.logger.error(f"[{caller_info}] LLM func unavailable in orchestrator.")
            return False, {"error_type": "SetupError", "message": "LLM func unavailable"}
        try:
            result = self._llm_call_func(
                api_url=api_url, api_key=api_key, payload=payload,
                temperature=temperature, timeout=timeout, caller_info=caller_info
            )
            if asyncio.iscoroutine(result):
                self.logger.debug(f"[{caller_info}] Awaiting result from LLM function.")
                return await result
            elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
                self.logger.debug(f"[{caller_info}] LLM function returned tuple directly (Success: {result[0]}).")
                return result
            else:
                self.logger.error(f"[{caller_info}] LLM function returned unexpected type: {type(result)}. Result: {result}")
                return False, {"error_type": "InternalError", "message": f"LLM function returned unexpected type {type(result)}"}
        except Exception as e:
            self.logger.error(f"Orchestrator LLM Wrapper Error [{caller_info}]: {e}", exc_info=True)
            return False, {"error_type": "AsyncWrapperError", "message": f"{type(e).__name__}: {str(e)}"}


    # --- Internal Helper: Final LLM Stream Call Wrapper ---
    async def _async_final_llm_stream_call(
        self,
        api_url: str,
        api_key: str,
        payload: Dict[str, Any], # Expects Google 'contents' format initially
        temperature: float,
        timeout: int = 120,
        caller_info: str = "Orchestrator_FinalStream",
        final_status_message: str = "Status: Stream completed.",
        event_emitter: Optional[Callable] = None
    ) -> AsyncGenerator[str, None]:
        """ Handles streaming calls to OpenAI-compatible endpoints. """
        # (Code remains the same as previous version)
        log_session_id = caller_info
        session_id_match = re.search(r'_(user_.*|chat_.*)', log_session_id)
        extracted_session_id = session_id_match.group(1) if session_id_match else "unknown_session"
        base_api_url = api_url
        final_payload_to_send = {}
        headers = {}
        stream_successful = False

        if not HTTPX_AVAILABLE:
            self.logger.error(f"[{log_session_id}] httpx library not available. Cannot stream.")
            yield f"[Streaming Error: httpx not installed]"
            return
        if not api_url or not api_key:
            error_msg = "Missing API Key" if not api_key else "Missing Final LLM URL"
            self.logger.error(f"[{log_session_id}] {error_msg}.")
            yield f"[Streaming Error: {error_msg}]"
            return

        try:
            converter_func = _convert_google_to_openai_payload
            if converter_func is None:
                self.logger.error(f"[{log_session_id}] Payload converter func unavailable. Cannot stream OpenAI.")
                yield "[Streaming Error: Payload converter missing]"
                return

            parsed_url = urllib.parse.urlparse(api_url)
            base_api_url = urllib.parse.urlunparse(parsed_url._replace(fragment=""))
            url_for_check = base_api_url.lower().rstrip('/')

            if "openrouter.ai/api/v1/chat/completions" in url_for_check or url_for_check.endswith("/v1/chat/completions"):
                if parsed_url.fragment:
                    model_name_for_conversion = parsed_url.fragment
                    self.logger.debug(f"[{log_session_id}] Final stream target is OpenAI-like. Model: '{model_name_for_conversion}'")
                    try:
                        openai_payload_converted = converter_func(payload, model_name_for_conversion, temperature)
                        openai_payload_converted["stream"] = True
                        final_payload_to_send = openai_payload_converted
                    except Exception as e_conv:
                        self.logger.error(f"[{log_session_id}] Error converting payload for OpenAI stream: {e_conv}", exc_info=True)
                        yield f"[Streaming Error: Payload conversion failed - {type(e_conv).__name__}]"
                        return
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "Accept": "text/event-stream"}
                else:
                    self.logger.error(f"[{log_session_id}] OpenAI-like URL requires model fragment for streaming.")
                    yield "[Streaming Error: Model fragment missing in URL]"
                    return
            else:
                self.logger.error(f"[{log_session_id}] Cannot determine final LLM API type for streaming from URL: {base_api_url}")
                yield "[Streaming Error: Cannot determine API type for streaming]"
                return
        except Exception as e_prep:
            self.logger.error(f"[{log_session_id}] Error parsing URL/preparing payload for stream: {e_prep}", exc_info=True)
            yield "[Streaming Error: URL/Payload prep failed]"
            return

        if not final_payload_to_send:
             self.logger.error(f"[{log_session_id}] Final payload for streaming is empty after preparation.")
             yield "[Streaming Error: Final payload empty]"
             return

        self.logger.info(f"[{log_session_id}] Starting final LLM stream call to {base_api_url[:80]}...")
        self.logger.debug(f"[{log_session_id}] Streaming Payload: {json.dumps(final_payload_to_send)}")
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                 async with client.stream("POST", base_api_url, headers=headers, json=final_payload_to_send, timeout=timeout) as response:
                    if response.status_code != 200:
                         error_body = await response.aread(); error_text = error_body.decode('utf-8', errors='replace')[:1000]
                         self.logger.error(f"[{log_session_id}] Stream API Error: Status {response.status_code}. Response: {error_text}...")
                         yield f"[Streaming Error: API returned status {response.status_code}]"
                         try: err_json = json.loads(error_text)
                         except json.JSONDecodeError: pass
                         else:
                             if "error" in err_json and isinstance(err_json["error"], dict): yield f"\n[Detail: {err_json['error'].get('message', 'Unknown API error')}]"
                         return

                    self.logger.debug(f"[{log_session_id}] Stream connection successful (Status {response.status_code}). Reading chunks...")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_content = line[len("data:"):].strip()
                            if data_content == "[DONE]": self.logger.debug(f"[{log_session_id}] Received [DONE] signal."); stream_successful = True; break
                            if data_content:
                                try:
                                    chunk = json.loads(data_content)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {}); text_chunk = delta.get("content")
                                    if text_chunk: yield text_chunk
                                except json.JSONDecodeError: self.logger.warning(f"[{log_session_id}] Failed to decode JSON data chunk: '{data_content}'")
                                except (IndexError, KeyError, TypeError) as e_parse: self.logger.warning(f"[{log_session_id}] Error parsing stream chunk structure ({type(e_parse).__name__}): {chunk}")
                        elif line.strip(): self.logger.debug(f"[{log_session_id}] Received non-data line: '{line}'")
        except httpx.TimeoutException as e_timeout: self.logger.error(f"[{log_session_id}] HTTPX Timeout during stream after {timeout}s: {e_timeout}", exc_info=False); yield f"[Streaming Error: Timeout after {timeout}s]"
        except httpx.RequestError as e_req: self.logger.error(f"[{log_session_id}] HTTPX RequestError during stream: {e_req}", exc_info=True); yield f"[Streaming Error: Network request failed - {type(e_req).__name__}]"
        except asyncio.CancelledError: self.logger.info(f"[{log_session_id}] Final stream call explicitly cancelled."); yield "[Streaming Error: Cancelled]"; raise
        except Exception as e_stream: self.logger.error(f"[{log_session_id}] Unexpected error during final LLM stream: {e_stream}", exc_info=True); yield f"[Streaming Error: Unexpected error - {type(e_stream).__name__}]"
        finally:
             self.logger.info(f"[{log_session_id}] Final LLM stream processing finished.")
             status_to_emit = final_status_message + ("" if stream_successful else " (Stream Interrupted or Failed)")
             await self._emit_status(event_emitter=event_emitter, session_id=extracted_session_id, description=status_to_emit, done=True)


    # --- Helper Methods for process_turn ---

    async def _determine_effective_query(
        self, session_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool
    ) -> Tuple[str, List[Dict]]:
        """ Determines the effective user query and the history slice preceding it. """
        # (Code remains the same as previous version)
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
        # (Code remains the same as previous version)
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
        # (Code remains the same as previous version)
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
        # (Code remains the same as previous version)
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
        # (Code remains the same as previous version)
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
    ) -> Tuple[str, str, int, int, bool, bool, bool, bool, str]: # Return includes formatted inventory
        """ Processes system prompt, refines context, includes inventory. """
        # (Code remains the same as previous version)
        await self._emit_status(event_emitter, session_id, "Status: Preparing context...")
        base_system_prompt_text = "You are helpful."; extracted_owi_context = None; initial_owi_context_tokens = -1; current_output_messages = body.get("messages", [])
        if self._process_system_prompt_func:
             try: base_system_prompt_text, extracted_owi_context = self._process_system_prompt_func(current_output_messages)
             except Exception as e_proc_sys: self.logger.error(f"[{session_id}] Error process_system_prompt: {e_proc_sys}.", exc_info=True); extracted_owi_context = None
        else: self.logger.error(f"[{session_id}] process_system_prompt unavailable."); base_system_prompt_text = "You are helpful."
        if extracted_owi_context and self._count_tokens_func and self._tokenizer:
             try: initial_owi_context_tokens = self._count_tokens_func(extracted_owi_context, self._tokenizer)
             except Exception: initial_owi_context_tokens = -1
        elif not extracted_owi_context: self.logger.debug(f"[{session_id}] No OWI <context> tag found.")
        if not base_system_prompt_text: base_system_prompt_text = "You are helpful."; self.logger.warning(f"[{session_id}] System prompt empty after clean. Using default.")

        session_text_block_to_remove = getattr(user_valves, 'text_block_to_remove', '') if user_valves else ''
        if session_text_block_to_remove:
            self.logger.info(f"[{session_id}] Attempting removal of text block from base system prompt...")
            original_len = len(base_system_prompt_text); temp_prompt = base_system_prompt_text.replace(session_text_block_to_remove, "")
            if len(temp_prompt) < original_len: base_system_prompt_text = temp_prompt; self.logger.info(f"[{session_id}] Removed text block ({original_len - len(temp_prompt)} chars).")
            else: self.logger.warning(f"[{session_id}] Text block for removal '{session_text_block_to_remove[:50]}...' NOT FOUND.")
        else: self.logger.debug(f"[{session_id}] No text block for removal specified.")

        session_process_owi_rag = bool(getattr(user_valves, 'process_owi_rag', True))
        if not session_process_owi_rag: self.logger.info(f"[{session_id}] Session valve 'process_owi_rag=False'. Discarding OWI context."); extracted_owi_context = None; initial_owi_context_tokens = 0

        formatted_inventory_string = "[Inventory Management Disabled]"; raw_session_inventories = {}; inventory_enabled = getattr(self.config, 'enable_inventory_management', False)
        # Use the *local* flag determined during module load
        if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and self._get_all_inventories_db_func and self._format_inventory_func and self.sqlite_cursor:
            self.logger.debug(f"[{session_id}] Inventory enabled, fetching data...")
            try:
                raw_session_inventories = await self._get_all_inventories_db_func(self.sqlite_cursor, session_id)
                if raw_session_inventories:
                    self.logger.info(f"[{session_id}] Retrieved inventory data for {len(raw_session_inventories)} characters.")
                    try: formatted_inventory_string = self._format_inventory_func(raw_session_inventories); self.logger.info(f"[{session_id}] Formatted inventory string generated (len: {len(formatted_inventory_string)}).")
                    except Exception as e_fmt_inv: self.logger.error(f"[{session_id}] Failed to format inventory string: {e_fmt_inv}", exc_info=True); formatted_inventory_string = "[Error Formatting Inventory]"
                else: self.logger.info(f"[{session_id}] No inventory data found in DB for this session."); formatted_inventory_string = "[No Inventory Data Available]"
            except Exception as e_get_inv: self.logger.error(f"[{session_id}] Error retrieving inventory data from DB: {e_get_inv}", exc_info=True); formatted_inventory_string = "[Error Retrieving Inventory]"
        elif not inventory_enabled: self.logger.debug(f"[{session_id}] Skipping inventory fetch: Feature disabled by global valve.")
        elif inventory_enabled and not _ORCH_INVENTORY_MODULE_AVAILABLE: self.logger.warning(f"[{session_id}] Skipping inventory fetch: Module unavailable (Import failed).")
        else:
             missing_inv_funcs = [f for f, fn in {"db_get": self._get_all_inventories_db_func, "formatter": self._format_inventory_func, "cursor": self.sqlite_cursor}.items() if not fn]; self.logger.warning(f"[{session_id}] Skipping inventory fetch: Missing prerequisites: {missing_inv_funcs}")
             formatted_inventory_string = "[Inventory Init/Config Error]"

        context_for_prompt = extracted_owi_context; refined_context_tokens = -1; cache_update_performed = False; cache_update_skipped = False; final_context_selection_performed = False; stateless_refinement_performed = False; updated_cache_text_intermediate = "[Cache not initialized or updated]"
        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False); enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False)

        if enable_rag_cache_global and self._cache_update_func and self._cache_select_func and self._get_rag_cache_db_func and self.sqlite_cursor:
            self.logger.info(f"[{session_id}] RAG Cache Feature ENABLED.")
            run_step1 = False; previous_cache_text = "";
            try: cache_result = await self._get_rag_cache_db_func(self.sqlite_cursor, session_id); previous_cache_text = cache_result if cache_result is not None else ""
            except Exception as e_get_cache: self.logger.error(f"[{session_id}] Error retrieving previous cache: {e_get_cache}", exc_info=True)
            if not session_process_owi_rag: self.logger.info(f"[{session_id}] Skipping RAG Cache Step 1 (session valve 'process_owi_rag=False')."); cache_update_skipped = True; run_step1 = False; updated_cache_text_intermediate = previous_cache_text
            else:
                 skip_len = False; skip_sim = False; owi_content_for_check = extracted_owi_context or ""; len_thresh = getattr(self.config, 'CACHE_UPDATE_SKIP_OWI_THRESHOLD', 50)
                 if len(owi_content_for_check.strip()) < len_thresh: skip_len = True; self.logger.info(f"[{session_id}] Cache S1 Skip: OWI len < {len_thresh}.")
                 elif self._calculate_similarity_func and previous_cache_text:
                      sim_thresh = getattr(self.config, 'CACHE_UPDATE_SIMILARITY_THRESHOLD', 0.9)
                      try: sim_score = self._calculate_similarity_func(owi_content_for_check, previous_cache_text)
                      except Exception as e_sim: self.logger.error(f"[{session_id}] Error calculating similarity: {e_sim}")
                      else:
                          if sim_score > sim_thresh: skip_sim = True; self.logger.info(f"[{session_id}] Cache S1 Skip: Sim ({sim_score:.2f}) > {sim_thresh:.2f}.")
                 cache_update_skipped = skip_len or skip_sim; run_step1 = not cache_update_skipped
                 if cache_update_skipped: await self._emit_status(event_emitter, session_id, "Status: Skipping cache update (redundant OWI)."); updated_cache_text_intermediate = previous_cache_text

            cache_update_llm_config = { "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None), "temp": getattr(self.config, 'refiner_llm_temperature', 0.3), "prompt_template": getattr(self.config, 'cache_update_prompt_template', DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT),}
            final_select_llm_config = { "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None), "temp": getattr(self.config, 'refiner_llm_temperature', 0.3), "prompt_template": DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,}
            configs_ok_step1 = all([cache_update_llm_config["url"], cache_update_llm_config["key"], cache_update_llm_config["prompt_template"]])
            configs_ok_step2 = all([final_select_llm_config["url"], final_select_llm_config["key"], final_select_llm_config["prompt_template"]])

            if not (configs_ok_step1 and configs_ok_step2): self.logger.error(f"[{session_id}] Cannot proceed with RAG Cache: Refiner config missing."); await self._emit_status(event_emitter, session_id, "ERROR: RAG Cache Refiner config incomplete.", done=False); updated_cache_text_intermediate = previous_cache_text; run_step1 = False
            else:
                 if run_step1:
                      await self._emit_status(event_emitter, session_id, "Status: Updating background cache..."); self.logger.info(f"[{session_id}] Executing RAG Cache Step 1 (Update)...")
                      try:
                          updated_cache_text_intermediate = await self._cache_update_func( session_id=session_id, current_owi_context=extracted_owi_context, history_messages=current_active_history, latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper, sqlite_cursor=self.sqlite_cursor, cache_update_llm_config=cache_update_llm_config, history_count=getattr(self.config, 'refiner_history_count', 6), dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_CacheUpdate_{session_id}",)
                          cache_update_performed = True; self.logger.info(f"[{session_id}] RAG Cache Step 1 (Update) completed.")
                      except Exception as e_cache_update: self.logger.error(f"[{session_id}] EXCEPTION during RAG Cache Step 1 (Update): {e_cache_update}", exc_info=True); updated_cache_text_intermediate = previous_cache_text

                 await self._emit_status(event_emitter, session_id, "Status: Selecting relevant context...")
                 temp_owi_for_select = extracted_owi_context or ""
                 inv_context_to_inject = formatted_inventory_string
                 if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and inv_context_to_inject and "[Error" not in inv_context_to_inject and "[Disabled]" not in inv_context_to_inject and "[No Inventory Data Available]" not in inv_context_to_inject: temp_owi_for_select += f"\n\n--- Current Inventory ---\n{inv_context_to_inject}"; self.logger.info(f"[{session_id}] TEMPORARY: Injected formatted inventory into OWI context for selection step.")

                 self.logger.info(f"[{session_id}] Executing RAG Cache Step 2 (Select)...")
                 final_selected_context = await self._cache_select_func( updated_cache_text=(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else ""), current_owi_context=temp_owi_for_select, history_messages=current_active_history, latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper, context_selection_llm_config=final_select_llm_config, history_count=getattr(self.config, 'refiner_history_count', 6), dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_CtxSelect_{session_id}",)
                 final_context_selection_performed = True; context_for_prompt = final_selected_context; log_step1_status = "Performed" if cache_update_performed else ("Skipped" if cache_update_skipped else "Not Run")
                 self.logger.info(f"[{session_id}] RAG Cache Step 2 complete. Context len: {len(context_for_prompt)}. Step 1: {log_step1_status}")
                 await self._emit_status(event_emitter, session_id, "Status: Context selection complete.", done=False)

        elif enable_stateless_refin_global and self._stateless_refine_func:
            self.logger.info(f"[{session_id}] Stateless Refinement ENABLED.")
            await self._emit_status(event_emitter, session_id, "Status: Refining OWI context (stateless)...")
            if not extracted_owi_context: self.logger.debug(f"[{session_id}] Skipping stateless refinement: No OWI context.")
            elif not latest_user_query_str: self.logger.warning(f"[{session_id}] Skipping stateless refinement: Query empty.")
            else:
                 stateless_refiner_config = { "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None), "temp": getattr(self.config, 'refiner_llm_temperature', 0.3), "prompt_template": getattr(self.config, 'stateless_refiner_prompt_template', None),}
                 if not stateless_refiner_config["url"] or not stateless_refiner_config["key"]: self.logger.error(f"[{session_id}] Skipping stateless refinement: Refiner URL/Key missing."); await self._emit_status(event_emitter, session_id, "ERROR: Stateless Refiner config incomplete.", done=False)
                 else:
                      try:
                          refined_stateless_context = await self._stateless_refine_func( external_context=extracted_owi_context, history_messages=current_active_history, latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper, refiner_llm_config=stateless_refiner_config, skip_threshold=getattr(self.config, 'stateless_refiner_skip_threshold', 500), history_count=getattr(self.config, 'refiner_history_count', 6), dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_StatelessRef_{session_id}",)
                          if refined_stateless_context != extracted_owi_context: context_for_prompt = refined_stateless_context; stateless_refinement_performed = True; self.logger.info(f"[{session_id}] Stateless refinement successful (Length: {len(context_for_prompt)})."); await self._emit_status(event_emitter, session_id, "Status: OWI context refined (stateless).", done=False)
                          else: self.logger.info(f"[{session_id}] Stateless refinement resulted in no change or was skipped by length.")
                      except Exception as e_refine_stateless: self.logger.error(f"[{session_id}] EXCEPTION during stateless refinement: {e_refine_stateless}", exc_info=True)

        if self._count_tokens_func and self._tokenizer:
            try: token_source = context_for_prompt; refined_context_tokens = self._count_tokens_func(token_source, self._tokenizer) if token_source and isinstance(token_source, str) else 0
            except Exception as e_tok_ref: refined_context_tokens = -1; self.logger.error(f"[{session_id}] Error calculating refined tokens: {e_tok_ref}")
        else: refined_context_tokens = -1
        self.logger.debug(f"[{session_id}] Refined context tokens (RefOUT): {refined_context_tokens}")

        combined_context_string = "[No background context generated]"
        if self._combine_context_func:
            try:
                # Check inventory availability again *locally* before passing
                inventory_context_for_combine = None
                if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and formatted_inventory_string and "[Error" not in formatted_inventory_string and "[Disabled]" not in formatted_inventory_string:
                    inventory_context_for_combine = formatted_inventory_string

                combined_context_string = self._combine_context_func( final_selected_context=(context_for_prompt if isinstance(context_for_prompt, str) else None), t1_summaries=retrieved_t1_summaries, t2_rag_results=retrieved_rag_summaries, inventory_context=inventory_context_for_combine )
            except Exception as e_combine: self.logger.error(f"[{session_id}] Error combining context: {e_combine}", exc_info=True); combined_context_string = "[Error combining context]"
        else: self.logger.error(f"[{session_id}] Cannot combine context: Function unavailable.")
        self.logger.debug(f"[{session_id}] Combined background context length: {len(combined_context_string)}.")

        return ( combined_context_string, base_system_prompt_text, initial_owi_context_tokens, refined_context_tokens, cache_update_performed, cache_update_skipped, final_context_selection_performed, stateless_refinement_performed, formatted_inventory_string )


    async def _select_t0_history_slice(self, session_id: str, history_for_processing: List[Dict]) -> Tuple[List[Dict], int]:
        """ Selects the T0 history slice based on token limit and dialogue roles. """
        # (Code remains the same as previous version)
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


    async def _construct_final_payload(
        self, session_id: str, base_system_prompt_text: str, t0_raw_history_slice: List[Dict],
        combined_context_string: str, latest_user_query_str: str, user_valves: Any
    ) -> Optional[List[Dict]]:
        """ Constructs the final payload contents list for the LLM. """
        # (Code remains the same as previous version)
        await self._emit_status(event_emitter=None, session_id=session_id, description="Status: Constructing final request...") # Emitter not needed here
        final_llm_payload_contents = None
        if self._construct_payload_func:
            try:
                memory_guidance = "\n\n--- Memory Guidance ---\nUse the dialogue history and the background information provided (if any) to inform your response and maintain context."
                enhanced_system_prompt = base_system_prompt_text.strip() + memory_guidance
                session_long_term_goal = str(getattr(user_valves, 'long_term_goal', ''))
                include_acks = getattr(self.config, 'include_ack_turns', True)
                payload_dict = self._construct_payload_func( system_prompt=enhanced_system_prompt, history=t0_raw_history_slice, context=combined_context_string, query=latest_user_query_str, long_term_goal=session_long_term_goal, strategy="standard", include_ack_turns=include_acks,)
                if isinstance(payload_dict, dict) and "contents" in payload_dict and isinstance(payload_dict["contents"], list): final_llm_payload_contents = payload_dict["contents"]; self.logger.info(f"[{session_id}] Constructed final payload ({len(final_llm_payload_contents)} turns).")
                else: self.logger.error(f"[{session_id}] Payload constructor returned invalid structure: {type(payload_dict)}. Payload: {payload_dict}")
            except Exception as e_payload: self.logger.error(f"[{session_id}] EXCEPTION during payload construction: {e_payload}", exc_info=True)
        else: self.logger.error(f"[{session_id}] Cannot construct final payload: Function unavailable.")
        return final_llm_payload_contents


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
        # (Code remains the same as previous version)
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
    ) -> OrchestratorResult:
        """ Executes the final LLM call or prepares the output body. """
        # (Code remains the same as previous version)
        output_body = body.copy() if isinstance(body, dict) else {}
        if not final_llm_payload_contents: self.logger.error(f"[{session_id}] Final payload construction failed."); await self._emit_status(event_emitter, session_id, "ERROR: Final payload preparation failed.", done=True); return {"error": "Orchestrator: Final payload construction failed.", "status_code": 500}
        output_body["messages"] = final_llm_payload_contents; preserved_keys = ["model", "stream", "options", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"]; keys_preserved = [k for k in preserved_keys if k in body]; [output_body.update({k: body[k]}) for k in keys_preserved]; self.logger.info(f"[{session_id}] Output body updated. Preserved keys: {keys_preserved}.")
        final_url = getattr(self.config, 'final_llm_api_url', None); final_key = getattr(self.config, 'final_llm_api_key', None); url_present = bool(final_url and isinstance(final_url, str) and final_url.strip()); key_present = bool(final_key and isinstance(final_key, str) and final_key.strip()); self.logger.debug(f"[{session_id}] Checking Final LLM Trigger. URL Present:{url_present}, Key Present:{key_present}"); final_llm_triggered = url_present and key_present

        if final_llm_triggered:
            self.logger.info(f"[{session_id}] Final LLM Call via Pipe TRIGGERED (Non-Streaming)."); await self._emit_status(event_emitter, session_id, "Status: Executing final LLM Call...", done=False)
            final_temp = getattr(self.config, 'final_llm_temperature', 0.7); final_timeout = getattr(self.config, 'final_llm_timeout', 120); final_call_payload_google_fmt = {"contents": final_llm_payload_contents}
            success, response_or_error = await self._async_llm_call_wrapper( api_url=final_url, api_key=final_key, payload=final_call_payload_google_fmt, temperature=final_temp, timeout=final_timeout, caller_info=f"Orch_FinalLLM_{session_id}" )
            intermediate_status = "Status: Final LLM Complete" + (" (Success)" if success else " (Failed)"); await self._emit_status(event_emitter, session_id, intermediate_status, done=False)
            if success and isinstance(response_or_error, str): self.logger.info(f"[{session_id}] Final LLM call successful. Returning response string."); return response_or_error
            elif not success and isinstance(response_or_error, dict): self.logger.error(f"[{session_id}] Final LLM call failed. Returning error dict: {response_or_error}"); return response_or_error
            else: self.logger.error(f"[{session_id}] Final LLM call returned unexpected format. Success={success}, Type={type(response_or_error)}"); return {"error": "Final LLM call returned unexpected result format.", "status_code": 500}
        else:
            self.logger.info(f"[{session_id}] Final LLM Call disabled. Passing modified payload downstream.")
            if getattr(self.config, 'debug_log_final_payload', False):
                try: payload_str = json.dumps(output_body, indent=2)
                except Exception: payload_str = str(output_body)
                self.logger.debug(f"[{session_id}] Returning Payload:\n{payload_str}")
            return output_body

    # --- LOGGING HELPER ---
    def _log_debug_final_payload(self, session_id: str, payload_body: Dict):
        """Logs the final payload if the debug valve is enabled."""
        # This is a helper; the logic to get the path is usually in the Pipe class
        # For now, just log to the main logger if enabled.
        if getattr(self.config, 'debug_log_final_payload', False):
             try: payload_str = json.dumps(payload_body, indent=2)
             except Exception: payload_str = str(payload_body)
             self.logger.debug(f"[{session_id}] DEBUG_FINAL_PAYLOAD:\n{payload_str}")


    # ==========================================================================
    # === Main Process Turn Method                                           ===
    # ==========================================================================
    async def process_turn(
        self,
        session_id: str,
        user_id: str,
        body: Dict,
        user_valves: Any, # Receives the parsed object from Pipe.pipe
        event_emitter: Optional[Callable],
        embedding_func: Optional[Callable] = None,
        chroma_embed_wrapper: Optional[Any] = None,
        is_regeneration_heuristic: bool = False
    ) -> OrchestratorResult:
        """ Processes a single turn by calling helper methods in sequence. """
        pipe_entry_time_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Orchestrator process_turn [{session_id}]: Started at {pipe_entry_time_iso} (Regen Flag: {is_regeneration_heuristic})")

        inventory_enabled = getattr(self.config, 'enable_inventory_management', False)
        self.logger.info(f"[{session_id}] Inventory Management Enabled (Global Valve): {inventory_enabled}")
        self.logger.info(f"[{session_id}] Inventory Module Available (Local Import Check): {_ORCH_INVENTORY_MODULE_AVAILABLE}")

        # Initialize variables
        summarization_performed = False; new_t1_summary_text = None; summarization_prompt_tokens = -1; summarization_output_tokens = -1; t1_retrieved_count = 0; t2_retrieved_count = 0; retrieved_rag_summaries = []; cache_update_performed = False; cache_update_skipped = False; final_context_selection_performed = False; stateless_refinement_performed = False; initial_owi_context_tokens = -1; refined_context_tokens = -1; t0_dialogue_tokens = -1; final_payload_tokens = -1; inventory_prompt_tokens = -1; formatted_inventory_string_for_status = ""; final_result: Optional[OrchestratorResult] = None; final_llm_payload_contents: Optional[List[Dict]] = None; inventory_update_completed = False; inventory_update_success_flag = False

        try:
            # 1. History Sync
            await self._emit_status(event_emitter, session_id, "Status: Orchestrator syncing history...")
            incoming_messages = body.get("messages", [])
            stored_history = self.session_manager.get_active_history(session_id) or []
            if incoming_messages != stored_history: self.session_manager.set_active_history(session_id, incoming_messages.copy()); self.logger.debug(f"[{session_id}] Updating active history (Len: {len(incoming_messages)}).")
            else: self.logger.debug(f"[{session_id}] Incoming history matches stored.")
            current_active_history = self.session_manager.get_active_history(session_id) or []
            if not current_active_history: raise ValueError("Active history is empty after sync.")

            # 2. Determine Query
            latest_user_query_str, history_for_processing = await self._determine_effective_query( session_id, current_active_history, is_regeneration_heuristic )
            if not latest_user_query_str and not is_regeneration_heuristic: raise ValueError("Cannot proceed without an effective user query (and not regeneration).")

            # 3. Tier 1 Summarization
            (summarization_performed, new_t1_summary_text, summarization_prompt_tokens, summarization_output_tokens) = await self._handle_tier1_summarization( session_id, user_id, current_active_history, is_regeneration_heuristic, event_emitter )

            # 4. Tier 1 -> T2 Transition
            await self._handle_tier2_transition( session_id, summarization_performed, chroma_embed_wrapper, event_emitter )

            # 5. Get Context Sources
            recent_t1_summaries, t1_retrieved_count = await self._get_t1_summaries(session_id)
            retrieved_rag_summaries, t2_retrieved_count = await self._get_t2_rag_results( session_id, history_for_processing, latest_user_query_str, embedding_func, chroma_embed_wrapper, event_emitter )

            # 6. Prepare & Refine Background
            (combined_context_string, base_system_prompt_text, initial_owi_context_tokens, refined_context_tokens, cache_update_performed, cache_update_skipped, final_context_selection_performed, stateless_refinement_performed, formatted_inventory_string_for_status ) = await self._prepare_and_refine_background( session_id, body, user_valves, recent_t1_summaries, retrieved_rag_summaries, current_active_history, latest_user_query_str, event_emitter )

            # 7. Select T0 History
            t0_raw_history_slice, t0_dialogue_tokens = await self._select_t0_history_slice( session_id, history_for_processing )

            # 8. Construct Payload
            final_llm_payload_contents = await self._construct_final_payload( session_id, base_system_prompt_text, t0_raw_history_slice, combined_context_string, latest_user_query_str, user_valves )

            # 9. Execute/Prepare Output
            final_result = await self._execute_or_prepare_output( session_id=session_id, body=body, final_llm_payload_contents=final_llm_payload_contents, event_emitter=event_emitter, status_message="Status: Core processing complete.", final_payload_tokens=-1 )

            # 10. Post-Turn Inventory Update
            # Check BOTH global valve AND local module availability
            if inventory_enabled and _ORCH_INVENTORY_MODULE_AVAILABLE and self._update_inventories_func:
                inventory_update_completed = True
                if isinstance(final_result, dict) and "error" in final_result: self.logger.warning(f"[{session_id}] Skipping inventory update due to upstream error: {final_result.get('error')}")
                elif isinstance(final_result, AsyncGenerator): self.logger.warning(f"[{session_id}] Skipping inventory update: Streaming response detected.")
                elif isinstance(final_result, str):
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
                                     # *** Pass the *locally determined* DB access functions ***
                                     update_success = await self._update_inventories_func(
                                         cursor=new_cursor, session_id=session_id, main_llm_response=final_result, user_query=latest_user_query_str, recent_history_str=history_for_inv_update_str,
                                         llm_call_func=self._async_llm_call_wrapper, # Pass wrapper
                                         db_get_inventory_func=get_character_inventory_data, # Pass direct DB func
                                         db_update_inventory_func=add_or_update_character_inventory, # Pass direct DB func
                                         inventory_llm_config=inv_llm_config
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
                elif isinstance(final_result, dict) and final_result.get("messages") == final_llm_payload_contents: self.logger.info(f"[{session_id}] Skipping inventory update: Final LLM call disabled or payload unchanged.")
                else: self.logger.error(f"[{session_id}] Unexpected type for final_result: {type(final_result)}. Skipping inventory update.")
            elif inventory_enabled and not _ORCH_INVENTORY_MODULE_AVAILABLE:
                 self.logger.warning(f"[{session_id}] Skipping inventory update: Module import failed.")
                 inventory_update_completed = False; # Mark as not completed if module missing
                 inventory_update_success_flag = False;
            elif inventory_enabled and not self._update_inventories_func:
                 self.logger.error(f"[{session_id}] Skipping inventory update: Update function alias is None.")
                 inventory_update_completed = False;
                 inventory_update_success_flag = False;
            else: # Inventory disabled globally
                 self.logger.debug(f"[{session_id}] Skipping inventory update: Disabled by global valve.")
                 inventory_update_completed = False;
                 inventory_update_success_flag = False;

            # 11. Calculate FINAL Status
            # (Recalculate final tokens just in case)
            if final_llm_payload_contents and self._count_tokens_func and self._tokenizer:
                try: final_payload_tokens = sum(self._count_tokens_func(part["text"], self._tokenizer) for turn in final_llm_payload_contents if isinstance(turn, dict) for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str))
                except Exception: final_payload_tokens = -1
            elif not final_llm_payload_contents: final_payload_tokens = 0
            else: final_payload_tokens = -1

            final_status_string_base, _ = await self._calculate_and_format_status( session_id=session_id, t1_retrieved_count=t1_retrieved_count, t2_retrieved_count=t2_retrieved_count, session_process_owi_rag=bool(getattr(user_valves, 'process_owi_rag', True)), final_context_selection_performed=final_context_selection_performed, cache_update_skipped=cache_update_skipped, stateless_refinement_performed=stateless_refinement_performed, initial_owi_context_tokens=initial_owi_context_tokens, refined_context_tokens=refined_context_tokens, summarization_prompt_tokens=summarization_prompt_tokens, summarization_output_tokens=summarization_output_tokens, t0_dialogue_tokens=t0_dialogue_tokens, inventory_prompt_tokens=inventory_prompt_tokens, final_llm_payload_contents=final_llm_payload_contents)

            # Append Inventory Status
            inv_stat_indicator = "Inv=OFF";
            if inventory_enabled:
                 # Check local flag first
                 if not _ORCH_INVENTORY_MODULE_AVAILABLE: inv_stat_indicator = "Inv=MISSING"
                 else:
                     inv_stat_indicator = "Inv=ON" # Assume ON if enabled and module present
                     if not inventory_update_completed: inv_stat_indicator = "Inv=SKIP" # Skipped due to error/streaming etc.
                     elif not inventory_update_success_flag: inv_stat_indicator = "Inv=FAIL" # Attempted but failed
            final_status_string = final_status_string_base + f" {inv_stat_indicator}"
            self.logger.debug(f"[{session_id}] Orchestrator: About to emit FINAL status: '{final_status_string}'")
            await self._emit_status(event_emitter, session_id, final_status_string, done=True)

            # 12. Log Final Payload (if applicable and debug enabled)
            if getattr(self.config, 'debug_log_final_payload', False):
                if isinstance(final_result, dict) and "messages" in final_result:
                     final_url = getattr(self.config, 'final_llm_api_url', None); final_key = getattr(self.config, 'final_llm_api_key', None); final_llm_triggered = bool(final_url and final_key)
                     if not final_llm_triggered: self.logger.info(f"[{session_id}] Logging final payload dict due to debug valve (Final LLM Off)."); self._log_debug_final_payload(session_id, final_result)
                     else: self.logger.debug(f"[{session_id}] Skipping final payload log: Final LLM was triggered.")

            # 13. Return
            pipe_end_time_iso = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"Orchestrator process_turn [{session_id}]: Finished at {pipe_end_time_iso}")
            if final_result is None: raise RuntimeError("Internal processing error, final result was None.")
            return final_result

        except asyncio.CancelledError: self.logger.info(f"[{session_id or 'unknown'}] Orchestrator process_turn cancelled."); await self._emit_status(event_emitter, session_id or 'unknown', "Status: Processing cancelled.", done=True); raise
        except ValueError as ve: # Catch specific errors like history empty or query missing
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