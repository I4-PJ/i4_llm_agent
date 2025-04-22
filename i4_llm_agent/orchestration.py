# [[START MODIFIED orchestration.py]]
# i4_llm_agent/orchestration.py

import logging
import asyncio
import re
import sqlite3
import json
import uuid
from .utils import TIKTOKEN_AVAILABLE
from datetime import datetime, timezone
from typing import (
    Tuple, Union, List, Dict, Optional, Any, Callable, Coroutine, AsyncGenerator
)

# --- Standard Library Imports ---
# (Add others as needed)

# --- i4_llm_agent Imports ---
from .session import SessionManager
from .database import (
    # SQLite T1/Cache
    add_tier1_summary, get_recent_tier1_summaries, get_tier1_summary_count,
    get_oldest_tier1_summary, delete_tier1_summary,
    get_max_t1_end_index, # Used for T1 start and T0 slice
    add_or_update_rag_cache, get_rag_cache,
    # Chroma T2
    get_or_create_chroma_collection, add_to_chroma_collection,
    query_chroma_collection, get_chroma_collection_count,
    CHROMADB_AVAILABLE, ChromaEmbeddingFunction, ChromaCollectionType, # Use types if needed
    InvalidDimensionException,
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
)
from .cache import ( # Import orchestrators from cache.py
    update_rag_cache, select_final_context
)
# Import the correct dispatcher function
from .api_client import call_google_llm_api # Use the dispatcher

from .utils import count_tokens, calculate_string_similarity

# --- Optional Imports ---
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False
    # Log warning during initialization if needed

# Need this for URL parsing within the stream call prep
import urllib.parse

# Need this for the stream call payload conversion
try:
    from .api_client import _convert_google_to_openai_payload
except ImportError:
    _convert_google_to_openai_payload = None # Handle gracefully if not found


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
    """

    def __init__(
        self,
        config: object, # Expects an object with attributes similar to Pipe.Valves
        session_manager: SessionManager,
        sqlite_cursor: sqlite3.Cursor,
        chroma_client: Optional[Any] = None, # Expects chromadb.ClientAPI
        logger_instance: Optional[logging.Logger] = None,
    ):
        # ... (Initialization remains the same as before) ...
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

        # Direct function aliases from library (assuming library functions handle None checks)
        self._llm_call_func = call_google_llm_api # Use dispatcher
        self._format_history_func = format_history_for_llm
        self._get_recent_turns_func = get_recent_turns
        self._manage_memory_func = manage_tier1_summarization
        self._construct_payload_func = construct_final_llm_payload
        self._clean_context_tags_func = clean_context_tags
        self._generate_rag_query_func = generate_rag_query
        self._combine_context_func = combine_background_context
        self._process_system_prompt_func = process_system_prompt
        self._count_tokens_func = count_tokens
        self._calculate_similarity_func = calculate_string_similarity
        self._dialogue_roles = DIALOGUE_ROLES

        # Refinement/Cache functions
        self._stateless_refine_func = refine_external_context
        self._cache_update_func = update_rag_cache
        self._cache_select_func = select_final_context
        self._get_rag_cache_db_func = get_rag_cache # Direct DB getter

        self.logger.info("SessionPipeOrchestrator initialized.")

    # --- Internal Helper: Status Emitter ---
    async def _emit_status(
        self,
        event_emitter: Optional[Callable],
        session_id: str, # Added session_id for logging context
        description: str,
        done: bool = False
    ):
        # ... (Implementation remains the same as before) ...
        if event_emitter and callable(event_emitter) and getattr(self.config, 'emit_status_updates', True):
            try:
                status_data = {
                    "type": "status",
                    "data": {"description": str(description), "done": bool(done)}
                }
                await event_emitter(status_data)
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
        # ... (Implementation remains the same as before) ...
        if not self._llm_call_func:
            self.logger.error(f"[{caller_info}] LLM func unavailable in orchestrator.")
            return False, {"error_type": "SetupError", "message": "LLM func unavailable"}
        try:
             return await asyncio.to_thread(
                 self._llm_call_func,
                 api_url=api_url, api_key=api_key, payload=payload,
                 temperature=temperature, timeout=timeout, caller_info=caller_info
             )
        except Exception as e:
            self.logger.error(f"Orchestrator LLM Wrapper Error [{caller_info}]: {e}", exc_info=True)
            return False, {"error_type": "AsyncWrapperError", "message": f"{type(e).__name__}"}


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
        # ... (Implementation remains the same as before) ...
        log_session_id = caller_info
        session_id_match = re.search(r'_(user_.*)', log_session_id)
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

            if "openrouter.ai/api/v1/chat/completions" in url_for_check or \
               url_for_check.endswith("/v1/chat/completions"):
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

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/event-stream"
                    }
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
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                 async with client.stream("POST", base_api_url, headers=headers, json=final_payload_to_send, timeout=timeout) as response:
                    if response.status_code != 200:
                         error_body = await response.aread()
                         error_text = error_body.decode('utf-8', errors='replace')[:1000]
                         self.logger.error(f"[{log_session_id}] Stream API Error: Status {response.status_code}. Response: {error_text}...")
                         yield f"[Streaming Error: API returned status {response.status_code}]"
                         try:
                             err_json = json.loads(error_text)
                             if "error" in err_json and isinstance(err_json["error"], dict):
                                 err_detail = err_json["error"].get("message", "Unknown API error")
                                 yield f"\n[Detail: {err_detail}]"
                         except json.JSONDecodeError: pass
                         return

                    self.logger.debug(f"[{log_session_id}] Stream connection successful (Status {response.status_code}). Reading chunks...")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_content = line[len("data:"):].strip()
                            if data_content == "[DONE]":
                                self.logger.debug(f"[{log_session_id}] Received [DONE] signal.")
                                stream_successful = True
                                break
                            if data_content:
                                try:
                                    chunk = json.loads(data_content)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    text_chunk = delta.get("content")
                                    if text_chunk: yield text_chunk
                                except json.JSONDecodeError: self.logger.warning(f"[{log_session_id}] Failed to decode JSON data chunk: '{data_content}'")
                                except (IndexError, KeyError, TypeError) as e_parse: self.logger.warning(f"[{log_session_id}] Error parsing stream chunk structure ({type(e_parse).__name__}): {chunk}")
                        elif line.strip(): self.logger.debug(f"[{log_session_id}] Received non-data line: '{line}'")

        except httpx.TimeoutException as e_timeout:
            self.logger.error(f"[{log_session_id}] HTTPX Timeout during stream after {timeout}s: {e_timeout}", exc_info=False)
            yield f"[Streaming Error: Timeout after {timeout}s]"
        except httpx.RequestError as e_req:
            self.logger.error(f"[{log_session_id}] HTTPX RequestError during stream: {e_req}", exc_info=True)
            yield f"[Streaming Error: Network request failed - {type(e_req).__name__}]"
        except asyncio.CancelledError:
             self.logger.info(f"[{log_session_id}] Final stream call explicitly cancelled.")
             yield "[Streaming Error: Cancelled]"
             raise
        except Exception as e_stream:
            self.logger.error(f"[{log_session_id}] Unexpected error during final LLM stream: {e_stream}", exc_info=True)
            yield f"[Streaming Error: Unexpected error - {type(e_stream).__name__}]"
        finally:
             self.logger.info(f"[{log_session_id}] Final LLM stream processing finished.")
             status_to_emit = final_status_message
             if not stream_successful:
                  status_to_emit += " (Stream Interrupted)"
             await self._emit_status(
                 event_emitter=event_emitter,
                 session_id=extracted_session_id,
                 description=status_to_emit,
                 done=True
             )


    # --- Helper Methods for process_turn ---

    async def _initialize_turn_state(
        self, body: Dict, __user__: Optional[dict]
    ) -> Tuple[str, str, Any, List[Dict], bool, Dict]:
        """
        Validates input, gets/creates session, parses user valves, syncs history,
        and detects regeneration.
        """
        user_id = "default_user"
        if isinstance(__user__, dict) and "id" in __user__:
            user_id = __user__["id"]
        else:
            self.logger.warning(f"User info/ID missing. Using '{user_id}'.")

        chat_id = body.get("chat_id") # Get chat_id from body if available (fallback)
        if not chat_id: # Prefer chat_id from body, fallback to session_id logic if needed
            session_id = body.get("session_id", f"user_{user_id}_chat_unknown") # Basic fallback
            self.logger.warning(f"chat_id missing in body, using derived session_id: {session_id}")
        else:
            safe_chat_id_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(chat_id))
            session_id = f"user_{user_id}_chat_{safe_chat_id_part}"

        self.logger.info(f"Derived Session ID: {session_id}")

        # Get/Create Session State & Handle User Valves
        session_state = self.session_manager.get_or_create_session(session_id)
        user_valves_obj = None
        if isinstance(__user__, dict) and "valves" in __user__:
            raw_user_valves = __user__["valves"]
            try:
                user_valves_obj = self.config.UserValves(**(raw_user_valves if isinstance(raw_user_valves, dict) else {})) # Use config's UserValves
                self.logger.info(f"[{session_id}] Parsed UserValves.")
            except Exception as e_parse_uv:
                self.logger.warning(f"[{session_id}] Failed to parse UserValves: {e_parse_uv}. Using defaults.")
                user_valves_obj = self.config.UserValves()
        else:
            self.logger.debug(f"[{session_id}] No __user__['valves'] found. Using default UserValves.")
            user_valves_obj = self.config.UserValves()

        self.session_manager.set_user_valves(session_id, user_valves_obj)

        # Synchronize History & Detect Regeneration
        incoming_messages = body.get("messages", [])
        stored_history = self.session_manager.get_active_history(session_id) or []
        previous_input = self.session_manager.get_previous_input_messages(session_id)

        if incoming_messages != stored_history:
            if len(incoming_messages) < len(stored_history):
                self.logger.warning(f"[{session_id}] Incoming history shorter than stored. Resetting.")
                self.session_manager.set_active_history(session_id, incoming_messages.copy())
                self.session_manager.set_last_summary_index(session_id, -1)
            else:
                self.logger.debug(f"[{session_id}] Updating active history (Len: {len(incoming_messages)}).")
                self.session_manager.set_active_history(session_id, incoming_messages.copy())
        else:
            self.logger.debug(f"[{session_id}] Incoming history matches stored.")

        current_active_history = self.session_manager.get_active_history(session_id) # Get the potentially updated history

        # Store current input for next cycle's regeneration check BEFORE potential modifications
        self.session_manager.set_previous_input_messages(session_id, incoming_messages.copy())

        is_regeneration_heuristic = (
             previous_input is not None and
             incoming_messages == previous_input and
             len(incoming_messages) > 0
        )

        return session_id, user_id, user_valves_obj, current_active_history, is_regeneration_heuristic, session_state


    async def _determine_effective_query(
        self, session_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool
    ) -> Tuple[str, List[Dict]]:
        """ Determines the effective user query and the history slice preceding it. """
        effective_user_message_index = -1
        user_message_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") == "user"]

        if not user_message_indices:
            self.logger.error(f"[{session_id}] No user messages found in history.")
            # Return empty defaults, main process_turn should handle this error earlier
            return "", []

        if is_regeneration_heuristic:
            effective_user_message_index = user_message_indices[-2] if len(user_message_indices) >= 2 else user_message_indices[-1]
            log_level = self.logger.info if len(user_message_indices) >= 2 else self.logger.warning
            log_level(f"[{session_id}] Regen: Using user message index {effective_user_message_index} as query.")
        else:
            effective_user_message_index = user_message_indices[-1]
            self.logger.debug(f"[{session_id}] Normal: Using user message index {effective_user_message_index} as query.")

        effective_user_message = current_active_history[effective_user_message_index]
        history_for_processing = current_active_history[:effective_user_message_index]
        latest_user_query_str = effective_user_message.get("content", "")

        self.logger.debug(f"[{session_id}] Effective query set (len: {len(latest_user_query_str)}). History slice len: {len(history_for_processing)}.")
        return latest_user_query_str, history_for_processing


    async def _handle_tier1_summarization(
        self, session_id: str, user_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool, event_emitter: Optional[Callable]
    ) -> Tuple[bool, Optional[str], int, int]:
        """ Checks and performs T1 summarization, skipping LLM call if regenerating an existing block. """
        await self._emit_status(event_emitter, session_id, "Status: Checking summarization...")

        # --- Regeneration check flag is now used INSIDE manage_tier1_summarization ---
        # We no longer skip the entire function call here.

        summarization_performed_successfully = False
        generated_summary = None
        summarization_prompt_tokens = -1
        summarization_output_tokens = -1

        can_summarize = all([
            self._manage_memory_func, self._tokenizer, self._count_tokens_func,
            self.sqlite_cursor, self._async_llm_call_wrapper,
            hasattr(self.config, 'summarizer_api_url') and self.config.summarizer_api_url,
            hasattr(self.config, 'summarizer_api_key') and self.config.summarizer_api_key,
            current_active_history,
        ])

        if can_summarize:
            summarizer_llm_config = {
                 "url": self.config.summarizer_api_url, "key": self.config.summarizer_api_key,
                 "temp": getattr(self.config, 'summarizer_temperature', 0.5),
                 "sys_prompt": getattr(self.config, 'summarizer_system_prompt', "Summarize this dialogue."),
             }
            new_last_summary_idx = -1
            prompt_tokens = -1
            t0_end_idx = -1

            # Get start index from DB
            db_max_index = None
            current_last_summary_index_for_memory = -1
            try:
                db_max_index = await get_max_t1_end_index(self.sqlite_cursor, session_id)
                if isinstance(db_max_index, int) and db_max_index >= 0:
                    current_last_summary_index_for_memory = db_max_index
                    self.logger.info(f"[{session_id}] T1: Start Index from DB: {current_last_summary_index_for_memory}")
                else:
                    self.logger.info(f"[{session_id}] T1: No valid start index in DB. Starting from -1.")
            except Exception as e_get_max:
                self.logger.error(f"[{session_id}] T1: Error getting start index: {e_get_max}. Starting from -1.", exc_info=True)
                current_last_summary_index_for_memory = -1

            # Nested save function
            async def _async_save_t1_summary(summary_id: str, session_id: str, user_id: str, summary_text: str, metadata: Dict):
                 return await add_tier1_summary(cursor=self.sqlite_cursor, summary_id=summary_id, session_id=session_id, user_id=user_id, summary_text=summary_text, metadata=metadata)

            try:
                self.logger.debug(f"[{session_id}] Calling manage_tier1_summarization with start index = {current_last_summary_index_for_memory} (Regen={is_regeneration_heuristic})")
                # <<< Pass cursor and regeneration flag >>>
                summarization_performed, generated_summary_text, new_last_summary_idx, prompt_tokens, t0_end_idx = await self._manage_memory_func(
                    current_last_summary_index=current_last_summary_index_for_memory,
                    active_history=current_active_history,
                    t0_token_limit=getattr(self.config, 't0_active_history_token_limit', 4000),
                    t1_chunk_size_target=getattr(self.config, 't1_summarization_chunk_token_target', 2000),
                    tokenizer=self._tokenizer,
                    llm_call_func=self._async_llm_call_wrapper,
                    llm_config=summarizer_llm_config,
                    add_t1_summary_func=_async_save_t1_summary,
                    session_id=session_id, user_id=user_id,
                    cursor=self.sqlite_cursor, # <<< Pass cursor
                    is_regeneration=is_regeneration_heuristic, # <<< Pass flag
                    dialogue_only_roles=self._dialogue_roles,
                )
                # <<< End Pass >>>

                if summarization_performed:
                    summarization_performed_successfully = True
                    generated_summary = generated_summary_text
                    summarization_prompt_tokens = prompt_tokens
                    # Update in-memory index for this cycle
                    self.session_manager.set_last_summary_index(session_id, new_last_summary_idx)
                    if generated_summary and self._count_tokens_func and self._tokenizer:
                        try: summarization_output_tokens = self._count_tokens_func(generated_summary, self._tokenizer)
                        except Exception: summarization_output_tokens = -1
                    self.logger.info(f"[{session_id}] T1 summary generated/saved. NewIdx: {new_last_summary_idx}.")
                    await self._emit_status(event_emitter, session_id, "Status: Summary generated.", done=False)
                else:
                    # This log now also covers the case where summarization was skipped due to regeneration check
                    self.logger.debug(f"[{session_id}] T1 summarization skipped or criteria not met.")
            except TypeError as e_type:
                 self.logger.error(f"[{session_id}] Orchestrator TYPE ERROR calling T1 manage func: {e_type}. Signature mismatch?", exc_info=True)
            except Exception as e_manage:
                self.logger.error(f"[{session_id}] Orchestrator EXCEPTION during T1 manage call: {e_manage}", exc_info=True)
        else:
             missing_prereqs = [p for p, v in {
                 "manage_func": self._manage_memory_func, "tokenizer": self._tokenizer,
                 "count_func": self._count_tokens_func, "db_cursor": self.sqlite_cursor,
                 "llm_wrapper": self._async_llm_call_wrapper,
                 "summ_url": getattr(self.config, 'summarizer_api_url', None),
                 "summ_key": getattr(self.config, 'summarizer_api_key', None),
                 "history": bool(current_active_history)
             }.items() if not v]
             self.logger.warning(f"[{session_id}] Skipping T1 check: Missing prerequisites: {', '.join(missing_prereqs)}.")

        return summarization_performed_successfully, generated_summary, summarization_prompt_tokens, summarization_output_tokens


    async def _handle_tier2_transition(
        self, session_id: str, t1_success: bool, chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> None:
        """ Handles the transition of the oldest T1 summary to T2 if needed. """
        await self._emit_status(event_emitter, session_id, "Status: Checking long-term memory capacity...")
        tier2_collection = None # Initialize locally

        if self.chroma_client and chroma_embed_wrapper:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_')
            safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]
            tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
            tier2_collection = await get_or_create_chroma_collection(
                 self.chroma_client, tier2_collection_name, chroma_embed_wrapper
            )

        can_transition = all([
            t1_success, tier2_collection is not None, # Check if T1 actually happened this turn
            chroma_embed_wrapper is not None, self.sqlite_cursor is not None,
            getattr(self.config, 'max_stored_summary_blocks', 0) > 0
        ])

        if not can_transition:
            self.logger.debug(f"[{session_id}] Skipping T1->T2 transition check: Prerequisites not met (T1 Success: {t1_success}).")
            return

        try:
            max_t1_blocks = self.config.max_stored_summary_blocks
            current_tier1_count = await get_tier1_summary_count(self.sqlite_cursor, session_id)

            if current_tier1_count == -1:
                self.logger.error(f"[{session_id}] Failed get T1 count. Skipping T1->T2 check.")
            elif current_tier1_count > max_t1_blocks:
                self.logger.info(f"[{session_id}] T1 limit ({max_t1_blocks}) exceeded ({current_tier1_count}). Transitioning...")
                await self._emit_status(event_emitter, session_id, "Status: Archiving oldest summary...")
                oldest_summary_data = await get_oldest_tier1_summary(self.sqlite_cursor, session_id)

                if oldest_summary_data:
                    oldest_id, oldest_text, oldest_metadata = oldest_summary_data
                    embedding_vector = None; embedding_successful = False
                    try:
                        embedding_list = await asyncio.to_thread(chroma_embed_wrapper, [oldest_text])
                        if isinstance(embedding_list, list) and len(embedding_list) == 1 and isinstance(embedding_list[0], list) and len(embedding_list[0]) > 0:
                             embedding_vector = embedding_list[0]; embedding_successful = True
                        else: self.logger.error(f"[{session_id}] T1->T2 Embed: Invalid structure: {embedding_list}")
                    except Exception as embed_e: self.logger.error(f"[{session_id}] EXCEPTION embedding T1->T2 {oldest_id}: {embed_e}", exc_info=True)

                    if embedding_successful and embedding_vector:
                        added_to_t2 = False; deleted_from_t1 = False
                        chroma_metadata = oldest_metadata.copy()
                        chroma_metadata["transitioned_from_t1"] = True
                        chroma_metadata["original_t1_id"] = oldest_id
                        sanitized_chroma_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in chroma_metadata.items() if v is not None}
                        tier2_id = f"t2_{oldest_id}"
                        self.logger.info(f"[{session_id}] Adding summary {tier2_id} to T2 '{tier2_collection.name}'...")
                        added_to_t2 = await add_to_chroma_collection(
                             tier2_collection, ids=[tier2_id], embeddings=[embedding_vector],
                             metadatas=[sanitized_chroma_metadata], documents=[oldest_text]
                        )
                        if added_to_t2:
                             self.logger.info(f"[{session_id}] Added {tier2_id} to T2. Deleting T1 {oldest_id}...")
                             deleted_from_t1 = await delete_tier1_summary(self.sqlite_cursor, oldest_id)
                             if deleted_from_t1: await self._emit_status(event_emitter, session_id, "Status: Summary archive complete.", done=False)
                             else: self.logger.warning(f"[{session_id}] Added {tier2_id} to T2, but FAILED delete T1 {oldest_id}.")
                    else: self.logger.error(f"[{session_id}] Skipping T2 add for {oldest_id}: embedding failed.")
                else: self.logger.warning(f"[{session_id}] T1 count exceeded limit, but couldn't retrieve oldest.")
            else:
                self.logger.debug(f"[{session_id}] T1 count ({current_tier1_count}) within limit. No transition needed.")
        except Exception as e_t2_trans:
            self.logger.error(f"[{session_id}] Unexpected error during T1->T2 transition: {e_t2_trans}", exc_info=True)


    async def _resolve_embedding_context(self, __request__, __user__) -> Tuple[Optional[Callable], Optional[Any]]:
        """Gets OWI embedding function and creates ChromaDB wrapper."""
        # This method requires access to the Pipe's _owi_embedding_func_cache
        # and OWI-specific imports/logic, so it's kept separate for now.
        # It might need to be called from the Pipe class itself, or the cache
        # needs to be managed differently if called from Orchestrator.
        # For now, assume it's called by the Pipe and results passed to Orchestrator.
        # Placeholder implementation:
        self.logger.warning("_resolve_embedding_context needs OWI specific logic from Pipe class.")
        return None, None # Simulate failure or needs implementation within Pipe class


    async def _get_t1_summaries(self, session_id: str) -> Tuple[List[str], int]:
        """ Fetches recent T1 summaries from the database. """
        recent_t1_summaries = []
        t1_retrieved_count = 0
        if self.sqlite_cursor and getattr(self.config, 'max_stored_summary_blocks', 0) > 0:
             try:
                 max_blocks = getattr(self.config, 'max_stored_summary_blocks', 10) # Ensure default
                 recent_t1_summaries = await get_recent_tier1_summaries(self.sqlite_cursor, session_id, max_blocks)
                 t1_retrieved_count = len(recent_t1_summaries)
             except Exception as e_get_t1: self.logger.error(f"[{session_id}] Error retrieving T1: {e_get_t1}", exc_info=True)
        if t1_retrieved_count > 0: self.logger.info(f"[{session_id}] Retrieved {t1_retrieved_count} T1 summaries.")
        return recent_t1_summaries, t1_retrieved_count


    async def _get_t2_rag_results(
        self, session_id: str, history_for_processing: List[Dict], latest_user_query_str: str,
        embedding_func: Optional[Callable], chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> Tuple[List[str], int]:
        """ Performs T2 RAG lookup. """
        await self._emit_status(event_emitter, session_id, "Status: Searching long-term memory...")
        retrieved_rag_summaries = []
        t2_retrieved_count = 0
        tier2_collection = None

        if self.chroma_client and chroma_embed_wrapper:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_')
            safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]
            tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
            tier2_collection = await get_or_create_chroma_collection(
                 self.chroma_client, tier2_collection_name, chroma_embed_wrapper
            )

        can_rag = all([
            tier2_collection is not None, latest_user_query_str, # Use query directly
            embedding_func is not None, self._generate_rag_query_func is not None,
            self._async_llm_call_wrapper is not None,
            getattr(self.config, 'ragq_llm_api_url', None), getattr(self.config, 'ragq_llm_api_key', None),
            getattr(self.config, 'ragq_llm_prompt', None), getattr(self.config, 'rag_summary_results_count', 0) > 0,
        ])

        if not can_rag:
            self.logger.info(f"[{session_id}] Skipping T2 RAG check: Prerequisites not met.")
            return [], 0

        t2_doc_count = await get_chroma_collection_count(tier2_collection)
        if t2_doc_count <= 0:
            self.logger.info(f"[{session_id}] Skipping T2 RAG: Collection '{tier2_collection.name}' is empty or count failed ({t2_doc_count}).")
            return [], 0

        try:
            await self._emit_status(event_emitter, session_id, "Status: Generating search query...")
            context_messages_for_ragq = self._get_recent_turns_func(
                 history_for_processing, count=6, exclude_last=False, roles=self._dialogue_roles
            )
            dialogue_context_str = self._format_history_func(context_messages_for_ragq) if context_messages_for_ragq else "[No recent history]"
            ragq_llm_config = {
                 "url": self.config.ragq_llm_api_url, "key": self.config.ragq_llm_api_key,
                 "temp": getattr(self.config, 'ragq_llm_temperature', 0.3), "prompt": self.config.ragq_llm_prompt,
            }
            rag_query = await self._generate_rag_query_func(
                 latest_message_str=latest_user_query_str, dialogue_context_str=dialogue_context_str,
                 llm_call_func=self._async_llm_call_wrapper, llm_config=ragq_llm_config,
                 caller_info=f"Orch_RAGQ_{session_id}",
            )

            if not (rag_query and isinstance(rag_query, str) and not rag_query.startswith("[Error:") and rag_query.strip()):
                 self.logger.error(f"[{session_id}] RAG Query Generation failed: {rag_query}.")
                 return [], 0

            if not embedding_func:
                self.logger.error(f"[{session_id}] Cannot embed RAG query: Embedding func missing.")
                return [], 0

            await self._emit_status(event_emitter, session_id, "Status: Embedding search query...")
            query_embedding = None; query_embedding_successful = False
            try:
                # Use the global import if available
                from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX
                query_embedding_list = await asyncio.to_thread(embedding_func, [rag_query], prefix=RAG_EMBEDDING_QUERY_PREFIX)
                if isinstance(query_embedding_list, list) and len(query_embedding_list) == 1 and isinstance(query_embedding_list[0], list) and len(query_embedding_list[0]) > 0:
                     query_embedding = query_embedding_list[0]; query_embedding_successful = True
                else: self.logger.error(f"[{session_id}] RAG query embed invalid structure: {query_embedding_list}.")
            except Exception as embed_e: self.logger.error(f"[{session_id}] EXCEPTION RAG query embedding: {embed_e}", exc_info=True)

            if not (query_embedding_successful and query_embedding):
                 self.logger.error(f"[{session_id}] Skipping T2 ChromaDB query: Embedding failed.")
                 return [], 0

            n_results = self.config.rag_summary_results_count
            await self._emit_status(event_emitter, session_id, f"Status: Searching vector store (top {n_results})...")
            rag_results_dict = await query_chroma_collection(
                  tier2_collection, query_embeddings=[query_embedding], n_results=n_results,
                  include=["documents", "distances", "metadatas"]
            )
            if rag_results_dict and isinstance(rag_results_dict.get("documents"), list) and rag_results_dict["documents"] and isinstance(rag_results_dict["documents"][0], list):
                  retrieved_docs = rag_results_dict["documents"][0]
                  if retrieved_docs:
                       retrieved_rag_summaries = retrieved_docs
                       t2_retrieved_count = len(retrieved_docs)
                       distances = rag_results_dict.get("distances", [[None]])[0]; ids = rag_results_dict.get("ids", [["N/A"]])[0]
                       dist_str = [f"{d:.4f}" for d in distances if d is not None]
                       self.logger.info(f"[{session_id}] Retrieved {t2_retrieved_count} docs from T2 RAG. IDs: {ids}, Dist: {dist_str}")
                  else: self.logger.info(f"[{session_id}] T2 RAG query executed but returned no documents.")
            else: self.logger.info(f"[{session_id}] T2 RAG query returned no matches or unexpected structure.")

        except Exception as e_rag_outer:
            self.logger.error(f"[{session_id}] Unexpected error during outer T2 RAG processing: {e_rag_outer}", exc_info=True)
            retrieved_rag_summaries = []
            t2_retrieved_count = 0

        return retrieved_rag_summaries, t2_retrieved_count


    async def _prepare_and_refine_background(
        self, session_id: str, body: Dict, user_valves: Any,
        retrieved_t1_summaries: List[str], retrieved_rag_summaries: List[str],
        current_active_history: List[Dict], latest_user_query_str: str,
        event_emitter: Optional[Callable]
    ) -> Tuple[str, str, int, int, bool, bool, bool, bool]:
        """ Processes system prompt, applies refinement/cache logic, combines context. """
        await self._emit_status(event_emitter, session_id, "Status: Preparing context...")

        # Process System Prompt & Extract Initial OWI Context
        base_system_prompt_text = "You are helpful."
        extracted_owi_context = None
        initial_owi_context_tokens = -1
        current_output_messages = body.get("messages", [])
        if self._process_system_prompt_func:
             try:
                 base_system_prompt_text, extracted_owi_context = self._process_system_prompt_func(current_output_messages)
                 if extracted_owi_context and self._count_tokens_func and self._tokenizer:
                      try: initial_owi_context_tokens = self._count_tokens_func(extracted_owi_context, self._tokenizer)
                      except Exception: initial_owi_context_tokens = -1
                 elif not extracted_owi_context: self.logger.debug(f"[{session_id}] No OWI <context> tag found.")
                 if not base_system_prompt_text: base_system_prompt_text = "You are helpful."; self.logger.warning(f"[{session_id}] System prompt empty after clean. Using default.")
             except Exception as e_proc_sys: self.logger.error(f"[{session_id}] Error process_system_prompt: {e_proc_sys}.", exc_info=True); base_system_prompt_text = "You are helpful."; extracted_owi_context = None
        else: self.logger.error(f"[{session_id}] process_system_prompt unavailable.")

        # Remove Specified Text Block
        session_text_block_to_remove = str(getattr(user_valves, 'text_block_to_remove', ''))
        if session_text_block_to_remove:
            self.logger.info(f"[{session_id}] Removing text block from base system prompt...")
            original_len = len(base_system_prompt_text)
            temp_prompt = base_system_prompt_text.replace(session_text_block_to_remove, "")
            if len(temp_prompt) < original_len:
                base_system_prompt_text = temp_prompt; self.logger.info(f"[{session_id}] Removed text block ({original_len - len(temp_prompt)} chars).")
            else: self.logger.warning(f"[{session_id}] Text block for removal NOT FOUND.")
        else: self.logger.debug(f"[{session_id}] No text block for removal specified.")

        # Apply session valve override for OWI processing
        session_process_owi_rag = bool(getattr(user_valves, 'process_owi_rag', True))
        if not session_process_owi_rag:
             self.logger.info(f"[{session_id}] Session valve 'process_owi_rag=False'. Discarding OWI context.")
             extracted_owi_context = None
             initial_owi_context_tokens = 0

        # Context Refinement (RAG Cache OR Stateless OR None)
        context_for_prompt = extracted_owi_context # Initialize
        refined_context_tokens = -1
        cache_update_performed = False
        cache_update_skipped = False
        final_context_selection_performed = False
        stateless_refinement_performed = False
        updated_cache_text_intermediate = "[Cache not initialized or updated]"

        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False)
        enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False)

        if enable_rag_cache_global and self._cache_update_func and self._cache_select_func and self._get_rag_cache_db_func and self.sqlite_cursor:
            self.logger.info(f"[{session_id}] RAG Cache Feature ENABLED.")
            run_step1 = False
            previous_cache_text = ""
            try:
                 cache_result = await self._get_rag_cache_db_func(self.sqlite_cursor, session_id)
                 if cache_result is not None: previous_cache_text = cache_result
            except Exception as e_get_cache: self.logger.error(f"[{session_id}] Error retrieving previous cache: {e_get_cache}", exc_info=True)

            if not session_process_owi_rag:
                 self.logger.info(f"[{session_id}] Skipping RAG Cache Step 1 (session valve 'process_owi_rag=False').")
                 cache_update_skipped = True; run_step1 = False
                 updated_cache_text_intermediate = previous_cache_text
            else:
                 skip_len = False; skip_sim = False
                 owi_content_for_check = extracted_owi_context or ""
                 len_thresh = getattr(self.config, 'CACHE_UPDATE_SKIP_OWI_THRESHOLD', 50)
                 if len(owi_content_for_check.strip()) < len_thresh: skip_len = True; self.logger.info(f"[{session_id}] Cache S1 Skip: OWI len < {len_thresh}.")
                 elif self._calculate_similarity_func and previous_cache_text:
                      sim_thresh = getattr(self.config, 'CACHE_UPDATE_SIMILARITY_THRESHOLD', 0.9)
                      try:
                          sim_score = self._calculate_similarity_func(owi_content_for_check, previous_cache_text)
                          if sim_score > sim_thresh: skip_sim = True; self.logger.info(f"[{session_id}] Cache S1 Skip: Sim > {sim_thresh:.2f}.")
                      except Exception as e_sim: self.logger.error(f"[{session_id}] Error calculating similarity: {e_sim}")
                 cache_update_skipped = skip_len or skip_sim
                 run_step1 = not cache_update_skipped
                 if cache_update_skipped:
                      await self._emit_status(event_emitter, session_id, "Status: Skipping cache update (redundant OWI).")
                      updated_cache_text_intermediate = previous_cache_text

            cache_update_llm_config = {
                "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None),
                "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                "prompt_template": getattr(self.config, 'cache_update_prompt_template', DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT),
            }
            final_select_llm_config = {
                "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None),
                "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                "prompt_template": getattr(self.config, 'final_context_selection_prompt_template', DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT),
            }

            configs_ok_step1 = all([cache_update_llm_config["url"], cache_update_llm_config["key"], cache_update_llm_config["prompt_template"]])
            configs_ok_step2 = all([final_select_llm_config["url"], final_select_llm_config["key"], final_select_llm_config["prompt_template"]])

            if not (configs_ok_step1 and configs_ok_step2):
                 self.logger.error(f"[{session_id}] Cannot proceed with RAG Cache: Refiner URL/Key/Prompts missing.")
                 await self._emit_status(event_emitter, session_id, "ERROR: RAG Cache Refiner config incomplete.", done=False)
                 updated_cache_text_intermediate = previous_cache_text; run_step1 = False
            else:
                 if run_step1:
                      await self._emit_status(event_emitter, session_id, "Status: Updating background cache...")
                      updated_cache_text_intermediate = await self._cache_update_func(
                           session_id=session_id, current_owi_context=extracted_owi_context,
                           history_messages=current_active_history, latest_user_query=latest_user_query_str,
                           llm_call_func=self._async_llm_call_wrapper, sqlite_cursor=self.sqlite_cursor,
                           cache_update_llm_config=cache_update_llm_config,
                           history_count=getattr(self.config, 'refiner_history_count', 6),
                           dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_CacheUpdate_{session_id}",
                      )
                      cache_update_performed = True

            if configs_ok_step2:
                 await self._emit_status(event_emitter, session_id, "Status: Selecting relevant context...")
                 final_selected_context = await self._cache_select_func(
                      updated_cache_text=(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else ""),
                      current_owi_context=extracted_owi_context, history_messages=current_active_history,
                      latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper,
                      context_selection_llm_config=final_select_llm_config,
                      history_count=getattr(self.config, 'refiner_history_count', 6),
                      dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_CtxSelect_{session_id}",
                 )
                 final_context_selection_performed = True
                 context_for_prompt = final_selected_context
                 log_step1_status = "Performed" if cache_update_performed else ("Skipped" if cache_update_skipped else "Not Run")
                 self.logger.info(f"[{session_id}] RAG Cache Step 2 complete. Context len: {len(context_for_prompt)}. Step 1: {log_step1_status}")
                 await self._emit_status(event_emitter, session_id, "Status: Context selection complete.", done=False)
            else:
                 self.logger.warning(f"[{session_id}] Skipping RAG Cache Step 2 (config). Using intermediate cache.")
                 context_for_prompt = updated_cache_text_intermediate

        # ELSE IF: Stateless Refinement
        elif enable_stateless_refin_global and self._stateless_refine_func:
            self.logger.info(f"[{session_id}] Stateless Refinement ENABLED.")
            await self._emit_status(event_emitter, session_id, "Status: Refining OWI context (stateless)...")
            if not extracted_owi_context: self.logger.debug(f"[{session_id}] Skipping stateless refinement: No OWI context.")
            elif not latest_user_query_str: self.logger.warning(f"[{session_id}] Skipping stateless refinement: Query empty.")
            else:
                 stateless_refiner_config = {
                     "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None),
                     "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                     "prompt_template": getattr(self.config, 'stateless_refiner_prompt_template', None),
                 }
                 if not stateless_refiner_config["url"] or not stateless_refiner_config["key"]:
                      self.logger.error(f"[{session_id}] Skipping stateless refinement: Refiner URL/Key missing.")
                      await self._emit_status(event_emitter, session_id, "ERROR: Stateless Refiner config incomplete.", done=False)
                 else:
                      try:
                          refined_stateless_context = await self._stateless_refine_func(
                               external_context=extracted_owi_context, history_messages=current_active_history,
                               latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper,
                               refiner_llm_config=stateless_refiner_config,
                               skip_threshold=getattr(self.config, 'stateless_refiner_skip_threshold', 500),
                               history_count=getattr(self.config, 'refiner_history_count', 6),
                               dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_StatelessRef_{session_id}",
                          )
                          if refined_stateless_context != extracted_owi_context:
                               context_for_prompt = refined_stateless_context; stateless_refinement_performed = True
                               self.logger.info(f"[{session_id}] Stateless refinement successful. Length: {len(context_for_prompt)}")
                               await self._emit_status(event_emitter, session_id, "Status: OWI context refined (stateless).", done=False)
                          else: self.logger.info(f"[{session_id}] Stateless refinement no change/skipped.")
                      except Exception as e_refine_stateless: self.logger.error(f"[{session_id}] EXCEPTION stateless refinement: {e_refine_stateless}", exc_info=True)

        # Calculate refined tokens
        if self._count_tokens_func and self._tokenizer:
            try:
                token_source = context_for_prompt if (final_context_selection_performed or stateless_refinement_performed) else extracted_owi_context
                if token_source: refined_context_tokens = self._count_tokens_func(token_source, self._tokenizer)
                self.logger.debug(f"[{session_id}] Refined context tokens (RefOUT): {refined_context_tokens}")
            except Exception as e_tok_ref: refined_context_tokens = -1; self.logger.error(f"[{session_id}] Error calculating refined tokens: {e_tok_ref}")
        elif not (final_context_selection_performed or stateless_refinement_performed):
             refined_context_tokens = initial_owi_context_tokens # If no refinement ran, RefOUT = OWI_IN

        # Combine Context Sources
        combined_context_string = "[No background context generated]"
        if self._combine_context_func:
            try:
                combined_context_string = self._combine_context_func(
                    final_selected_context=(context_for_prompt if isinstance(context_for_prompt, str) else None),
                    t1_summaries=retrieved_t1_summaries, t2_rag_results=retrieved_rag_summaries,
                )
            except Exception as e_combine: self.logger.error(f"[{session_id}] Error combining context: {e_combine}", exc_info=True); combined_context_string = "[Error combining context]"
        else: self.logger.error(f"[{session_id}] Cannot combine context: Function unavailable.")
        self.logger.debug(f"[{session_id}] Combined background context length: {len(combined_context_string)}.")

        # Return relevant state for the next steps
        return (
            combined_context_string,
            base_system_prompt_text,
            initial_owi_context_tokens,
            refined_context_tokens,
            cache_update_performed,
            cache_update_skipped,
            final_context_selection_performed,
            stateless_refinement_performed
        )


    async def _select_t0_history_slice(self, session_id: str, history_for_processing: List[Dict]) -> Tuple[List[Dict], int]:
        """ Selects the T0 history slice based on the last DB summary index. """
        t0_raw_history_slice = []
        t0_dialogue_tokens = -1
        db_max_index_for_t0 = -1

        try:
            db_index_result = await get_max_t1_end_index(self.sqlite_cursor, session_id)
            if isinstance(db_index_result, int) and db_index_result >= 0:
                db_max_index_for_t0 = db_index_result
            self.logger.debug(f"[{session_id}] T0 Slice: Using DB-derived index {db_max_index_for_t0} as basis.")
        except Exception as e_get_max_t0:
            self.logger.error(f"[{session_id}] T0 Slice: Error getting DB index: {e_get_max_t0}. Using -1.")
            db_max_index_for_t0 = -1

        start_idx_for_t0 = db_max_index_for_t0 + 1
        if start_idx_for_t0 < len(history_for_processing):
            history_to_consider_for_t0 = history_for_processing[start_idx_for_t0:]
            # Filter this slice for dialogue roles
            t0_raw_history_slice = [msg for msg in history_to_consider_for_t0 if isinstance(msg, dict) and msg.get("role") in self._dialogue_roles]
            self.logger.info(f"[{session_id}] T0 Slice: Selected {len(t0_raw_history_slice)} dialogue msgs (from orig index {start_idx_for_t0}).")
        else:
            self.logger.info(f"[{session_id}] T0 Slice: No relevant history range (start {start_idx_for_t0} >= hist len {len(history_for_processing)}).")
            t0_raw_history_slice = []

        # Calculate T0 tokens
        if t0_raw_history_slice and self._count_tokens_func and self._tokenizer:
            try: t0_dialogue_tokens = sum(self._count_tokens_func(msg["content"], self._tokenizer) for msg in t0_raw_history_slice if isinstance(msg, dict) and isinstance(msg.get("content"), str))
            except Exception as e_tok_t0: t0_dialogue_tokens = -1; self.logger.error(f"[{session_id}] Error calc T0 tokens: {e_tok_t0}")
        elif not t0_raw_history_slice: t0_dialogue_tokens = 0
        else: t0_dialogue_tokens = -1

        return t0_raw_history_slice, t0_dialogue_tokens


    async def _construct_final_payload(
        self, session_id: str, base_system_prompt_text: str, t0_raw_history_slice: List[Dict],
        combined_context_string: str, latest_user_query_str: str, user_valves: Any
    ) -> Optional[List[Dict]]:
        """ Constructs the final payload contents list for the LLM. """
        await self._emit_status(event_emitter=None, session_id=session_id, description="Status: Constructing final request...") # Emitter not needed here
        final_llm_payload_contents = None

        if self._construct_payload_func:
            try:
                memory_guidance = "\n\n--- Memory Guidance ---\nUse dialogue history and background info for context."
                enhanced_system_prompt = base_system_prompt_text.strip() + memory_guidance
                session_long_term_goal = str(getattr(user_valves, 'long_term_goal', ''))

                payload_dict = self._construct_payload_func(
                    system_prompt=enhanced_system_prompt, history=t0_raw_history_slice,
                    context=combined_context_string, query=latest_user_query_str,
                    long_term_goal=session_long_term_goal, strategy="standard",
                    include_ack_turns=getattr(self.config, 'include_ack_turns', True),
                )
                if isinstance(payload_dict, dict) and "contents" in payload_dict and isinstance(payload_dict["contents"], list):
                    final_llm_payload_contents = payload_dict["contents"]
                    self.logger.info(f"[{session_id}] Constructed final payload ({len(final_llm_payload_contents)} turns).")
                else:
                    self.logger.error(f"[{session_id}] Payload constructor returned invalid structure: {payload_dict}")
            except Exception as e_payload:
                self.logger.error(f"[{session_id}] EXCEPTION during payload construction: {e_payload}", exc_info=True)
        else:
            self.logger.error(f"[{session_id}] Cannot construct final payload: Function unavailable.")

        return final_llm_payload_contents


    async def _calculate_and_format_status(
        self, session_id: str, t1_retrieved_count: int, t2_retrieved_count: int,
        session_process_owi_rag: bool, final_context_selection_performed: bool,
        cache_update_skipped: bool, stateless_refinement_performed: bool,
        initial_owi_context_tokens: int, refined_context_tokens: int,
        summarization_prompt_tokens: int, summarization_output_tokens: int,
        t0_dialogue_tokens: int, final_llm_payload_contents: Optional[List[Dict]]
    ) -> str:
        """ Calculates final payload tokens and formats the status message string. """
        final_payload_tokens = -1
        if final_llm_payload_contents and self._count_tokens_func and self._tokenizer:
            try: final_payload_tokens = sum( self._count_tokens_func(part["text"], self._tokenizer) for turn in final_llm_payload_contents for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str) )
            except Exception as e_tok_final: final_payload_tokens = -1; self.logger.error(f"[{session_id}] Error calculating final payload tokens: {e_tok_final}")
        elif not final_llm_payload_contents: final_payload_tokens = 0

        # Assemble Final Status Message
        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False)
        enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False)
        refinement_status = "Refined=N"
        if enable_rag_cache_global and final_context_selection_performed: refinement_status = f"Refined=Cache(S1Skip={cache_update_skipped})"
        elif enable_stateless_refin_global and stateless_refinement_performed: refinement_status = "Refined=Stateless"
        owi_proc_status = f"OWIProc={'ON' if session_process_owi_rag else 'OFF'}"
        status_parts = [f"T1={t1_retrieved_count}", f"T2={t2_retrieved_count}", owi_proc_status, refinement_status]
        token_parts = []
        if initial_owi_context_tokens >= 0: token_parts.append(f"OWI_IN={initial_owi_context_tokens}")
        if refined_context_tokens >= 0: token_parts.append(f"RefOUT={refined_context_tokens}")
        if summarization_prompt_tokens >= 0: token_parts.append(f"SumIN={summarization_prompt_tokens}")
        if summarization_output_tokens >= 0: token_parts.append(f"SumOUT={summarization_output_tokens}")
        if t0_dialogue_tokens >= 0: token_parts.append(f"Hist={t0_dialogue_tokens}")
        if final_payload_tokens >= 0: token_parts.append(f"FinalIN={final_payload_tokens}")
        status_message = "Status: " + ", ".join(status_parts) + (" | " + " ".join(token_parts) if token_parts else "")

        return status_message, final_payload_tokens


    async def _execute_or_prepare_output(
        self, session_id: str, body: Dict, final_llm_payload_contents: Optional[List[Dict]],
        event_emitter: Optional[Callable], status_message: str, final_payload_tokens: int # Pass calculated tokens
    ) -> OrchestratorResult:
        """ Executes the final LLM call (if configured) or prepares the output body. """

        output_body = body.copy() if isinstance(body, dict) else {}

        if final_llm_payload_contents:
            output_body["messages"] = final_llm_payload_contents
            preserved_keys = ["model", "stream", "options", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"]
            keys_preserved = [k for k in preserved_keys if k in body]
            for k in keys_preserved: output_body[k] = body[k]
            self.logger.info(f"[{session_id}] Output body updated. Preserved: {keys_preserved}.")
        else:
            self.logger.error(f"[{session_id}] Final payload failed. Output body not updated.")
            # Emit final error status directly here if payload failed
            await self._emit_status(event_emitter, session_id, "ERROR: Final payload preparation failed.", done=True)
            return {"error": "Orchestrator: Final payload construction failed.", "status_code": 500}


        # Check Final LLM Trigger
        final_url = getattr(self.config, 'final_llm_api_url', None)
        final_key = getattr(self.config, 'final_llm_api_key', None)
        url_present = bool(final_url and isinstance(final_url, str) and final_url.strip())
        key_present = bool(final_key and isinstance(final_key, str) and final_key.strip())
        self.logger.debug(f"[{session_id}] Checking Final LLM Trigger. URL:{url_present}, Key:{key_present}")
        final_llm_triggered = url_present and key_present

        if final_llm_triggered:
            self.logger.info(f"[{session_id}] Final LLM Call via Pipe TRIGGERED.")
            await self._emit_status(event_emitter, session_id, "Status: Executing final LLM Call (Streaming)...", done=False)

            final_call_payload_google_fmt = {"contents": final_llm_payload_contents}
            final_response_generator = self._async_final_llm_stream_call(
                api_url=final_url, api_key=final_key, payload=final_call_payload_google_fmt,
                temperature=getattr(self.config, 'final_llm_temperature', 0.7),
                timeout=getattr(self.config, 'final_llm_timeout', 120),
                caller_info=f"Orch_FinalLLM_{session_id}",
                final_status_message=status_message, # Pass calculated status
                event_emitter=event_emitter
            )
            self.logger.info(f"[{session_id}] Returning final LLM stream generator.")
            return final_response_generator

        else:
            # Return Modified Payload Body
            self.logger.info(f"[{session_id}] Final LLM Call disabled. Passing modified payload downstream.")
            # Emit the FINAL status message here for NON-STREAMING path
            await self._emit_status(event_emitter, session_id, status_message, done=True)
            return output_body


# --- Main Processing Method (Orchestrator) ---
    async def process_turn(
        self,
        session_id: str, # Now passed directly
        user_id: str,    # Now passed directly
        body: Dict,
        user_valves: Any,
        event_emitter: Optional[Callable],
        embedding_func: Optional[Callable] = None,
        chroma_embed_wrapper: Optional[Any] = None,
        is_regeneration_heuristic: bool = False # <<< ADDED parameter to signature
    ) -> OrchestratorResult:
        """
        Processes a single turn by calling helper methods in sequence.
        Accepts regeneration flag from the caller.
        """
        pipe_entry_time_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Orchestrator process_turn [{session_id}]: Started at {pipe_entry_time_iso} (Regen Flag: {is_regeneration_heuristic})") # Log flag

        # --- Variable Initializations ---
        summarization_performed = False
        new_t1_summary_text = None
        summarization_prompt_tokens = -1
        summarization_output_tokens = -1
        t1_retrieved_count = 0
        t2_retrieved_count = 0
        retrieved_rag_summaries = []
        cache_update_performed = False
        cache_update_skipped = False
        final_context_selection_performed = False
        stateless_refinement_performed = False
        initial_owi_context_tokens = -1
        refined_context_tokens = -1
        t0_dialogue_tokens = -1
        final_payload_tokens = -1
        status_message = "Status: Processing..." # Default intermediate status

        try:
            # --- 1. Initialization & History Handling ---
            # <<< START History Sync (Moved from Pipe) >>>
            await self._emit_status(event_emitter, session_id, "Status: Orchestrator syncing history...")
            incoming_messages = body.get("messages", [])
            stored_history = self.session_manager.get_active_history(session_id) or []

            if incoming_messages != stored_history:
                if len(incoming_messages) < len(stored_history):
                    self.logger.warning(f"[{session_id}] Incoming history shorter than stored. Resetting.")
                    self.session_manager.set_active_history(session_id, incoming_messages.copy())
                    # Reset T1 index if history goes backward
                    self.session_manager.set_last_summary_index(session_id, -1)
                else:
                    self.logger.debug(f"[{session_id}] Updating active history (Len: {len(incoming_messages)}).")
                    self.session_manager.set_active_history(session_id, incoming_messages.copy())
            else:
                self.logger.debug(f"[{session_id}] Incoming history matches stored.")

            # Get the latest, potentially updated history
            current_active_history = self.session_manager.get_active_history(session_id) or []
            if not current_active_history:
                 self.logger.error(f"[{session_id}] Active history is empty after sync. Cannot proceed.")
                 await self._emit_status(event_emitter, session_id, "ERROR: History synchronization failed.", done=True)
                 return {"error": "Active history is empty.", "status_code": 500}
            # <<< END History Sync >>>

            # --- 2. Determine Effective Query ---
            # <<< Pass the received is_regeneration_heuristic flag >>>
            latest_user_query_str, history_for_processing = await self._determine_effective_query(
                session_id, current_active_history, is_regeneration_heuristic
            )
            if not latest_user_query_str and not is_regeneration_heuristic: # Check still valid here
                 self.logger.error(f"[{session_id}] Cannot proceed without an effective user query.")
                 await self._emit_status(event_emitter, session_id, "ERROR: Could not determine user query.", done=True)
                 return {"error": "Could not determine user query.", "status_code": 400}

            # --- 3. Tier 1 Summarization ---
            # <<< Pass the received is_regeneration_heuristic flag >>>
            (summarization_performed, new_t1_summary_text,
             summarization_prompt_tokens, summarization_output_tokens) = await self._handle_tier1_summarization(
                session_id, user_id, current_active_history, is_regeneration_heuristic, event_emitter
            )

            # --- 4. Tier 1 -> T2 Transition ---
            await self._handle_tier2_transition(
                session_id, summarization_performed, chroma_embed_wrapper, event_emitter
            )

            # --- 5. Prepare Context Sources (T1 & T2 RAG) ---
            recent_t1_summaries, t1_retrieved_count = await self._get_t1_summaries(session_id)
            retrieved_rag_summaries, t2_retrieved_count = await self._get_t2_rag_results(
                session_id, history_for_processing, latest_user_query_str,
                embedding_func, chroma_embed_wrapper, event_emitter
            )

            # --- 6. Prepare & Refine Background Context ---
            (combined_context_string, base_system_prompt_text,
             initial_owi_context_tokens, refined_context_tokens,
             cache_update_performed, cache_update_skipped,
             final_context_selection_performed, stateless_refinement_performed
            ) = await self._prepare_and_refine_background(
                session_id, body, user_valves, recent_t1_summaries, retrieved_rag_summaries,
                current_active_history, latest_user_query_str, event_emitter
            )

            # --- 7. Select T0 History Slice ---
            t0_raw_history_slice, t0_dialogue_tokens = await self._select_t0_history_slice(
                session_id, history_for_processing
            )

            # --- 8. Construct Final Payload ---
            final_llm_payload_contents = await self._construct_final_payload(
                session_id, base_system_prompt_text, t0_raw_history_slice,
                combined_context_string, latest_user_query_str, user_valves
            )

            # --- 9. Calculate Status Message ---
            session_process_owi_rag = bool(getattr(user_valves, 'process_owi_rag', True))
            status_message, final_payload_tokens = await self._calculate_and_format_status(
                session_id=session_id, t1_retrieved_count=t1_retrieved_count, t2_retrieved_count=t2_retrieved_count,
                session_process_owi_rag=session_process_owi_rag,
                final_context_selection_performed=final_context_selection_performed,
                cache_update_skipped=cache_update_skipped,
                stateless_refinement_performed=stateless_refinement_performed,
                initial_owi_context_tokens=initial_owi_context_tokens,
                refined_context_tokens=refined_context_tokens,
                summarization_prompt_tokens=summarization_prompt_tokens,
                summarization_output_tokens=summarization_output_tokens,
                t0_dialogue_tokens=t0_dialogue_tokens,
                final_llm_payload_contents=final_llm_payload_contents
            )
            await self._emit_status(event_emitter, session_id, status_message, done=False)


            # --- 10. Execute Final LLM or Prepare Output ---
            final_result = await self._execute_or_prepare_output(
                session_id=session_id, body=body, final_llm_payload_contents=final_llm_payload_contents,
                event_emitter=event_emitter, status_message=status_message, final_payload_tokens=final_payload_tokens
            )

            pipe_end_time_iso = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"Orchestrator process_turn [{session_id}]: Finished at {pipe_end_time_iso}")
            return final_result

        except asyncio.CancelledError:
             self.logger.info(f"[{session_id or 'unknown'}] Orchestrator process_turn cancelled.")
             await self._emit_status(event_emitter, session_id or 'unknown', "Status: Processing cancelled.", done=True)
             raise
        except Exception as e_orch:
            session_id_for_log = session_id if 'session_id' in locals() and session_id else 'unknown'
            self.logger.critical(f"[{session_id_for_log}] Orchestrator UNHANDLED EXCEPTION in process_turn: {e_orch}", exc_info=True)
            await self._emit_status(event_emitter, session_id_for_log, f"ERROR: Orchestrator Failed ({type(e_orch).__name__})", done=True)
            return {"error": f"Orchestrator failed: {type(e_orch).__name__}", "status_code": 500}

# === END OF FILE i4_llm_agent/orchestration.py ===
# [[END MODIFIED orchestration.py]]