# i4_llm_agent/orchestration.py

import logging
import asyncio
import re
import sqlite3
import uuid
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
from .api_client import call_google_llm_api # For non-streaming calls
from .utils import count_tokens, calculate_string_similarity

# --- Optional Imports ---
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False


logger = logging.getLogger(__name__) # i4_llm_agent.orchestration

# Type Alias for the complex return type
OrchestratorResult = Union[Dict, AsyncGenerator[str, None], str]

# ==============================================================================
# === Session Pipe Orchestrator Class                                        ===
# ==============================================================================

class SessionPipeOrchestrator:
    """
    Orchestrates the core processing logic of the Session Memory Pipe.
    This class encapsulates the steps previously performed directly within the
    OWI pipe script's `pipe` method.
    """

    def __init__(
        self,
        config: object, # Expects an object with attributes similar to Pipe.Valves
        session_manager: SessionManager,
        sqlite_cursor: sqlite3.Cursor,
        chroma_client: Optional[Any] = None, # Expects chromadb.ClientAPI
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initializes the SessionPipeOrchestrator.

        Args:
            config: Configuration object holding settings (like Pipe.Valves).
            session_manager: An instance of SessionManager.
            sqlite_cursor: An active SQLite database cursor.
            chroma_client: An initialized ChromaDB client instance (optional).
            logger_instance: A logger instance (optional, defaults to module logger).
        """
        self.config = config
        self.session_manager = session_manager
        self.sqlite_cursor = sqlite_cursor
        self.chroma_client = chroma_client if CHROMADB_AVAILABLE else None
        self.logger = logger_instance or logger

        self._tokenizer = None
        if TIKTOKEN_AVAILABLE and hasattr(self.config, 'tokenizer_encoding_name'):
            try:
                self.logger.info(f"Orchestrator: Initializing tokenizer '{self.config.tokenizer_encoding_name}'...")
                self._tokenizer = tiktoken.get_encoding(self.config.tokenizer_encoding_name)
                self.logger.info("Orchestrator: Tokenizer initialized.")
            except Exception as e:
                self.logger.error(f"Orchestrator: Tokenizer init failed: {e}. Token counting disabled.", exc_info=True)
        elif not TIKTOKEN_AVAILABLE:
             self.logger.warning("Orchestrator: tiktoken unavailable. Token counting disabled.")

        # Direct function aliases from library (assuming library functions handle None checks)
        self._llm_call_func = call_google_llm_api
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
        """Safely emits status updates if an emitter is provided."""
        if event_emitter and callable(event_emitter) and getattr(self.config, 'emit_status_updates', True):
            try:
                await event_emitter({
                    "type": "status",
                    "data": {"description": str(description), "done": bool(done)}
                })
            except Exception as e_emit:
                self.logger.warning(f"[{session_id}] Orchestrator failed to emit status '{description}': {e_emit}")
        else:
             self.logger.debug(f"[{session_id}] Orchestrator status update (not emitted): '{description}' (Done: {done})")

    # --- Internal Helper: Async LLM Call Wrapper ---
    # Replicates the wrapper from the pipe script for internal use
    async def _async_llm_call_wrapper(
        self,
        api_url: str,
        api_key: str,
        payload: Dict[str, Any],
        temperature: float,
        timeout: int = 90,
        caller_info: str = "Orchestrator_LLM",
    ) -> Tuple[bool, Union[str, Dict]]:
        """Async wrapper for the main LLM call function."""
        if not self._llm_call_func:
            self.logger.error(f"[{caller_info}] LLM func unavailable in orchestrator.")
            return False, {"error_type": "SetupError", "message": "LLM func unavailable"}
        # Use asyncio.to_thread to run the synchronous call_google_llm_api
        try:
             # Assuming LLM_CALL_FUNC (call_google_llm_api) is synchronous
             # If it was made async, we wouldn't need to_thread here.
             return await asyncio.to_thread(
                 self._llm_call_func,
                 api_url=api_url, api_key=api_key, payload=payload,
                 temperature=temperature, timeout=timeout, caller_info=caller_info
             )
        except Exception as e:
            self.logger.error(f"Orchestrator LLM Wrapper Error [{caller_info}]: {e}", exc_info=True)
            return False, {"error_type": "AsyncWrapperError", "message": f"{type(e).__name__}"}

    # --- Internal Helper: Final LLM Stream Call Wrapper ---
    # Replicates the streaming helper from the pipe script
    async def _async_final_llm_stream_call(
        self,
        api_url: str,
        api_key: str,
        payload: Dict[str, Any], # Expects Google 'contents' format initially
        temperature: float,
        timeout: int = 120,
        caller_info: str = "Orchestrator_FinalStream",
    ) -> AsyncGenerator[str, None]:
        """
        Calls the final LLM using httpx for streaming (OpenAI SSE format assumed).
        Handles Google->OpenAI payload conversion internally if needed.
        Yields text chunks.
        """
        log_session_id = caller_info # Use caller_info which should include session_id
        base_api_url = api_url # Initialize
        final_payload_to_send = {} # Initialize
        headers = {} # Initialize

        if not HTTPX_AVAILABLE:
            self.logger.error(f"[{log_session_id}] httpx library not available. Cannot stream.")
            yield f"[Streaming Error: httpx not installed]"
            return

        if not api_url or not api_key:
            error_msg = "Missing API Key" if not api_key else "Missing Final LLM URL"
            self.logger.error(f"[{log_session_id}] {error_msg}.")
            yield f"[Streaming Error: {error_msg}]"
            return

        # --- Determine API Type and Prepare Payload ---
        try:
            # Re-import necessary sub-component if not always available
            # Better: Ensure dependencies are handled at the library level
            from .api_client import _convert_google_to_openai_payload # Assume available
            import urllib.parse # Import standard lib

            parsed_url = urllib.parse.urlparse(api_url)
            base_api_url = urllib.parse.urlunparse(parsed_url._replace(fragment=""))
            url_for_check = base_api_url.lower().rstrip('/')

            if "openrouter.ai/api/v1/chat/completions" in url_for_check or \
               url_for_check.endswith("/v1/chat/completions"):
                if parsed_url.fragment:
                    model_name_for_conversion = parsed_url.fragment
                    self.logger.debug(f"[{log_session_id}] Final stream target is OpenAI-like. Model: '{model_name_for_conversion}'")
                    try:
                        openai_payload_converted = _convert_google_to_openai_payload(payload, model_name_for_conversion, temperature)
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
            # Add elif for Google ':streamGenerateContent' later if needed
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

        # --- Execute Streaming Call ---
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


    # --- Main Processing Method ---
    async def process_turn(
        self,
        session_id: str,
        user_id: str,
        body: Dict, # Original request body
        user_valves: Any, # UserValves object
        event_emitter: Optional[Callable],
        embedding_func: Optional[Callable] = None, # Pass the resolved OWI embedding func
        chroma_embed_wrapper: Optional[Any] = None, # Pass the embedder wrapper
    ) -> OrchestratorResult:
        """
        Processes a single turn of the conversation, applying memory management,
        RAG, context refinement, and payload construction. Mimics the core logic
        of the original Pipe.pipe method.

        Args:
            session_id: Unique session identifier.
            user_id: Identifier for the user.
            body: The original request body dictionary.
            user_valves: The resolved UserValves object for this session.
            event_emitter: Async callable for sending status updates.
            embedding_func: The resolved OWI embedding function (optional).
            chroma_embed_wrapper: The initialized ChromaDB-compatible embedder (optional).


        Returns:
            Union[Dict, AsyncGenerator[str, None], str]:
                - Dict: Modified payload if final LLM call is disabled.
                - AsyncGenerator[str, None]: Stream chunks if final LLM call is enabled & streaming.
                - str: Error message string in case of failure.
        """
        pipe_entry_time = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Orchestrator process_turn [{session_id}]: Started at {pipe_entry_time}")

        # --- 0. Initialization & Setup ---
        await self._emit_status(event_emitter, session_id, "Status: Orchestrator initializing turn...")
        output_body = body.copy() if isinstance(body, dict) else {}
        status_message = "Status: Initializing..."
        # Flags & Metrics
        summarization_performed_successfully = False
        new_t1_summary_text = None
        summarization_prompt_tokens = -1
        summarization_output_tokens = -1
        t1_retrieved_count = 0
        t2_retrieved_count = 0
        cache_update_performed = False
        cache_update_skipped = False
        final_context_selection_performed = False
        stateless_refinement_performed = False
        initial_owi_context_tokens = -1
        refined_context_tokens = -1
        t0_dialogue_tokens = -1
        final_payload_tokens = -1

        # --- Get Session State ---
        # Assumes session exists, created by the caller (Pipe) if needed
        session_state = self.session_manager.get_session(session_id)
        if not session_state:
            self.logger.error(f"[{session_id}] Orchestrator cannot proceed: Session state not found.")
            await self._emit_status(event_emitter, session_id, "ERROR: Session state missing.", done=True)
            return {"error": "Orchestrator could not find session state.", "status_code": 500} # Return Dict error

        # --- Extract User Valves from session state (set by Pipe previously) ---
        session_long_term_goal = ""
        session_process_owi_rag = True
        session_text_block_to_remove = ""
        if user_valves:
            try:
                # Using getattr for compatibility with potential non-pydantic objects
                session_long_term_goal = str(getattr(user_valves, 'long_term_goal', ''))
                session_process_owi_rag = bool(getattr(user_valves, 'process_owi_rag', True))
                session_text_block_to_remove = str(getattr(user_valves, 'text_block_to_remove', ''))
            except Exception as e_uv:
                self.logger.warning(f"[{session_id}] Error reading user valves object in orchestrator: {e_uv}. Using defaults.")

        # --- History Handling & Regeneration Detection ---
        # Uses data already processed and stored by SessionManager via Pipe
        await self._emit_status(event_emitter, session_id, "Status: Handling history...")
        current_active_history = session_state.get("active_history", [])
        previous_input_messages = session_state.get("previous_input_messages")
        incoming_messages = body.get("messages", [])

        # Check if the incoming messages are identical to the previous ones
        is_regeneration_heuristic = (
             previous_input_messages is not None and
             incoming_messages == previous_input_messages and
             len(incoming_messages) > 0
        )
        if is_regeneration_heuristic:
             self.logger.info(f"[{session_id}] Orchestrator detected identical input (regeneration heuristic).")
             await self._emit_status(event_emitter, session_id, "Status: Regeneration detected...")

        # Determine effective user message and history slice for *this* processing cycle
        effective_user_message = None
        history_for_processing = []
        latest_user_query_str = ""
        effective_user_message_index = -1

        user_message_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") == "user"]

        if not user_message_indices:
            self.logger.error(f"[{session_id}] Orchestrator: No user messages found in history. Cannot proceed.")
            await self._emit_status(event_emitter, session_id, "ERROR: No user messages found.", done=True)
            return {"error": "Orchestrator cannot process request without user messages.", "status_code": 400}

        if is_regeneration_heuristic:
            if len(user_message_indices) >= 2:
                effective_user_message_index = user_message_indices[-2]
                self.logger.info(f"[{session_id}] Orchestrator Regen: Using user message at index {effective_user_message_index} as effective query.")
            else:
                effective_user_message_index = user_message_indices[-1]
                self.logger.warning(f"[{session_id}] Orchestrator Regen: Only one user message found. Using last user message at index {effective_user_message_index}.")
        else:
            effective_user_message_index = user_message_indices[-1]
            self.logger.debug(f"[{session_id}] Orchestrator Normal: Using last user message at index {effective_user_message_index} as query.")

        if effective_user_message_index >= 0:
            effective_user_message = current_active_history[effective_user_message_index]
            history_for_processing = current_active_history[:effective_user_message_index] # History *before* the effective query
            latest_user_query_str = effective_user_message.get("content", "")
            self.logger.debug(f"[{session_id}] Orchestrator effective query set (len: {len(latest_user_query_str)}). History slice len: {len(history_for_processing)}.")
        else:
             self.logger.error(f"[{session_id}] Orchestrator failed to determine effective user message index. Aborting.")
             await self._emit_status(event_emitter, session_id, "ERROR: Failed to identify query message.", done=True)
             return {"error": "Orchestrator internal error identifying query message.", "status_code": 500}

        # --- Core Component Checks ---
        if not self._tokenizer or not self._count_tokens_func:
            self.logger.error(f"[{session_id}] Orchestrator: Tokenizer/counter unavailable.")
            # Don't necessarily fail here, but log it
        if not self.sqlite_cursor:
             self.logger.error(f"[{session_id}] Orchestrator: SQLite cursor unavailable. DB operations will fail.")
             # Allow proceeding but expect failures later

        # --- 1. Tier 1 Summarization Check & Execution ---
        await self._emit_status(event_emitter, session_id, "Status: Checking summarization...")
        # Check prerequisites based on orchestrator's state/config
        can_summarize = all([
            self._manage_memory_func, self._tokenizer, self._count_tokens_func,
            self.sqlite_cursor, self._async_llm_call_wrapper,
            hasattr(self.config, 'summarizer_api_url') and self.config.summarizer_api_url,
            hasattr(self.config, 'summarizer_api_key') and self.config.summarizer_api_key,
            current_active_history,
        ])
        if can_summarize:
             summarizer_llm_config = {
                 "url": self.config.summarizer_api_url,
                 "key": self.config.summarizer_api_key,
                 "temp": getattr(self.config, 'summarizer_temperature', 0.5),
                 "sys_prompt": getattr(self.config, 'summarizer_system_prompt', "Summarize this dialogue."),
             }
             try:
                 # Directly update session state via SessionManager if needed, or handle index update here
                 # manage_tier1_summarization might need adjustment if it directly modified session_state dict before
                 # Let's assume manage_tier1_summarization returns the new index, and we update the manager here.
                 (
                     summarization_performed,
                     generated_summary,
                     new_last_summary_idx, # Get the potentially updated index
                     prompt_tokens,
                     t0_end_idx,
                 ) = await self._manage_memory_func(
                     # Pass necessary state components, not the whole dict
                     session_state={ # Reconstruct needed state for the function
                         "last_summary_turn_index": self.session_manager.get_last_summary_index(session_id)
                     },
                     active_history=current_active_history,
                     t0_token_limit=getattr(self.config, 't0_active_history_token_limit', 4000),
                     t1_chunk_size_target=getattr(self.config, 't1_summarization_chunk_token_target', 2000),
                     tokenizer=self._tokenizer,
                     llm_call_func=self._async_llm_call_wrapper, # Use internal wrapper
                     llm_config=summarizer_llm_config,
                     add_t1_summary_func=lambda sid, sess_id, uid, txt, meta: add_tier1_summary(self.sqlite_cursor, sid, sess_id, uid, txt, meta), # Wrap DB call
                     session_id=session_id,
                     user_id=user_id,
                     dialogue_only_roles=self._dialogue_roles,
                 )
                 if summarization_performed:
                     summarization_performed_successfully = True
                     new_t1_summary_text = generated_summary
                     summarization_prompt_tokens = prompt_tokens
                     # Update session state via manager *after* successful summarization and save
                     self.session_manager.set_last_summary_index(session_id, new_last_summary_idx)
                     if new_t1_summary_text and self._count_tokens_func and self._tokenizer:
                          try: summarization_output_tokens = self._count_tokens_func(new_t1_summary_text, self._tokenizer)
                          except Exception: summarization_output_tokens = -1
                     self.logger.info(f"[{session_id}] Orchestrator: T1 summary generated/saved. NewIdx: {new_last_summary_idx}. SumIN: {prompt_tokens}. SumOUT: {summarization_output_tokens}. TrigIdx: {t0_end_idx}.")
                     await self._emit_status(event_emitter, session_id, "Status: Summary generated.", done=False)
                 else:
                     self.logger.debug(f"[{session_id}] Orchestrator: T1 criteria not met.")
             except Exception as e_manage:
                 self.logger.error(f"[{session_id}] Orchestrator EXCEPTION during T1 manage call: {e_manage}", exc_info=True)
                 summarization_performed_successfully = False
        else:
             self.logger.warning(f"[{session_id}] Orchestrator: Skipping T1 check: Missing prerequisites.")


        # --- 2. Tier 1 -> T2 Transition Check ---
        await self._emit_status(event_emitter, session_id, "Status: Checking long-term memory capacity...")
        tier2_collection = None
        if self.chroma_client and chroma_embed_wrapper:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_')
            safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]
            tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
            # Get collection using the DB function
            tier2_collection = await get_or_create_chroma_collection(
                 self.chroma_client, tier2_collection_name, chroma_embed_wrapper
            )

        can_transition = all([
            summarization_performed_successfully,
            tier2_collection is not None,
            chroma_embed_wrapper is not None, # Passed in
            self.sqlite_cursor is not None,
            getattr(self.config, 'max_stored_summary_blocks', 0) > 0
        ])

        if can_transition:
            max_t1_blocks = self.config.max_stored_summary_blocks
            current_tier1_count = await get_tier1_summary_count(self.sqlite_cursor, session_id)

            if current_tier1_count == -1:
                self.logger.error(f"[{session_id}] Orchestrator: Failed get T1 count. Skipping T1->T2 check.")
            elif current_tier1_count > max_t1_blocks:
                self.logger.info(f"[{session_id}] Orchestrator: T1 limit ({max_t1_blocks}) exceeded ({current_tier1_count}). Transitioning...")
                await self._emit_status(event_emitter, session_id, "Status: Archiving oldest summary...")
                oldest_summary_data = await get_oldest_tier1_summary(self.sqlite_cursor, session_id)

                if oldest_summary_data:
                    oldest_id, oldest_text, oldest_metadata = oldest_summary_data
                    embedding_vector = None; embedding_successful = False
                    try:
                        # Embed using the passed-in wrapper
                        embedding_list = await asyncio.to_thread(chroma_embed_wrapper, [oldest_text])
                        # Validate (assuming wrapper returns list of lists)
                        if isinstance(embedding_list, list) and len(embedding_list) == 1 and isinstance(embedding_list[0], list) and len(embedding_list[0]) > 0:
                             embedding_vector = embedding_list[0]
                             embedding_successful = True
                        else: self.logger.error(f"[{session_id}] Orchestrator: Embedding T1 {oldest_id} returned invalid structure: {embedding_list}")
                    except Exception as embed_e: self.logger.error(f"[{session_id}] Orchestrator: EXCEPTION embedding T1->T2 {oldest_id}: {embed_e}", exc_info=True)

                    if embedding_successful and embedding_vector:
                        chroma_metadata = oldest_metadata.copy()
                        chroma_metadata["transitioned_from_t1"] = True
                        chroma_metadata["original_t1_id"] = oldest_id
                        sanitized_chroma_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in chroma_metadata.items() if v is not None}
                        tier2_id = f"t2_{oldest_id}"
                        self.logger.info(f"[{session_id}] Orchestrator: Adding summary {tier2_id} to T2 collection '{tier2_collection.name}'...")
                        # Add using DB function
                        added_to_t2 = await add_to_chroma_collection(
                             tier2_collection, ids=[tier2_id], embeddings=[embedding_vector],
                             metadatas=[sanitized_chroma_metadata], documents=[oldest_text]
                        )
                        if added_to_t2:
                             self.logger.info(f"[{session_id}] Orchestrator: Added {tier2_id} to T2. Deleting original T1...")
                             # Delete using DB function
                             deleted_from_t1 = await delete_tier1_summary(self.sqlite_cursor, oldest_id)
                             if deleted_from_t1: await self._emit_status(event_emitter, session_id, "Status: Summary archive complete.", done=False)
                             else: self.logger.warning(f"[{session_id}] Orchestrator: Added {tier2_id} to T2, but FAILED delete T1 {oldest_id}.")
                        # Error handling for add_to_chroma_collection happens within the function
                    else: self.logger.error(f"[{session_id}] Orchestrator: Skipping T2 add for {oldest_id}: embedding failed.")
                else: self.logger.warning(f"[{session_id}] Orchestrator: T1 count exceeded limit, but couldn't retrieve oldest summary.")
            else: self.logger.debug(f"[{session_id}] Orchestrator: T1 count ({current_tier1_count}) within limit. No transition needed.")
        else: self.logger.debug(f"[{session_id}] Orchestrator: Skipping T1->T2 transition: Prerequisites not met.")


        # --- 3. Tier 2 RAG Lookup ---
        await self._emit_status(event_emitter, session_id, "Status: Searching long-term memory...")
        retrieved_rag_summaries: List[str] = []
        t2_retrieved_count = 0
        can_rag = all([
            tier2_collection is not None, # Already retrieved or None above
            effective_user_message is not None,
            embedding_func is not None, # Passed in
            self._generate_rag_query_func is not None,
            self._async_llm_call_wrapper is not None,
            getattr(self.config, 'ragq_llm_api_url', None),
            getattr(self.config, 'ragq_llm_api_key', None),
            getattr(self.config, 'ragq_llm_prompt', None),
            getattr(self.config, 'rag_summary_results_count', 0) > 0,
        ])

        if not can_rag: self.logger.info(f"[{session_id}] Orchestrator: Skipping T2 RAG: Prerequisites not met.")
        else:
            # Check if collection has documents
            t2_doc_count = await get_chroma_collection_count(tier2_collection)
            if t2_doc_count <= 0:
                 self.logger.info(f"[{session_id}] Orchestrator: Skipping T2 RAG: Collection '{tier2_collection.name}' is empty or count failed ({t2_doc_count}).")
                 can_rag = False

        if can_rag:
            rag_query = None; query_embedding = None; query_embedding_successful = False
            try:
                 await self._emit_status(event_emitter, session_id, "Status: Generating search query...")
                 context_messages_for_ragq = self._get_recent_turns_func(
                     current_active_history, count=6, exclude_last=True, roles=self._dialogue_roles
                 )
                 dialogue_context_str = self._format_history_func(context_messages_for_ragq) if context_messages_for_ragq else "[No recent history]"
                 # latest_user_query_str already derived
                 ragq_llm_config = {
                     "url": self.config.ragq_llm_api_url, "key": self.config.ragq_llm_api_key,
                     "temp": getattr(self.config, 'ragq_llm_temperature', 0.3),
                     "prompt": self.config.ragq_llm_prompt,
                 }
                 rag_query_result = await self._generate_rag_query_func(
                     latest_message_str=latest_user_query_str,
                     dialogue_context_str=dialogue_context_str,
                     llm_call_func=self._async_llm_call_wrapper,
                     llm_config=ragq_llm_config,
                     caller_info=f"Orch_RAGQ_{session_id}",
                 )
                 if rag_query_result and isinstance(rag_query_result, str) and not rag_query_result.startswith("[Error:") and rag_query_result.strip():
                     rag_query = rag_query_result.strip()
                 else: self.logger.error(f"[{session_id}] Orchestrator: RAG Query Generation failed: {rag_query_result}.")

                 if rag_query and embedding_func:
                     await self._emit_status(event_emitter, session_id, "Status: Embedding search query...")
                     try:
                         # Use OWI func directly with query prefix
                         # Assumes embedding_func handles user context if needed
                         from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX # Need this constant
                         query_embedding_list = await asyncio.to_thread(embedding_func, [rag_query], prefix=RAG_EMBEDDING_QUERY_PREFIX) # User passed implicitly if wrapper used
                         if isinstance(query_embedding_list, list) and len(query_embedding_list) == 1 and isinstance(query_embedding_list[0], list) and len(query_embedding_list[0]) > 0:
                              query_embedding = query_embedding_list[0]; query_embedding_successful = True
                         else: self.logger.error(f"[{session_id}] Orchestrator: RAG query embedding returned invalid structure: {query_embedding_list}.")
                     except Exception as embed_e: self.logger.error(f"[{session_id}] Orchestrator: EXCEPTION during RAG query embedding: {embed_e}", exc_info=True)

                 if query_embedding_successful and query_embedding:
                     n_results = self.config.rag_summary_results_count
                     await self._emit_status(event_emitter, session_id, f"Status: Searching vector store (top {n_results})...")
                     # Query using DB function
                     rag_results_dict = await query_chroma_collection(
                          tier2_collection, query_embeddings=[query_embedding], n_results=n_results,
                          include=["documents", "distances", "metadatas"] # Ensure include param exists
                     )
                     if rag_results_dict and isinstance(rag_results_dict.get("documents"), list) and rag_results_dict["documents"] and isinstance(rag_results_dict["documents"][0], list):
                          retrieved_docs = rag_results_dict["documents"][0]
                          if retrieved_docs:
                               retrieved_rag_summaries = retrieved_docs
                               t2_retrieved_count = len(retrieved_docs)
                               distances = rag_results_dict.get("distances", [[None]])[0]; ids = rag_results_dict.get("ids", [["N/A"]])[0]
                               dist_str = [f"{d:.4f}" for d in distances if d is not None]
                               self.logger.info(f"[{session_id}] Orchestrator: Retrieved {t2_retrieved_count} docs from T2 RAG. IDs: {ids}, Distances: {dist_str}")
                          else: self.logger.info(f"[{session_id}] Orchestrator: T2 RAG query executed but returned no documents.")
                     else: self.logger.info(f"[{session_id}] Orchestrator: T2 RAG query returned no matches or unexpected structure.")
                 elif rag_query: self.logger.error(f"[{session_id}] Orchestrator: Skipping T2 ChromaDB query because query embedding failed.")
            except Exception as e_rag_outer: self.logger.error(f"[{session_id}] Orchestrator: Unexpected error during outer T2 RAG processing block: {e_rag_outer}", exc_info=True)


        # --- 4. Prepare Final Payload Inputs & Context Refinement ---
        await self._emit_status(event_emitter, session_id, "Status: Preparing context...")
        # --- 4a: Retrieve T1 Summaries ---
        recent_t1_summaries = []
        t1_retrieved_count = 0
        if self.sqlite_cursor and getattr(self.config, 'max_stored_summary_blocks', 0) > 0:
             try:
                 recent_t1_summaries = await get_recent_tier1_summaries(self.sqlite_cursor, session_id, self.config.max_stored_summary_blocks)
                 t1_retrieved_count = len(recent_t1_summaries)
             except Exception as e_get_t1: self.logger.error(f"[{session_id}] Orchestrator: Error retrieving T1: {e_get_t1}", exc_info=True)
        if t1_retrieved_count > 0: self.logger.info(f"[{session_id}] Orchestrator: Retrieved {t1_retrieved_count} T1 summaries.")

        # --- 4b: Process System Prompt & Extract Initial OWI Context ---
        base_system_prompt_text = "You are helpful."
        extracted_owi_context = None
        initial_owi_context_tokens = -1
        current_output_messages = output_body.get("messages", []) # Use messages from input body copy
        if self._process_system_prompt_func:
             try:
                 base_system_prompt_text, extracted_owi_context = self._process_system_prompt_func(current_output_messages)
                 if extracted_owi_context and self._count_tokens_func and self._tokenizer:
                      try: initial_owi_context_tokens = self._count_tokens_func(extracted_owi_context, self._tokenizer)
                      except Exception: initial_owi_context_tokens = -1
                 elif not extracted_owi_context: self.logger.debug(f"[{session_id}] Orchestrator: No OWI <context> tag found.")
                 if not base_system_prompt_text: base_system_prompt_text = "You are helpful."; self.logger.warning(f"[{session_id}] Orchestrator: System prompt empty after clean. Using default.")
             except Exception as e_proc_sys: self.logger.error(f"[{session_id}] Orchestrator: Error process_system_prompt: {e_proc_sys}.", exc_info=True); base_system_prompt_text = "You are helpful."; extracted_owi_context = None
        else: self.logger.error(f"[{session_id}] Orchestrator: process_system_prompt unavailable.")

        # --- 4b.1 Remove Specified Text Block ---
        if session_text_block_to_remove:
            self.logger.info(f"[{session_id}] Orchestrator: Attempting to remove text block from base system prompt...")
            original_len = len(base_system_prompt_text)
            temp_prompt = base_system_prompt_text.replace(session_text_block_to_remove, "")
            if len(temp_prompt) < original_len:
                base_system_prompt_text = temp_prompt
                self.logger.info(f"[{session_id}] Orchestrator: Successfully removed text block ({original_len - len(temp_prompt)} chars).")
            else: self.logger.warning(f"[{session_id}] Orchestrator: Specified text block for removal NOT FOUND.")
        else: self.logger.debug(f"[{session_id}] Orchestrator: No text block for removal specified.")

        # --- 4c.1: Apply session valve override for OWI processing ---
        if not session_process_owi_rag:
             self.logger.info(f"[{session_id}] Orchestrator: Session valve 'process_owi_rag=False'. Discarding OWI context.")
             extracted_owi_context = None
             initial_owi_context_tokens = 0

        # --- 4c.2: Context Refinement (RAG Cache OR Stateless OR None) ---
        context_for_prompt = extracted_owi_context
        refined_context_tokens = -1
        cache_update_performed = False; cache_update_skipped = False
        final_context_selection_performed = False; stateless_refinement_performed = False
        updated_cache_text_intermediate = "[Cache not initialized or updated]" # For step 2 input

        # Check Global RAG Cache Enable Valve
        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False)
        enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False)

        if enable_rag_cache_global and self._cache_update_func and self._cache_select_func and self._get_rag_cache_db_func and self.sqlite_cursor:
            self.logger.info(f"[{session_id}] Orchestrator: Global RAG Cache Feature ENABLED. Checking steps...")
            run_step1 = False
            previous_cache_text: Optional[str] = None

            # Retrieve previous cache first
            try:
                 previous_cache_text = await self._get_rag_cache_db_func(self.sqlite_cursor, session_id)
                 if previous_cache_text is None: previous_cache_text = "" # Treat as empty
            except Exception as e_get_cache:
                 self.logger.error(f"[{session_id}] Orchestrator: Error retrieving previous cache: {e_get_cache}", exc_info=True)
                 previous_cache_text = "" # Fallback

            # Check if Step 1 should run (considering session valve FIRST)
            if not session_process_owi_rag:
                 self.logger.info(f"[{session_id}] Orchestrator: Skipping RAG Cache Step 1 (session valve 'process_owi_rag=False').")
                 cache_update_skipped = True; run_step1 = False
                 updated_cache_text_intermediate = previous_cache_text # Use previous cache for step 2
            else:
                 # Session allows OWI processing, check length/similarity
                 skip_len = False; skip_sim = False
                 owi_content_for_check = extracted_owi_context or ""
                 owi_len = len(owi_content_for_check.strip())
                 len_thresh = getattr(self.config, 'CACHE_UPDATE_SKIP_OWI_THRESHOLD', 50)
                 if owi_len < len_thresh: skip_len = True; self.logger.info(f"[{session_id}] Orchestrator: Cache Step 1 Skip: OWI length ({owi_len}) < threshold ({len_thresh}).")
                 elif self._calculate_similarity_func and previous_cache_text:
                      sim_thresh = getattr(self.config, 'CACHE_UPDATE_SIMILARITY_THRESHOLD', 0.9)
                      try:
                          sim_score = self._calculate_similarity_func(owi_content_for_check, previous_cache_text)
                          self.logger.debug(f"[{session_id}] Orchestrator: OWI vs Cache Similarity Score: {sim_score:.4f}")
                          if sim_score > sim_thresh: skip_sim = True; self.logger.info(f"[{session_id}] Orchestrator: Cache Step 1 Skip: Similarity ({sim_score:.4f}) > threshold ({sim_thresh}).")
                      except Exception as e_sim: self.logger.error(f"[{session_id}] Orchestrator: Error calculating similarity: {e_sim}")
                 # Determine if step 1 runs based on skips
                 cache_update_skipped = skip_len or skip_sim
                 run_step1 = not cache_update_skipped
                 if cache_update_skipped:
                      await self._emit_status(event_emitter, session_id, "Status: Skipping cache update (redundant OWI).")
                      updated_cache_text_intermediate = previous_cache_text # Use previous cache

            # --- Prepare LLM Configs for Cache Steps ---
            cache_update_llm_config = {
                "url": getattr(self.config, 'refiner_llm_api_url', None),
                "key": getattr(self.config, 'refiner_llm_api_key', None),
                "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                "prompt_template": getattr(self.config, 'cache_update_prompt_template', DEFAULT_CACHE_UPDATE_PROMPT_TEXT),
            }
            final_select_llm_config = {
                "url": getattr(self.config, 'refiner_llm_api_url', None),
                "key": getattr(self.config, 'refiner_llm_api_key', None),
                "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                "prompt_template": getattr(self.config, 'final_context_selection_prompt_template', DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT),
            }

            # Check configs
            if not all([cache_update_llm_config["url"], cache_update_llm_config["key"],
                        final_select_llm_config["url"], final_select_llm_config["key"],
                        cache_update_llm_config["prompt_template"], final_select_llm_config["prompt_template"]]):
                 self.logger.error(f"[{session_id}] Orchestrator: Cannot proceed with RAG Cache: Refiner URL/Key/Prompts missing in config.")
                 await self._emit_status(event_emitter, session_id, "ERROR: RAG Cache Refiner config incomplete.", done=False)
                 updated_cache_text_intermediate = previous_cache_text # Fallback
                 run_step1 = False # Prevent step 1 execution
            else:
                 # --- Execute Step 1 (if not skipped) ---
                 if run_step1:
                      await self._emit_status(event_emitter, session_id, "Status: Updating background cache...")
                      updated_cache_text_intermediate = await self._cache_update_func(
                           session_id=session_id,
                           current_owi_context=extracted_owi_context,
                           history_messages=current_active_history,
                           latest_user_query=latest_user_query_str,
                           llm_call_func=self._async_llm_call_wrapper,
                           sqlite_cursor=self.sqlite_cursor,
                           cache_update_llm_config=cache_update_llm_config,
                           history_count=getattr(self.config, 'refiner_history_count', 6),
                           dialogue_only_roles=self._dialogue_roles,
                           caller_info=f"Orch_CacheUpdate_{session_id}",
                      )
                      cache_update_performed = True

            # --- Execute Step 2: Select Final Context (Always run if Step 1 setup was valid) ---
            if all([final_select_llm_config["url"], final_select_llm_config["key"], final_select_llm_config["prompt_template"]]):
                 await self._emit_status(event_emitter, session_id, "Status: Selecting relevant context...")
                 final_selected_context = await self._cache_select_func(
                      updated_cache_text=(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else ""),
                      current_owi_context=extracted_owi_context,
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
                 self.logger.info(f"[{session_id}] Orchestrator: RAG Cache Step 2 complete. Using selected context (len: {len(context_for_prompt)}). Step 1: {log_step1_status}")
                 await self._emit_status(event_emitter, session_id, "Status: Context selection complete.", done=False)
            else:
                 # If step 2 config failed, use the intermediate text (either updated or previous cache)
                 self.logger.warning(f"[{session_id}] Orchestrator: Skipping RAG Cache Step 2 due to config issues. Using intermediate cache text.")
                 context_for_prompt = updated_cache_text_intermediate


        # --- ELSE IF: Stateless Refinement ---
        elif enable_stateless_refin_global and self._stateless_refine_func:
            self.logger.info(f"[{session_id}] Orchestrator: Stateless Refinement ENABLED globally (RAG Cache disabled/unavailable).")
            await self._emit_status(event_emitter, session_id, "Status: Refining OWI context (stateless)...")
            if not extracted_owi_context: self.logger.debug(f"[{session_id}] Orchestrator: Skipping stateless refinement: No OWI context.")
            elif not latest_user_query_str: self.logger.warning(f"[{session_id}] Orchestrator: Skipping stateless refinement: Query empty.")
            else:
                 stateless_refiner_config = {
                     "url": getattr(self.config, 'refiner_llm_api_url', None),
                     "key": getattr(self.config, 'refiner_llm_api_key', None),
                     "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                     "prompt_template": getattr(self.config, 'stateless_refiner_prompt_template', None), # Can be None
                 }
                 if not stateless_refiner_config["url"] or not stateless_refiner_config["key"]:
                      self.logger.error(f"[{session_id}] Orchestrator: Skipping stateless refinement: Refiner URL/Key missing.")
                      await self._emit_status(event_emitter, session_id, "ERROR: Stateless Refiner config incomplete.", done=False)
                 else:
                      try:
                          refined_stateless_context = await self._stateless_refine_func(
                               external_context=extracted_owi_context,
                               history_messages=current_active_history,
                               latest_user_query=latest_user_query_str,
                               llm_call_func=self._async_llm_call_wrapper,
                               refiner_llm_config=stateless_refiner_config,
                               skip_threshold=getattr(self.config, 'stateless_refiner_skip_threshold', 500),
                               history_count=getattr(self.config, 'refiner_history_count', 6),
                               dialogue_only_roles=self._dialogue_roles,
                               caller_info=f"Orch_StatelessRef_{session_id}",
                          )
                          if refined_stateless_context != extracted_owi_context:
                               context_for_prompt = refined_stateless_context; stateless_refinement_performed = True
                               self.logger.info(f"[{session_id}] Orchestrator: Stateless refinement successful. Length: {len(context_for_prompt)}")
                               await self._emit_status(event_emitter, session_id, "Status: OWI context refined (stateless).", done=False)
                          else: self.logger.info(f"[{session_id}] Orchestrator: Stateless refinement no change/skipped.")
                      except Exception as e_refine_stateless: self.logger.error(f"[{session_id}] Orchestrator: EXCEPTION stateless refinement: {e_refine_stateless}", exc_info=True)

        # Calculate refined tokens
        if self._count_tokens_func and self._tokenizer:
            try:
                # Use initial tokens if no refinement happened, otherwise count the final context_for_prompt
                token_source = context_for_prompt if (final_context_selection_performed or stateless_refinement_performed) else extracted_owi_context
                if token_source:
                    refined_context_tokens = self._count_tokens_func(token_source, self._tokenizer)
                    self.logger.debug(f"[{session_id}] Orchestrator: Refined context tokens (RefOUT): {refined_context_tokens}")
            except Exception as e_tok_ref: refined_context_tokens = -1; self.logger.error(f"[{session_id}] Error calculating refined tokens: {e_tok_ref}")
        elif not (final_context_selection_performed or stateless_refinement_performed):
             refined_context_tokens = initial_owi_context_tokens # Use initial if no refinement and tokenizer unavailable

        # --- 4d: Select T0 Dialogue History Slice ---
        # Simple slice of history *before* the effective query, after last summary
        t0_raw_history_slice = []
        last_summary_idx = self.session_manager.get_last_summary_index(session_id)
        start_index_for_t0_selection = last_summary_idx + 1
        end_index_for_t0_selection = len(history_for_processing) # Index before the effective query

        if end_index_for_t0_selection > start_index_for_t0_selection:
             history_to_consider_for_t0 = history_for_processing[start_index_for_t0_selection:end_index_for_t0_selection]
             # Filter for dialogue roles
             t0_raw_history_slice = [msg for msg in history_to_consider_for_t0 if isinstance(msg, dict) and msg.get("role") in self._dialogue_roles]
             self.logger.info(f"[{session_id}] Orchestrator: Selected T0 history slice: {len(t0_raw_history_slice)} dialogue msgs (from original index {start_index_for_t0_selection} up to before effective query).")
        else: self.logger.info(f"[{session_id}] Orchestrator: No relevant history range found for T0 slice.")

        # Calculate T0 tokens
        if t0_raw_history_slice and self._count_tokens_func and self._tokenizer:
            try: t0_dialogue_tokens = sum(self._count_tokens_func(msg["content"], self._tokenizer) for msg in t0_raw_history_slice if isinstance(msg, dict) and isinstance(msg.get("content"), str))
            except Exception as e_tok_t0: t0_dialogue_tokens = -1; self.logger.error(f"[{session_id}] Error calc T0 tokens: {e_tok_t0}")
        elif not t0_raw_history_slice: t0_dialogue_tokens = 0
        else: t0_dialogue_tokens = -1


        # --- 4e: Combine Context Sources ---
        combined_context_string = "[Error combining context]"
        if self._combine_context_func:
            try:
                combined_context_string = self._combine_context_func(
                    final_selected_context=(context_for_prompt if isinstance(context_for_prompt, str) else None),
                    t1_summaries=recent_t1_summaries,
                    t2_rag_results=retrieved_rag_summaries,
                )
            except Exception as e_combine: self.logger.error(f"[{session_id}] Orchestrator: Error combining context: {e_combine}", exc_info=True)
        else: self.logger.error(f"[{session_id}] Orchestrator: Cannot combine context: Function unavailable.")
        self.logger.debug(f"[{session_id}] Orchestrator: Combined background context length: {len(combined_context_string)}.")


        # --- 4f: Prepare Final System Prompt ---
        memory_guidance = "\n\n--- Memory Guidance ---\nUse dialogue history and background info for context."
        enhanced_system_prompt = base_system_prompt_text.strip() + memory_guidance


        # --- 5. Construct Final LLM Payload ---
        await self._emit_status(event_emitter, session_id, "Status: Constructing final request...")
        final_llm_payload_contents: Optional[List[Dict]] = None
        final_payload_tokens: int = -1
        if self._construct_payload_func:
            try:
                payload_dict = self._construct_payload_func(
                    system_prompt=enhanced_system_prompt,
                    history=t0_raw_history_slice,
                    context=combined_context_string,
                    query=latest_user_query_str,
                    long_term_goal=session_long_term_goal, # Use goal resolved earlier
                    strategy="standard", # Or make configurable
                    include_ack_turns=getattr(self.config, 'include_ack_turns', True),
                )
                if isinstance(payload_dict, dict) and "contents" in payload_dict and isinstance(payload_dict["contents"], list):
                    final_llm_payload_contents = payload_dict["contents"]
                    self.logger.info(f"[{session_id}] Orchestrator: Constructed final payload ({len(final_llm_payload_contents)} turns).")
                    if self._count_tokens_func and self._tokenizer and final_llm_payload_contents:
                        try: # Simplified token counting loop
                            total_tok = sum( self._count_tokens_func(part["text"], self._tokenizer) for turn in final_llm_payload_contents for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str) )
                            final_payload_tokens = total_tok
                            self.logger.debug(f"[{session_id}] Orchestrator: Calculated final payload tokens (FinalIN): {final_payload_tokens}")
                        except Exception as e_tok_final: final_payload_tokens = -1; self.logger.error(f"[{session_id}] Error calculating final payload tokens: {e_tok_final}")
                    elif not final_llm_payload_contents: final_payload_tokens = 0
                    else: final_payload_tokens = -1; self.logger.warning(f"[{session_id}] Skipping final token calculation: Tokenizer unavailable.")
                else: self.logger.error(f"[{session_id}] Orchestrator: Payload constructor returned invalid structure: {payload_dict}")
            except Exception as e_payload: self.logger.error(f"[{session_id}] Orchestrator: EXCEPTION during payload construction: {e_payload}", exc_info=True)
        else: self.logger.error(f"[{session_id}] Orchestrator: Cannot construct final payload: Function unavailable.")


        # --- 6. Update Output Body (Prepare for return/final call) ---
        if final_llm_payload_contents:
            output_body["messages"] = final_llm_payload_contents
            preserved_keys = ["model", "stream", "options", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"]
            keys_preserved = [k for k in preserved_keys if k in body]
            for k in keys_preserved: output_body[k] = body[k]
            self.logger.info(f"[{session_id}] Orchestrator: Output body updated. Preserved: {keys_preserved}.")
        else: self.logger.error(f"[{session_id}] Orchestrator: Final payload failed. Output body not updated.")


        # --- 7. Assemble Final Status Message ---
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
        status_message = "Status: " + ", ".join(status_parts)
        if token_parts: status_message += " | " + " ".join(token_parts)
        await self._emit_status(event_emitter, session_id, status_message, done=False) # Emit intermediate status


        # --- 8. Optional Final LLM Call ---
        final_llm_triggered = bool(
            getattr(self.config, 'final_llm_api_url', None) and
            getattr(self.config, 'final_llm_api_key', None)
        )

        if final_llm_triggered:
            self.logger.info(f"[{session_id}] Orchestrator: Final LLM Call via Pipe TRIGGERED (Streaming).")
            await self._emit_status(event_emitter, session_id, "Status: Executing final LLM Call (Streaming)...")
            if not final_llm_payload_contents:
                self.logger.error(f"[{session_id}] Orchestrator: Cannot execute Final LLM Call: Payload construction failed.")
                await self._emit_status(event_emitter, session_id, "ERROR: Final payload preparation failed.", done=True)
                return "Apologies, error preparing final request." # Return error string
            else:
                # Prepare the payload dict (still Google format before stream wrapper converts it)
                final_call_payload_google_fmt = {"contents": final_llm_payload_contents}
                # Call the streaming helper
                final_response_generator = self._async_final_llm_stream_call(
                    api_url=self.config.final_llm_api_url,
                    api_key=self.config.final_llm_api_key,
                    payload=final_call_payload_google_fmt,
                    temperature=getattr(self.config, 'final_llm_temperature', 0.7),
                    timeout=getattr(self.config, 'final_llm_timeout', 120),
                    caller_info=f"Orch_FinalLLM_{session_id}",
                )
                # Return the generator
                self.logger.info(f"[{session_id}] Orchestrator: Returning final LLM stream generator.")
                await self._emit_status(event_emitter, session_id, "Status: Streaming final response...", done=True)
                self.logger.info(f"Orchestrator process_turn [{session_id}]: Finished (Returning Stream Generator).")
                return final_response_generator # Return the async generator

        else:
            # --- 9. Return Modified Payload Body (Default Action) ---
            self.logger.info(f"[{session_id}] Orchestrator: Final LLM Call disabled. Passing modified payload downstream.")
            await self._emit_status(event_emitter, session_id, status_message, done=True) # Emit final status
            self.logger.info(f"Orchestrator process_turn [{session_id}]: Finished (Returning Payload Dict).")
            return output_body # Return Dict
