# --- START OF FILE orchestration.py ---

# [[START MODIFIED orchestration.py - FULL]]
# i4_llm_agent/orchestration.py

import logging
import asyncio
import re
import sqlite3
import json
import uuid
from .utils import TIKTOKEN_AVAILABLE # Import directly from utils
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
    # Add inventory DB functions
    get_all_inventories_for_session,
    get_character_inventory_data, # Needed by inventory module potentially
    add_or_update_character_inventory, # Needed by inventory module
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
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT, # Now includes inventory placeholder
    # Add inventory prompt formatter
    format_inventory_update_prompt, DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
)
from .cache import ( # Import orchestrators from cache.py
    update_rag_cache, select_final_context
)
# Import the correct dispatcher function
from .api_client import call_google_llm_api # Use the dispatcher

from .utils import count_tokens, calculate_string_similarity

# --- Inventory Module Import ---
# Import using a try-except block to handle potential unavailability
try:
    from .inventory import (
        format_inventory_for_prompt,
        update_inventories_from_llm,
        INVENTORY_MODULE_AVAILABLE
    )
except ImportError:
    INVENTORY_MODULE_AVAILABLE = False
    # Define dummy functions if module not available
    def format_inventory_for_prompt(*args, **kwargs): return "[Inventory Module Unavailable]"
    async def update_inventories_from_llm(*args, **kwargs): return False
    logging.getLogger(__name__).warning("Inventory module not found. Inventory features disabled.")


# --- Optional Imports ---
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False
    # Logging handled by Pipe startup

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
    Includes inventory management features.
    """

    def __init__(
        self,
        config: object, # Expects an object with attributes similar to Pipe.Valves
        session_manager: SessionManager,
        sqlite_cursor: sqlite3.Cursor,
        chroma_client: Optional[Any] = None, # Expects chromadb.ClientAPI
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
        self._get_all_inventories_db_func = get_all_inventories_for_session # <<< NEW Inventory DB Alias
        self._get_char_inventory_db_func = get_character_inventory_data # <<< NEW Inventory DB Alias
        self._update_char_inventory_db_func = add_or_update_character_inventory # <<< NEW Inventory DB Alias
        # Inventory Module Functions
        # Use dummy functions if module not available
        self._format_inventory_func = format_inventory_for_prompt if INVENTORY_MODULE_AVAILABLE else lambda *args, **kwargs: "[Inventory Module Unavailable]"
        self._update_inventories_func = update_inventories_from_llm if INVENTORY_MODULE_AVAILABLE else lambda *args, **kwargs: asyncio.sleep(0, result=False)


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
                # Check if emitter is async or sync (though Pipe usually provides async)
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
        """
        Wraps the library's LLM call function for error handling.
        Handles both awaitable coroutines and direct tuple returns for early errors.
        """
        if not self._llm_call_func:
            self.logger.error(f"[{caller_info}] LLM func unavailable in orchestrator.")
            return False, {"error_type": "SetupError", "message": "LLM func unavailable"}
        try:
            # Call the underlying function (which might be sync or async internally)
            result = self._llm_call_func(
                api_url=api_url, api_key=api_key, payload=payload,
                temperature=temperature, timeout=timeout, caller_info=caller_info
            )

            # Check if the result is awaitable (a coroutine)
            if asyncio.iscoroutine(result):
                # If it's awaitable, await it
                self.logger.debug(f"[{caller_info}] Awaiting result from LLM function.")
                return await result
            elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
                # If it's already a tuple (likely an early sync error return), return it directly
                self.logger.debug(f"[{caller_info}] LLM function returned tuple directly (Success: {result[0]}).")
                return result
            else:
                # Unexpected return type from the underlying function
                self.logger.error(f"[{caller_info}] LLM function returned unexpected type: {type(result)}. Result: {result}")
                return False, {"error_type": "InternalError", "message": f"LLM function returned unexpected type {type(result)}"}

        except Exception as e:
            # Catch exceptions during the call itself or during await
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
        """Handles streaming calls to OpenAI-compatible endpoints."""
        # Extract session ID for logging if present in caller_info
        log_session_id = caller_info
        session_id_match = re.search(r'_(user_.*|chat_.*)', log_session_id) # Adjusted regex
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

            # Check for known streaming endpoints
            if "openrouter.ai/api/v1/chat/completions" in url_for_check or \
               url_for_check.endswith("/v1/chat/completions"): # Common OpenAI pattern
                if parsed_url.fragment:
                    model_name_for_conversion = parsed_url.fragment
                    self.logger.debug(f"[{log_session_id}] Final stream target is OpenAI-like. Model: '{model_name_for_conversion}'")
                    try:
                        openai_payload_converted = converter_func(payload, model_name_for_conversion, temperature)
                        openai_payload_converted["stream"] = True
                        # Add other potential streaming params if needed (e.g., based on original body)
                        # Example: if 'max_tokens' in original_body: openai_payload_converted['max_tokens'] = original_body['max_tokens']
                        final_payload_to_send = openai_payload_converted
                    except Exception as e_conv:
                        self.logger.error(f"[{log_session_id}] Error converting payload for OpenAI stream: {e_conv}", exc_info=True)
                        yield f"[Streaming Error: Payload conversion failed - {type(e_conv).__name__}]"
                        return

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/event-stream"
                        # Add site URL / App name if needed for providers like OpenRouter
                        # "HTTP-Referer": ..., "X-Title": ...
                    }
                else:
                    self.logger.error(f"[{log_session_id}] OpenAI-like URL requires model fragment (e.g., #model/name) for streaming.")
                    yield "[Streaming Error: Model fragment missing in URL]"
                    return
            # Add other potential streaming API types here if needed (e.g., Anthropic)
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
        # Log payload *before* call for debugging stream issues
        self.logger.debug(f"[{log_session_id}] Streaming Payload: {json.dumps(final_payload_to_send)}")
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client: # Slightly longer timeout for client
                 async with client.stream("POST", base_api_url, headers=headers, json=final_payload_to_send, timeout=timeout) as response:
                    # Log headers right after request potentially
                    # self.logger.debug(f"[{log_session_id}] Stream Request Headers: {response.request.headers}")
                    if response.status_code != 200:
                         error_body = await response.aread() # Read error body
                         error_text = error_body.decode('utf-8', errors='replace')[:1000] # Decode safely
                         self.logger.error(f"[{log_session_id}] Stream API Error: Status {response.status_code}. Response: {error_text}...")
                         yield f"[Streaming Error: API returned status {response.status_code}]"
                         # Try to parse common error structure
                         try:
                             err_json = json.loads(error_text)
                             if "error" in err_json and isinstance(err_json["error"], dict):
                                 err_detail = err_json["error"].get("message", "Unknown API error")
                                 yield f"\n[Detail: {err_detail}]"
                         except json.JSONDecodeError: pass # Ignore if error body wasn't JSON
                         return # Stop generation

                    self.logger.debug(f"[{log_session_id}] Stream connection successful (Status {response.status_code}). Reading chunks...")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_content = line[len("data:"):].strip()
                            if data_content == "[DONE]":
                                self.logger.debug(f"[{log_session_id}] Received [DONE] signal.")
                                stream_successful = True
                                break # Exit the loop cleanly
                            if data_content:
                                try:
                                    chunk = json.loads(data_content)
                                    # Common OpenAI structure: choices[0].delta.content
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    text_chunk = delta.get("content")
                                    if text_chunk: # Ensure content exists and is not empty/null
                                        yield text_chunk
                                    # Log other delta info if needed (e.g., role, finish_reason)
                                    # finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                                    # if finish_reason: logger.debug(f"[{log_session_id}] Stream finish reason: {finish_reason}")
                                except json.JSONDecodeError:
                                    self.logger.warning(f"[{log_session_id}] Failed to decode JSON data chunk: '{data_content}'")
                                except (IndexError, KeyError, TypeError) as e_parse:
                                    self.logger.warning(f"[{log_session_id}] Error parsing stream chunk structure ({type(e_parse).__name__}): {chunk}")
                        elif line.strip(): # Log non-empty, non-data lines if needed
                             self.logger.debug(f"[{log_session_id}] Received non-data line: '{line}'")

        except httpx.TimeoutException as e_timeout:
            self.logger.error(f"[{log_session_id}] HTTPX Timeout during stream after {timeout}s: {e_timeout}", exc_info=False)
            yield f"[Streaming Error: Timeout after {timeout}s]"
        except httpx.RequestError as e_req:
            # Handles connection errors, DNS errors etc.
            self.logger.error(f"[{log_session_id}] HTTPX RequestError during stream: {e_req}", exc_info=True)
            yield f"[Streaming Error: Network request failed - {type(e_req).__name__}]"
        except asyncio.CancelledError:
             self.logger.info(f"[{log_session_id}] Final stream call explicitly cancelled.")
             yield "[Streaming Error: Cancelled]"
             raise # Propagate cancellation
        except Exception as e_stream:
            self.logger.error(f"[{log_session_id}] Unexpected error during final LLM stream: {e_stream}", exc_info=True)
            yield f"[Streaming Error: Unexpected error - {type(e_stream).__name__}]"
        finally:
             # Ensure final status is emitted regardless of how the stream ended
             self.logger.info(f"[{log_session_id}] Final LLM stream processing finished.")
             status_to_emit = final_status_message
             if not stream_successful:
                  status_to_emit += " (Stream Interrupted or Failed)"
             await self._emit_status(
                 event_emitter=event_emitter,
                 session_id=extracted_session_id, # Use extracted ID
                 description=status_to_emit,
                 done=True # Mark as done since streaming finished/failed
             )

    # --- Helper Methods for process_turn ---

    async def _determine_effective_query(
        self, session_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool
    ) -> Tuple[str, List[Dict]]:
        """ Determines the effective user query and the history slice preceding it. """
        effective_user_message_index = -1
        # Ensure messages are dicts with 'role' before checking
        user_message_indices = [i for i, msg in enumerate(current_active_history) if isinstance(msg, dict) and msg.get("role") == "user"]

        if not user_message_indices:
            self.logger.error(f"[{session_id}] No user messages found in history.")
            return "", [] # Return empty defaults, caller must handle

        # Determine index based on regeneration
        if is_regeneration_heuristic:
            # Use second-to-last user msg if available, else the last (shouldn't happen if regen needs >1 turn)
            effective_user_message_index = user_message_indices[-2] if len(user_message_indices) >= 2 else user_message_indices[-1]
            log_level = self.logger.info if len(user_message_indices) >= 2 else self.logger.warning
            log_level(f"[{session_id}] Regen: Using user message at index {effective_user_message_index} as query base.")
        else:
            # Use the very last user message
            effective_user_message_index = user_message_indices[-1]
            self.logger.debug(f"[{session_id}] Normal: Using user message at index {effective_user_message_index} as query base.")

        # Ensure index is valid before slicing
        if effective_user_message_index < 0 or effective_user_message_index >= len(current_active_history):
             self.logger.error(f"[{session_id}] Calculated effective user index {effective_user_message_index} is out of bounds.")
             return "", [] # Return empty defaults

        effective_user_message = current_active_history[effective_user_message_index]
        # History for processing includes everything *before* the effective user message
        history_for_processing = current_active_history[:effective_user_message_index]
        latest_user_query_str = effective_user_message.get("content", "") if isinstance(effective_user_message, dict) else ""

        self.logger.debug(f"[{session_id}] Effective query set (len: {len(latest_user_query_str)}). History slice for processing len: {len(history_for_processing)}.")
        return latest_user_query_str, history_for_processing


    async def _handle_tier1_summarization(
        self, session_id: str, user_id: str, current_active_history: List[Dict], is_regeneration_heuristic: bool, event_emitter: Optional[Callable]
    ) -> Tuple[bool, Optional[str], int, int]:
        """ Checks and performs T1 summarization, skipping LLM call if regenerating an existing block. """
        await self._emit_status(event_emitter, session_id, "Status: Checking summarization...")

        summarization_performed_successfully = False
        generated_summary = None
        summarization_prompt_tokens = -1
        summarization_output_tokens = -1

        # Check prerequisites
        can_summarize = all([
            self._manage_memory_func, self._tokenizer, self._count_tokens_func,
            self.sqlite_cursor, self._async_llm_call_wrapper,
            hasattr(self.config, 'summarizer_api_url') and self.config.summarizer_api_url,
            hasattr(self.config, 'summarizer_api_key') and self.config.summarizer_api_key,
            current_active_history, # Ensure history is not empty
        ])

        if not can_summarize:
             missing_prereqs = [p for p, v in {
                 "manage_func": self._manage_memory_func, "tokenizer": self._tokenizer,
                 "count_func": self._count_tokens_func, "db_cursor": self.sqlite_cursor,
                 "llm_wrapper": self._async_llm_call_wrapper,
                 "summ_url": getattr(self.config, 'summarizer_api_url', None),
                 "summ_key": getattr(self.config, 'summarizer_api_key', None),
                 "history": bool(current_active_history)
             }.items() if not v]
             self.logger.warning(f"[{session_id}] Skipping T1 check: Missing prerequisites: {', '.join(missing_prereqs)}.")
             return False, None, -1, -1 # Return defaults

        # Prepare config and call memory manager
        summarizer_llm_config = {
             "url": self.config.summarizer_api_url, "key": self.config.summarizer_api_key,
             "temp": getattr(self.config, 'summarizer_temperature', 0.5),
             "sys_prompt": getattr(self.config, 'summarizer_system_prompt', "Summarize this dialogue."),
        }
        new_last_summary_idx = -1
        prompt_tokens = -1
        t0_end_idx = -1
        db_max_index = None
        current_last_summary_index_for_memory = -1

        # Get start index from DB
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

        # Define nested save function
        async def _async_save_t1_summary(summary_id: str, session_id: str, user_id: str, summary_text: str, metadata: Dict) -> bool:
             """Nested async function to save T1 summary via library function."""
             try:
                 return await add_tier1_summary(cursor=self.sqlite_cursor, summary_id=summary_id, session_id=session_id, user_id=user_id, summary_text=summary_text, metadata=metadata)
             except Exception as e_save:
                 self.logger.error(f"[{session_id}] Exception in nested _async_save_t1_summary for {summary_id}: {e_save}", exc_info=True)
                 return False

        # Call the main memory management function
        try:
            self.logger.debug(f"[{session_id}] Calling manage_tier1_summarization with start index = {current_last_summary_index_for_memory} (Regen={is_regeneration_heuristic})")
            summarization_performed, generated_summary_text, new_last_summary_idx, prompt_tokens, t0_end_idx = await self._manage_memory_func(
                current_last_summary_index=current_last_summary_index_for_memory,
                active_history=current_active_history,
                t0_token_limit=getattr(self.config, 't0_active_history_token_limit', 4000),
                t1_chunk_size_target=getattr(self.config, 't1_summarization_chunk_token_target', 2000),
                tokenizer=self._tokenizer,
                llm_call_func=self._async_llm_call_wrapper,
                llm_config=summarizer_llm_config,
                add_t1_summary_func=_async_save_t1_summary, # Pass nested save func
                session_id=session_id, user_id=user_id,
                cursor=self.sqlite_cursor, # Pass cursor
                is_regeneration=is_regeneration_heuristic, # Pass flag
                dialogue_only_roles=self._dialogue_roles,
            )

            if summarization_performed:
                summarization_performed_successfully = True
                generated_summary = generated_summary_text
                summarization_prompt_tokens = prompt_tokens
                # Calculate output tokens
                if generated_summary and self._count_tokens_func and self._tokenizer:
                    try: summarization_output_tokens = self._count_tokens_func(generated_summary, self._tokenizer)
                    except Exception: summarization_output_tokens = -1
                self.logger.info(f"[{session_id}] T1 summary generated/saved. New Idx: {new_last_summary_idx}, PromptTok: {summarization_prompt_tokens}, OutTok: {summarization_output_tokens}.")
                await self._emit_status(event_emitter, session_id, "Status: Summary generated.", done=False)
            else:
                # Log covers skipped due to regen check or criteria not met
                self.logger.debug(f"[{session_id}] T1 summarization skipped or criteria not met (Returned Index: {new_last_summary_idx}).")
        except TypeError as e_type:
             self.logger.error(f"[{session_id}] Orchestrator TYPE ERROR calling T1 manage func: {e_type}. Signature mismatch?", exc_info=True)
        except Exception as e_manage:
            self.logger.error(f"[{session_id}] Orchestrator EXCEPTION during T1 manage call: {e_manage}", exc_info=True)

        # Return calculated values
        return summarization_performed_successfully, generated_summary, summarization_prompt_tokens, summarization_output_tokens


    async def _handle_tier2_transition(
        self, session_id: str, t1_success: bool, chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> None:
        """ Handles the transition of the oldest T1 summary to T2 if needed. """
        await self._emit_status(event_emitter, session_id, "Status: Checking long-term memory capacity...")
        tier2_collection = None

        # Check prerequisites for T2 transition
        inventory_enabled = getattr(self.config, 'enable_inventory_management', False) # Check if inventory active
        max_t1_blocks = getattr(self.config, 'max_stored_summary_blocks', 0)

        can_transition = all([
            t1_success, # Only run if a T1 summary was actually generated this turn
            self.chroma_client is not None,
            chroma_embed_wrapper is not None,
            self.sqlite_cursor is not None,
            max_t1_blocks > 0
        ])

        if not can_transition:
            reason = "(T1 did not run)" if not t1_success else "(Prerequisites not met)"
            self.logger.debug(f"[{session_id}] Skipping T1->T2 transition check: {reason}.")
            return

        # Get T2 collection instance
        try:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_')
            safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]
            tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
            tier2_collection = await get_or_create_chroma_collection(
                 self.chroma_client, tier2_collection_name, chroma_embed_wrapper
            )
            if not tier2_collection:
                 self.logger.error(f"[{session_id}] Failed to get/create T2 collection '{tier2_collection_name}'. Skipping transition.")
                 return
        except Exception as e_get_coll:
             self.logger.error(f"[{session_id}] Error getting T2 collection: {e_get_coll}. Skipping transition.", exc_info=True)
             return

        # Perform transition logic
        try:
            current_tier1_count = await get_tier1_summary_count(self.sqlite_cursor, session_id)

            if current_tier1_count == -1:
                self.logger.error(f"[{session_id}] Failed get T1 count. Skipping T1->T2 check.")
            elif current_tier1_count > max_t1_blocks:
                self.logger.info(f"[{session_id}] T1 limit ({max_t1_blocks}) exceeded ({current_tier1_count}). Transitioning oldest...")
                await self._emit_status(event_emitter, session_id, "Status: Archiving oldest summary...")

                oldest_summary_data = await get_oldest_tier1_summary(self.sqlite_cursor, session_id)
                if not oldest_summary_data:
                    self.logger.warning(f"[{session_id}] T1 count exceeded limit, but couldn't retrieve oldest summary.")
                    return

                oldest_id, oldest_text, oldest_metadata = oldest_summary_data
                embedding_vector = None; embedding_successful = False

                # Embed the oldest summary text
                try:
                    # Ensure wrapper call runs in a thread if it's synchronous internally
                    embedding_list = await asyncio.to_thread(chroma_embed_wrapper, [oldest_text])
                    if isinstance(embedding_list, list) and len(embedding_list) == 1 and isinstance(embedding_list[0], list) and len(embedding_list[0]) > 0:
                         embedding_vector = embedding_list[0]; embedding_successful = True
                    else: self.logger.error(f"[{session_id}] T1->T2 Embed: Invalid structure returned by wrapper: {type(embedding_list)}")
                except Exception as embed_e:
                    self.logger.error(f"[{session_id}] EXCEPTION embedding T1->T2 summary {oldest_id}: {embed_e}", exc_info=True)

                # Add to T2 and delete from T1 if embedding was successful
                if embedding_successful and embedding_vector:
                    added_to_t2 = False; deleted_from_t1 = False
                    chroma_metadata = oldest_metadata.copy() # Avoid modifying original dict
                    chroma_metadata["transitioned_from_t1"] = True
                    chroma_metadata["original_t1_id"] = oldest_id
                    # Sanitize metadata for ChromaDB (only basic types allowed)
                    sanitized_chroma_metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in chroma_metadata.items() if v is not None}
                    tier2_id = f"t2_{oldest_id}"

                    self.logger.info(f"[{session_id}] Adding summary {tier2_id} to T2 collection '{tier2_collection.name}'...")
                    added_to_t2 = await add_to_chroma_collection(
                         tier2_collection, ids=[tier2_id], embeddings=[embedding_vector],
                         metadatas=[sanitized_chroma_metadata], documents=[oldest_text]
                    )

                    if added_to_t2:
                         self.logger.info(f"[{session_id}] Added {tier2_id} to T2. Deleting T1 summary {oldest_id}...")
                         deleted_from_t1 = await delete_tier1_summary(self.sqlite_cursor, oldest_id)
                         if deleted_from_t1:
                             self.logger.info(f"[{session_id}] Successfully deleted T1 summary {oldest_id}.")
                             await self._emit_status(event_emitter, session_id, "Status: Summary archive complete.", done=False)
                         else:
                             # Critical issue if T1 deletion fails after T2 add
                             self.logger.critical(f"[{session_id}] Added {tier2_id} to T2, but FAILED TO DELETE T1 {oldest_id}. Potential data duplication!")
                    else:
                        self.logger.error(f"[{session_id}] Failed to add summary {tier2_id} to T2 collection.")
                else:
                    self.logger.error(f"[{session_id}] Skipping T2 addition for T1 summary {oldest_id}: Embedding failed.")
            else:
                self.logger.debug(f"[{session_id}] T1 count ({current_tier1_count}) within limit ({max_t1_blocks}). No transition needed.")
        except Exception as e_t2_trans:
            self.logger.error(f"[{session_id}] Unexpected error during T1->T2 transition: {e_t2_trans}", exc_info=True)


    async def _get_t1_summaries(self, session_id: str) -> Tuple[List[str], int]:
        """ Fetches recent T1 summaries from the database. """
        recent_t1_summaries = []
        t1_retrieved_count = 0
        max_blocks_t1 = getattr(self.config, 'max_stored_summary_blocks', 0)

        if self.sqlite_cursor and max_blocks_t1 > 0:
             try:
                 # Use max_blocks as the limit for retrieval
                 recent_t1_summaries = await get_recent_tier1_summaries(self.sqlite_cursor, session_id, max_blocks_t1)
                 t1_retrieved_count = len(recent_t1_summaries)
             except Exception as e_get_t1:
                 self.logger.error(f"[{session_id}] Error retrieving T1 summaries: {e_get_t1}", exc_info=True)
                 recent_t1_summaries = []; t1_retrieved_count = 0 # Reset on error
        elif not self.sqlite_cursor:
             self.logger.warning(f"[{session_id}] Cannot get T1 summaries: SQLite cursor unavailable.")
        elif max_blocks_t1 <= 0:
             self.logger.debug(f"[{session_id}] Skipping T1 summary retrieval: max_stored_summary_blocks is {max_blocks_t1}.")

        if t1_retrieved_count > 0: self.logger.info(f"[{session_id}] Retrieved {t1_retrieved_count} T1 summaries for context.")
        return recent_t1_summaries, t1_retrieved_count


    async def _get_t2_rag_results(
        self, session_id: str, history_for_processing: List[Dict], latest_user_query_str: str,
        embedding_func: Optional[Callable], chroma_embed_wrapper: Optional[Any], event_emitter: Optional[Callable]
    ) -> Tuple[List[str], int]:
        """ Performs T2 RAG lookup based on a generated query. """
        await self._emit_status(event_emitter, session_id, "Status: Searching long-term memory...")
        retrieved_rag_summaries = []
        t2_retrieved_count = 0
        tier2_collection = None
        n_results_t2 = getattr(self.config, 'rag_summary_results_count', 0)

        # Check prerequisites for T2 RAG
        can_rag = all([
            self.chroma_client is not None,
            chroma_embed_wrapper is not None, # Wrapper needed to get collection
            latest_user_query_str, # Need query to generate RAGQ
            embedding_func is not None, # Need OWI embedding func for query
            self._generate_rag_query_func is not None,
            self._async_llm_call_wrapper is not None,
            getattr(self.config, 'ragq_llm_api_url', None),
            getattr(self.config, 'ragq_llm_api_key', None),
            getattr(self.config, 'ragq_llm_prompt', None),
            n_results_t2 > 0 # Need to retrieve at least 1 result
        ])

        if not can_rag:
            self.logger.info(f"[{session_id}] Skipping T2 RAG check: Prerequisites not met (RAG Results Count: {n_results_t2}).")
            return [], 0

        # Get T2 Collection
        try:
            base_prefix = getattr(self.config, 'summary_collection_prefix', 'sm_t2_')
            safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]
            tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
            tier2_collection = await get_or_create_chroma_collection(
                 self.chroma_client, tier2_collection_name, chroma_embed_wrapper
            )
            if not tier2_collection:
                 self.logger.error(f"[{session_id}] Failed get/create T2 collection '{tier2_collection_name}'. Skipping RAG.")
                 return [], 0
        except Exception as e_get_coll_rag:
             self.logger.error(f"[{session_id}] Error getting T2 collection for RAG: {e_get_coll_rag}. Skipping.", exc_info=True)
             return [], 0

        # Check if collection is empty
        try:
            t2_doc_count = await get_chroma_collection_count(tier2_collection)
            if t2_doc_count <= 0:
                self.logger.info(f"[{session_id}] Skipping T2 RAG: Collection '{tier2_collection.name}' is empty or count failed ({t2_doc_count}).")
                return [], 0
        except Exception as e_count:
             self.logger.error(f"[{session_id}] Error checking T2 collection count: {e_count}. Skipping RAG.", exc_info=True)
             return [], 0

        # Proceed with query generation and execution
        try:
            await self._emit_status(event_emitter, session_id, "Status: Generating search query...")
            # Get dialogue context for the RAG query generator LLM
            context_messages_for_ragq = self._get_recent_turns_func(
                 history_for_processing, # Use history BEFORE the current query
                 count=getattr(self.config, 'refiner_history_count', 6), # Use same history count?
                 exclude_last=False, # Include all of this slice
                 roles=self._dialogue_roles
            )
            dialogue_context_str = self._format_history_func(context_messages_for_ragq) if context_messages_for_ragq else "[No recent history]"

            # Prepare config for RAG Query LLM
            ragq_llm_config = {
                 "url": self.config.ragq_llm_api_url, "key": self.config.ragq_llm_api_key,
                 "temp": getattr(self.config, 'ragq_llm_temperature', 0.3),
                 "prompt": self.config.ragq_llm_prompt,
            }
            rag_query = await self._generate_rag_query_func(
                 latest_message_str=latest_user_query_str, # The user's current effective query
                 dialogue_context_str=dialogue_context_str,
                 llm_call_func=self._async_llm_call_wrapper,
                 llm_config=ragq_llm_config,
                 caller_info=f"Orch_RAGQ_{session_id}",
            )

            if not (rag_query and isinstance(rag_query, str) and not rag_query.startswith("[Error:") and rag_query.strip()):
                 self.logger.error(f"[{session_id}] RAG Query Generation failed or returned empty: Result='{rag_query}'. Skipping RAG.")
                 return [], 0
            self.logger.info(f"[{session_id}] Generated RAG Query: '{rag_query[:100]}...'")

            # Embed the generated query using the OWI function
            await self._emit_status(event_emitter, session_id, "Status: Embedding search query...")
            query_embedding = None; query_embedding_successful = False
            try:
                # Ensure embedding_func exists and is callable
                if not callable(embedding_func):
                     self.logger.error(f"[{session_id}] Cannot embed RAG query: OWI Embedding function unavailable/invalid.")
                     return [], 0

                # Use the global import if available
                from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX
                # Embed in a thread as OWI func might be sync
                query_embedding_list = await asyncio.to_thread(embedding_func, [rag_query], prefix=RAG_EMBEDDING_QUERY_PREFIX)
                if isinstance(query_embedding_list, list) and len(query_embedding_list) == 1 and isinstance(query_embedding_list[0], list) and len(query_embedding_list[0]) > 0:
                     query_embedding = query_embedding_list[0]; query_embedding_successful = True
                     self.logger.debug(f"[{session_id}] RAG query embedding successful (vector dim: {len(query_embedding)}).")
                else:
                     self.logger.error(f"[{session_id}] RAG query embed invalid structure returned: {type(query_embedding_list)}.")
            except Exception as embed_e:
                 self.logger.error(f"[{session_id}] EXCEPTION during RAG query embedding: {embed_e}", exc_info=True)

            if not (query_embedding_successful and query_embedding):
                 self.logger.error(f"[{session_id}] Skipping T2 ChromaDB query: RAG query embedding failed.")
                 return [], 0

            # Query ChromaDB
            await self._emit_status(event_emitter, session_id, f"Status: Searching vector store (top {n_results_t2})...")
            rag_results_dict = await query_chroma_collection(
                  tier2_collection, query_embeddings=[query_embedding], n_results=n_results_t2,
                  include=["documents", "distances", "metadatas"] # Request necessary fields
            )

            # Process results
            if rag_results_dict and isinstance(rag_results_dict.get("documents"), list) and rag_results_dict["documents"] and isinstance(rag_results_dict["documents"][0], list):
                  retrieved_docs = rag_results_dict["documents"][0]
                  if retrieved_docs:
                       retrieved_rag_summaries = retrieved_docs
                       t2_retrieved_count = len(retrieved_docs)
                       # Log distances and IDs for debugging relevance
                       distances = rag_results_dict.get("distances", [[None]])[0]; ids = rag_results_dict.get("ids", [["N/A"]])[0]
                       dist_str = [f"{d:.4f}" for d in distances if d is not None]
                       self.logger.info(f"[{session_id}] Retrieved {t2_retrieved_count} docs from T2 RAG. IDs: {ids}, Dist: {dist_str}")
                  else:
                       self.logger.info(f"[{session_id}] T2 RAG query executed but returned no documents.")
            else:
                 self.logger.info(f"[{session_id}] T2 RAG query returned no matches or unexpected structure: {type(rag_results_dict)}")

        except Exception as e_rag_outer:
            self.logger.error(f"[{session_id}] Unexpected error during outer T2 RAG processing: {e_rag_outer}", exc_info=True)
            retrieved_rag_summaries = []
            t2_retrieved_count = 0 # Ensure reset on error

        return retrieved_rag_summaries, t2_retrieved_count


    async def _prepare_and_refine_background(
        self, session_id: str, body: Dict, user_valves: Any,
        retrieved_t1_summaries: List[str], retrieved_rag_summaries: List[str],
        current_active_history: List[Dict], latest_user_query_str: str,
        event_emitter: Optional[Callable]
    ) -> Tuple[str, str, int, int, bool, bool, bool, bool, str]: # Added formatted_inventory_string to return
        """
        Processes system prompt, applies refinement/cache logic, combines context (now including inventory).
        MODIFIED to fetch, format, and include inventory data if enabled.
        """
        await self._emit_status(event_emitter, session_id, "Status: Preparing context...")

        # --- 1. Process System Prompt & Extract OWI Context ---
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
        else: self.logger.error(f"[{session_id}] process_system_prompt unavailable."); base_system_prompt_text = "You are helpful."

        # --- 2. Remove Text Block from System Prompt ---
        session_text_block_to_remove = str(getattr(user_valves, 'text_block_to_remove', ''))
        if session_text_block_to_remove:
            self.logger.info(f"[{session_id}] Removing text block from base system prompt...")
            original_len = len(base_system_prompt_text)
            temp_prompt = base_system_prompt_text.replace(session_text_block_to_remove, "")
            if len(temp_prompt) < original_len:
                base_system_prompt_text = temp_prompt; self.logger.info(f"[{session_id}] Removed text block ({original_len - len(temp_prompt)} chars).")
            else: self.logger.warning(f"[{session_id}] Text block for removal '{session_text_block_to_remove[:50]}...' NOT FOUND.")
        else: self.logger.debug(f"[{session_id}] No text block for removal specified.")

        # --- 3. Apply OWI RAG Processing Valve ---
        session_process_owi_rag = bool(getattr(user_valves, 'process_owi_rag', True))
        if not session_process_owi_rag:
             self.logger.info(f"[{session_id}] Session valve 'process_owi_rag=False'. Discarding OWI context.")
             extracted_owi_context = None
             initial_owi_context_tokens = 0

        # --- 4. Fetch and Format Inventory Data (Conditional) ---
        formatted_inventory_string = "[Inventory Management Disabled]"
        raw_session_inventories = {}
        inventory_enabled = getattr(self.config, 'enable_inventory_management', False)

        if inventory_enabled and self._get_all_inventories_db_func and self._format_inventory_func and self.sqlite_cursor:
            self.logger.debug(f"[{session_id}] Inventory enabled, fetching data...")
            try:
                raw_session_inventories = await self._get_all_inventories_db_func(self.sqlite_cursor, session_id)
                if raw_session_inventories:
                    self.logger.info(f"[{session_id}] Retrieved inventory data for {len(raw_session_inventories)} characters.")
                    try:
                        formatted_inventory_string = self._format_inventory_func(raw_session_inventories)
                        self.logger.info(f"[{session_id}] Formatted inventory string generated (len: {len(formatted_inventory_string)}).")
                    except Exception as e_fmt_inv:
                        self.logger.error(f"[{session_id}] Failed to format inventory string: {e_fmt_inv}", exc_info=True)
                        formatted_inventory_string = "[Error Formatting Inventory]"
                else:
                    self.logger.info(f"[{session_id}] No inventory data found in DB for this session.")
                    formatted_inventory_string = "[No Inventory Data Available]"
            except Exception as e_get_inv:
                self.logger.error(f"[{session_id}] Error retrieving inventory data from DB: {e_get_inv}", exc_info=True)
                formatted_inventory_string = "[Error Retrieving Inventory]"
        elif not inventory_enabled:
            self.logger.debug(f"[{session_id}] Skipping inventory fetch: Feature disabled by valve.")
        else:
            missing_inv_funcs = [f for f, fn in {"db_get": self._get_all_inventories_db_func, "formatter": self._format_inventory_func, "cursor": self.sqlite_cursor}.items() if not fn]
            self.logger.warning(f"[{session_id}] Skipping inventory fetch: Missing prerequisites: {missing_inv_funcs}")
            formatted_inventory_string = "[Inventory Init/Config Error]"


        # --- 5. Context Refinement (RAG Cache OR Stateless OR None) ---
        context_for_prompt = extracted_owi_context # Start with OWI context (or None)
        refined_context_tokens = -1
        cache_update_performed = False
        cache_update_skipped = False
        final_context_selection_performed = False
        stateless_refinement_performed = False
        updated_cache_text_intermediate = "[Cache not initialized or updated]"

        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False)
        enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False)

        # --- RAG Cache Path ---
        if enable_rag_cache_global and self._cache_update_func and self._cache_select_func and self._get_rag_cache_db_func and self.sqlite_cursor:
            self.logger.info(f"[{session_id}] RAG Cache Feature ENABLED.")
            # Retrieve previous cache
            run_step1 = False
            previous_cache_text = ""
            try:
                 cache_result = await self._get_rag_cache_db_func(self.sqlite_cursor, session_id)
                 if cache_result is not None: previous_cache_text = cache_result
            except Exception as e_get_cache: self.logger.error(f"[{session_id}] Error retrieving previous cache: {e_get_cache}", exc_info=True)

            # Determine if Step 1 (Update) should run
            if not session_process_owi_rag:
                 self.logger.info(f"[{session_id}] Skipping RAG Cache Step 1 (session valve 'process_owi_rag=False').")
                 cache_update_skipped = True; run_step1 = False
                 updated_cache_text_intermediate = previous_cache_text
            else:
                 skip_len = False; skip_sim = False
                 owi_content_for_check = extracted_owi_context or ""
                 len_thresh = getattr(self.config, 'CACHE_UPDATE_SKIP_OWI_THRESHOLD', 50)
                 if len(owi_content_for_check.strip()) < len_thresh: skip_len = True; self.logger.info(f"[{session_id}] Cache S1 Skip: OWI len ({len(owi_content_for_check.strip())}) < {len_thresh}.")
                 elif self._calculate_similarity_func and previous_cache_text:
                      sim_thresh = getattr(self.config, 'CACHE_UPDATE_SIMILARITY_THRESHOLD', 0.9)
                      try:
                          sim_score = self._calculate_similarity_func(owi_content_for_check, previous_cache_text)
                          if sim_score > sim_thresh: skip_sim = True; self.logger.info(f"[{session_id}] Cache S1 Skip: Sim ({sim_score:.2f}) > {sim_thresh:.2f}.")
                      except Exception as e_sim: self.logger.error(f"[{session_id}] Error calculating similarity: {e_sim}")
                 cache_update_skipped = skip_len or skip_sim
                 run_step1 = not cache_update_skipped
                 if cache_update_skipped:
                      await self._emit_status(event_emitter, session_id, "Status: Skipping cache update (redundant OWI).")
                      updated_cache_text_intermediate = previous_cache_text

            # Prepare LLM configs
            cache_update_llm_config = {
                "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None),
                "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                "prompt_template": getattr(self.config, 'cache_update_prompt_template', DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT),
            }
            final_select_llm_config = {
                "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None),
                "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                "prompt_template": DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT, # Use imported default
            }
            configs_ok_step1 = all([cache_update_llm_config["url"], cache_update_llm_config["key"], cache_update_llm_config["prompt_template"]])
            configs_ok_step2 = all([final_select_llm_config["url"], final_select_llm_config["key"], final_select_llm_config["prompt_template"]])

            # Execute Step 1 (Update Cache)
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

            # Execute Step 2 (Select Final Context)
            if configs_ok_step2:
                 await self._emit_status(event_emitter, session_id, "Status: Selecting relevant context...")
                 # --- TEMPORARY WORKAROUND: Inject inventory into OWI context for selection ---
                 # TODO: Refactor cache.py to accept inventory string explicitly via prompt formatting args
                 temp_owi_for_select = extracted_owi_context or ""
                 inv_context_to_inject = formatted_inventory_string # Use the string fetched earlier
                 if inventory_enabled and inv_context_to_inject and "[Error" not in inv_context_to_inject and "[Inventory Management Disabled]" not in inv_context_to_inject and "[No Inventory Data Available]" not in inv_context_to_inject:
                      # Only inject if inventory is enabled and formatted string seems valid
                      temp_owi_for_select += f"\n\n--- Current Inventory ---\n{inv_context_to_inject}"
                      self.logger.info(f"[{session_id}] TEMPORARY: Injected formatted inventory into OWI context for selection step.")

                 # Call the selection function (from cache.py)
                 final_selected_context = await self._cache_select_func(
                      updated_cache_text=(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else ""),
                      current_owi_context=temp_owi_for_select, # Pass potentially augmented OWI
                      history_messages=current_active_history,
                      latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper,
                      context_selection_llm_config=final_select_llm_config,
                      history_count=getattr(self.config, 'refiner_history_count', 6),
                      dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_CtxSelect_{session_id}",
                 )
                 final_context_selection_performed = True
                 context_for_prompt = final_selected_context # Use the output of selection
                 log_step1_status = "Performed" if cache_update_performed else ("Skipped" if cache_update_skipped else "Not Run")
                 self.logger.info(f"[{session_id}] RAG Cache Step 2 complete. Context len: {len(context_for_prompt)}. Step 1: {log_step1_status}")
                 await self._emit_status(event_emitter, session_id, "Status: Context selection complete.", done=False)
            else:
                 self.logger.warning(f"[{session_id}] Skipping RAG Cache Step 2 (config missing). Using intermediate cache as context.")
                 context_for_prompt = updated_cache_text_intermediate

        # --- ELSE IF: Stateless Refinement Path ---
        elif enable_stateless_refin_global and self._stateless_refine_func:
            self.logger.info(f"[{session_id}] Stateless Refinement ENABLED.")
            await self._emit_status(event_emitter, session_id, "Status: Refining OWI context (stateless)...")
            if not extracted_owi_context: self.logger.debug(f"[{session_id}] Skipping stateless refinement: No OWI context.")
            elif not latest_user_query_str: self.logger.warning(f"[{session_id}] Skipping stateless refinement: Query empty.")
            else:
                 stateless_refiner_config = {
                     "url": getattr(self.config, 'refiner_llm_api_url', None), "key": getattr(self.config, 'refiner_llm_api_key', None),
                     "temp": getattr(self.config, 'refiner_llm_temperature', 0.3),
                     "prompt_template": getattr(self.config, 'stateless_refiner_prompt_template', None), # Use template from valve
                 }
                 if not stateless_refiner_config["url"] or not stateless_refiner_config["key"]:
                      self.logger.error(f"[{session_id}] Skipping stateless refinement: Refiner URL/Key missing.")
                      await self._emit_status(event_emitter, session_id, "ERROR: Stateless Refiner config incomplete.", done=False)
                 else:
                      try:
                          # Note: Inventory is NOT currently passed into stateless refinement
                          refined_stateless_context = await self._stateless_refine_func(
                               external_context=extracted_owi_context, history_messages=current_active_history,
                               latest_user_query=latest_user_query_str, llm_call_func=self._async_llm_call_wrapper,
                               refiner_llm_config=stateless_refiner_config,
                               skip_threshold=getattr(self.config, 'stateless_refiner_skip_threshold', 500),
                               history_count=getattr(self.config, 'refiner_history_count', 6),
                               dialogue_only_roles=self._dialogue_roles, caller_info=f"Orch_StatelessRef_{session_id}",
                          )
                          # Check if refinement actually changed the context
                          if refined_stateless_context != extracted_owi_context:
                               context_for_prompt = refined_stateless_context # Use refined context
                               stateless_refinement_performed = True
                               self.logger.info(f"[{session_id}] Stateless refinement successful (Length: {len(context_for_prompt)}).")
                               await self._emit_status(event_emitter, session_id, "Status: OWI context refined (stateless).", done=False)
                          else:
                               self.logger.info(f"[{session_id}] Stateless refinement resulted in no change or was skipped by length.")
                               # Keep context_for_prompt as original extracted_owi_context
                      except Exception as e_refine_stateless:
                          self.logger.error(f"[{session_id}] EXCEPTION during stateless refinement: {e_refine_stateless}", exc_info=True)
                          # Keep context_for_prompt as original extracted_owi_context on error

        # --- 6. Calculate Refined Context Tokens ---
        # Calculate tokens based on whatever ended up in context_for_prompt
        if self._count_tokens_func and self._tokenizer:
            try:
                token_source = context_for_prompt # This holds the result of refinement or the original OWI context
                if token_source and isinstance(token_source, str):
                     refined_context_tokens = self._count_tokens_func(token_source, self._tokenizer)
                else: refined_context_tokens = 0 # No context to count
                self.logger.debug(f"[{session_id}] Refined context tokens (RefOUT): {refined_context_tokens}")
            except Exception as e_tok_ref:
                refined_context_tokens = -1; self.logger.error(f"[{session_id}] Error calculating refined tokens: {e_tok_ref}")
        else: # Tokenizer unavailable
             refined_context_tokens = -1

        # --- 7. Combine Context Sources (Includes Inventory via Formatted String) ---
        combined_context_string = "[No background context generated]"
        if self._combine_context_func:
            try:
                # Pass the final selected/refined context and other sources
                # `formatted_inventory_string` was fetched earlier in this method
                combined_context_string = self._combine_context_func(
                    final_selected_context=(context_for_prompt if isinstance(context_for_prompt, str) else None),
                    t1_summaries=retrieved_t1_summaries,
                    t2_rag_results=retrieved_rag_summaries,
                    # Pass formatted inventory if enabled and valid
                    inventory_context=(formatted_inventory_string if inventory_enabled and formatted_inventory_string and "[Error" not in formatted_inventory_string and "[Disabled]" not in formatted_inventory_string else None)
                )
            except Exception as e_combine:
                 self.logger.error(f"[{session_id}] Error combining context: {e_combine}", exc_info=True); combined_context_string = "[Error combining context]"
        else:
            self.logger.error(f"[{session_id}] Cannot combine context: Function unavailable.")
            # Fallback: manually assemble if combine func missing? Or just use refined?
            # For now, stick with error placeholder.
        self.logger.debug(f"[{session_id}] Combined background context length: {len(combined_context_string)}.")

        # --- 8. Return relevant state ---
        return (
            combined_context_string,
            base_system_prompt_text,
            initial_owi_context_tokens,
            refined_context_tokens, # Tokens of the context *before* combining with T1/T2/Inventory
            cache_update_performed,
            cache_update_skipped,
            final_context_selection_performed,
            stateless_refinement_performed,
            formatted_inventory_string # Return the formatted string for status logging
        )


    async def _select_t0_history_slice(self, session_id: str, history_for_processing: List[Dict]) -> Tuple[List[Dict], int]:
        """ Selects the T0 history slice based on token limit and dialogue roles. """
        t0_raw_history_slice = []
        t0_dialogue_tokens = -1
        t0_token_limit = getattr(self.config, 't0_active_history_token_limit', 4000)

        try:
             # Use the library function select_turns_for_t0
             if self._tokenizer:
                  t0_raw_history_slice = select_turns_for_t0(
                      full_history=history_for_processing, # Pass history *before* the query
                      target_tokens=t0_token_limit,
                      tokenizer=self._tokenizer,
                      dialogue_only_roles=self._dialogue_roles
                  )
                  self.logger.info(f"[{session_id}] T0 Slice: Selected {len(t0_raw_history_slice)} dialogue msgs using select_turns_for_t0.")
             else:
                 self.logger.warning(f"[{session_id}] Tokenizer unavailable. Using simple turn count fallback for T0.")
                 fallback_turns = 10 # Or get from config?
                 dialogue_history = [msg for msg in history_for_processing if isinstance(msg, dict) and msg.get("role") in self._dialogue_roles]
                 start_idx = max(0, len(dialogue_history) - fallback_turns)
                 t0_raw_history_slice = dialogue_history[start_idx:]

             # Calculate T0 tokens based on the selected slice
             if t0_raw_history_slice and self._count_tokens_func and self._tokenizer:
                 try:
                     t0_dialogue_tokens = sum(self._count_tokens_func(msg["content"], self._tokenizer) for msg in t0_raw_history_slice if isinstance(msg, dict) and isinstance(msg.get("content"), str))
                 except Exception as e_tok_t0:
                     t0_dialogue_tokens = -1; self.logger.error(f"[{session_id}] Error calculating T0 tokens: {e_tok_t0}")
             elif not t0_raw_history_slice:
                 t0_dialogue_tokens = 0 # Empty slice has 0 tokens
             else: # Tokenizer unavailable or count failed
                 t0_dialogue_tokens = -1

        except Exception as e_select_t0:
             self.logger.error(f"[{session_id}] Error during T0 slice selection: {e_select_t0}", exc_info=True)
             t0_raw_history_slice = []; t0_dialogue_tokens = -1

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
                # Add guidance about using memory
                memory_guidance = "\n\n--- Memory Guidance ---\nUse the dialogue history and the background information provided (if any) to inform your response and maintain context."
                enhanced_system_prompt = base_system_prompt_text.strip() + memory_guidance
                session_long_term_goal = str(getattr(user_valves, 'long_term_goal', ''))
                include_acks = getattr(self.config, 'include_ack_turns', True)

                payload_dict = self._construct_payload_func(
                    system_prompt=enhanced_system_prompt,
                    history=t0_raw_history_slice, # Pass the filtered T0 dialogue slice
                    context=combined_context_string, # Pass the combined background context
                    query=latest_user_query_str,
                    long_term_goal=session_long_term_goal,
                    strategy="standard", # Or configure via valve? 'standard' puts history before context
                    include_ack_turns=include_acks,
                )
                # Validate the output structure
                if isinstance(payload_dict, dict) and "contents" in payload_dict and isinstance(payload_dict["contents"], list):
                    final_llm_payload_contents = payload_dict["contents"]
                    self.logger.info(f"[{session_id}] Constructed final payload ({len(final_llm_payload_contents)} turns).")
                else:
                    self.logger.error(f"[{session_id}] Payload constructor returned invalid structure: {type(payload_dict)}. Payload: {payload_dict}")
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
    ) -> Tuple[str, int]: # Returns status string and final token count
        """ Calculates final payload tokens and formats the status message string. """
        final_payload_tokens = -1
        # Calculate final payload tokens if possible
        if final_llm_payload_contents and self._count_tokens_func and self._tokenizer:
            try:
                # Sum tokens from all 'text' parts in the Google 'contents' format
                 final_payload_tokens = sum(
                     self._count_tokens_func(part["text"], self._tokenizer)
                     for turn in final_llm_payload_contents if isinstance(turn, dict)
                     for part in turn.get("parts", []) if isinstance(part, dict) and isinstance(part.get("text"), str)
                 )
            except Exception as e_tok_final:
                final_payload_tokens = -1; self.logger.error(f"[{session_id}] Error calculating final payload tokens: {e_tok_final}")
        elif not final_llm_payload_contents:
            final_payload_tokens = 0 # No payload means 0 tokens

        # Assemble Final Status Message Parts
        enable_rag_cache_global = getattr(self.config, 'enable_rag_cache', False)
        enable_stateless_refin_global = getattr(self.config, 'enable_stateless_refinement', False)

        # Determine Refinement Status indicator
        refinement_status = "Refined=N" # Default: No refinement ran or configured
        if enable_rag_cache_global and final_context_selection_performed:
            refinement_status = f"Refined=Cache(S1Skip={'Y' if cache_update_skipped else 'N'})"
        elif enable_stateless_refin_global and stateless_refinement_performed:
            refinement_status = "Refined=Stateless"

        # OWI Processing Status
        owi_proc_status = f"OWIProc={'ON' if session_process_owi_rag else 'OFF'}"

        # Basic Status Parts
        status_parts = [f"T1={t1_retrieved_count}", f"T2={t2_retrieved_count}", owi_proc_status, refinement_status]

        # Token Parts (only include if value is valid >= 0)
        token_parts = []
        if initial_owi_context_tokens >= 0: token_parts.append(f"OWI_IN={initial_owi_context_tokens}")
        if refined_context_tokens >= 0: token_parts.append(f"RefOUT={refined_context_tokens}")
        if summarization_prompt_tokens >= 0: token_parts.append(f"SumIN={summarization_prompt_tokens}")
        if summarization_output_tokens >= 0: token_parts.append(f"SumOUT={summarization_output_tokens}")
        if t0_dialogue_tokens >= 0: token_parts.append(f"Hist={t0_dialogue_tokens}")
        if final_payload_tokens >= 0: token_parts.append(f"FinalIN={final_payload_tokens}")

        # Combine into final message string
        status_message = "Status: " + ", ".join(status_parts) + (" | Tokens: " + " ".join(token_parts) if token_parts else "")

        return status_message, final_payload_tokens


    async def _execute_or_prepare_output(
        self, session_id: str, body: Dict, final_llm_payload_contents: Optional[List[Dict]],
        event_emitter: Optional[Callable], status_message: str, final_payload_tokens: int
    ) -> OrchestratorResult:
        """
        Executes the final LLM call (non-streaming if triggered) or prepares the output body.
        """

        output_body = body.copy() if isinstance(body, dict) else {}

        # Check if payload construction was successful
        if final_llm_payload_contents:
            output_body["messages"] = final_llm_payload_contents
            preserved_keys = ["model", "stream", "options", "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"]
            keys_preserved = [k for k in preserved_keys if k in body]
            for k in keys_preserved: output_body[k] = body[k]
            self.logger.info(f"[{session_id}] Output body updated with final payload. Preserved keys: {keys_preserved}.")
        else:
            self.logger.error(f"[{session_id}] Final payload construction failed. Cannot proceed.")
            await self._emit_status(event_emitter, session_id, "ERROR: Final payload preparation failed.", done=True)
            return {"error": "Orchestrator: Final payload construction failed.", "status_code": 500}

        # Check Final LLM Trigger Valves
        final_url = getattr(self.config, 'final_llm_api_url', None)
        final_key = getattr(self.config, 'final_llm_api_key', None)
        url_present = bool(final_url and isinstance(final_url, str) and final_url.strip())
        key_present = bool(final_key and isinstance(final_key, str) and final_key.strip())
        self.logger.debug(f"[{session_id}] Checking Final LLM Trigger. URL Present:{url_present}, Key Present:{key_present}")
        final_llm_triggered = url_present and key_present

        if final_llm_triggered:
            # --- Final LLM Call Triggered: ALWAYS Use Non-Streaming ---
            self.logger.info(f"[{session_id}] Final LLM Call via Pipe TRIGGERED (Non-Streaming).")
            await self._emit_status(event_emitter, session_id, "Status: Executing final LLM Call...", done=False)

            final_temp = getattr(self.config, 'final_llm_temperature', 0.7)
            final_timeout = getattr(self.config, 'final_llm_timeout', 120)
            final_call_payload_google_fmt = {"contents": final_llm_payload_contents}

            # Use the standard non-streaming wrapper (_async_llm_call_wrapper handles await/tuple)
            success, response_or_error = await self._async_llm_call_wrapper(
                api_url=final_url, api_key=final_key, payload=final_call_payload_google_fmt,
                temperature=final_temp, timeout=final_timeout,
                caller_info=f"Orch_FinalLLM_{session_id}"
            )

            final_status = status_message + (" (Success)" if success else " (Failed)")
            # DO NOT SET done=True here, wait for inventory
            await self._emit_status(event_emitter, session_id, final_status, done=False)

            if success and isinstance(response_or_error, str):
                self.logger.info(f"[{session_id}] Final LLM call successful. Returning response string.")
                return response_or_error # Return the string directly
            elif not success and isinstance(response_or_error, dict):
                self.logger.error(f"[{session_id}] Final LLM call failed. Returning error dict: {response_or_error}")
                # Emit final error status here if needed, but usually handled by process_turn
                await self._emit_status(event_emitter, session_id, status_message + " (Failed!)", done=True)
                return response_or_error # Return error dict
            else:
                self.logger.error(f"[{session_id}] Final LLM call returned unexpected format. Success={success}, Type={type(response_or_error)}")
                await self._emit_status(event_emitter, session_id, status_message + " (Unexpected Result)", done=True)
                return {"error": "Final LLM call returned unexpected result format.", "status_code": 500}

        else:
            # --- No Final LLM Call ---
            # Return Modified Payload Body for OWI processing (could be streaming or not based on original request)
            self.logger.info(f"[{session_id}] Final LLM Call disabled. Passing modified payload downstream.")
            # The status message calculated before this method was called will be emitted AFTER inventory
            # No need to emit status here.
            if getattr(self.config, 'debug_log_final_payload', False):
                 try: payload_str = json.dumps(output_body, indent=2)
                 except Exception: payload_str = str(output_body)
                 self.logger.debug(f"[{session_id}] Returning Payload:\n{payload_str}")
            # OWI will handle streaming based on output_body['stream'] if present
            return output_body


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
        """
        Processes a single turn by calling helper methods in sequence.
        Includes inventory management enable check and post-turn update call.
        Ensures database cursor is passed for inventory updates.
        (Reverted to use verbose status formatting and early history check).
        """
        pipe_entry_time_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Orchestrator process_turn [{session_id}]: Started at {pipe_entry_time_iso} (Regen Flag: {is_regeneration_heuristic})")

        # --- Check if Inventory Management is Enabled Globally ---
        inventory_enabled = getattr(self.config, 'enable_inventory_management', False)
        self.logger.info(f"[{session_id}] Inventory Management Enabled: {inventory_enabled}")

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
        formatted_inventory_string_for_status = ""
        status_message = "Status: Processing..." # Default initial status
        final_result: Optional[OrchestratorResult] = None
        final_llm_payload_contents: Optional[List[Dict]] = None
        inventory_update_completed = False # Flag to track if inventory step ran
        inventory_update_success_flag = False # Flag to track success of inventory step

        try:
            # --- 1. Initialization & History Handling ---
            await self._emit_status(event_emitter, session_id, "Status: Orchestrator syncing history...")
            incoming_messages = body.get("messages", [])
            stored_history = self.session_manager.get_active_history(session_id) or []
            if incoming_messages != stored_history:
                if len(incoming_messages) < len(stored_history):
                    self.logger.warning(f"[{session_id}] Incoming history shorter than stored. Resetting.")
                    self.session_manager.set_active_history(session_id, incoming_messages.copy())
                    # Reset summary index if history changed significantly (e.g., shortened)
                    # self.session_manager.set_last_summary_index(session_id, -1) # Optional reset
                else:
                    self.logger.debug(f"[{session_id}] Updating active history (Len: {len(incoming_messages)}).")
                    self.session_manager.set_active_history(session_id, incoming_messages.copy())
            else:
                self.logger.debug(f"[{session_id}] Incoming history matches stored.")

            current_active_history = self.session_manager.get_active_history(session_id) or []
            # *** REVERTED: Includes the early check for empty history ***
            if not current_active_history:
                 self.logger.error(f"[{session_id}] Active history is empty after sync. Cannot proceed.")
                 await self._emit_status(event_emitter, session_id, "ERROR: History synchronization failed.", done=True)
                 return {"error": "Active history is empty.", "status_code": 500}

            # --- 2. Determine Effective Query ---
            latest_user_query_str, history_for_processing = await self._determine_effective_query(
                session_id, current_active_history, is_regeneration_heuristic
            )
            if not latest_user_query_str and not is_regeneration_heuristic:
                 self.logger.error(f"[{session_id}] Cannot proceed without an effective user query (and not regeneration).")
                 await self._emit_status(event_emitter, session_id, "ERROR: Could not determine user query.", done=True)
                 return {"error": "Could not determine user query.", "status_code": 400}

            # --- 3. Tier 1 Summarization ---
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

            # --- 6. Prepare & Refine Background Context (Includes Inventory Fetch/Format) ---
            (combined_context_string, base_system_prompt_text,
             initial_owi_context_tokens, refined_context_tokens,
             cache_update_performed, cache_update_skipped,
             final_context_selection_performed, stateless_refinement_performed,
             formatted_inventory_string_for_status # Keep this name
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

            # --- 9. Calculate Status Message (Includes Inventory Status) ---
            # Note: This is the reverted position - calculation happens *before* final LLM call
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
            # Append Inventory Status (Reverted position)
            if inventory_enabled:
                inv_status = "ON"
                if not INVENTORY_MODULE_AVAILABLE: inv_status = "ERR_MOD"
                elif "[Error" in formatted_inventory_string_for_status: inv_status = "ERR_FMT"
                elif "[Disabled]" in formatted_inventory_string_for_status: inv_status = "OFF" # Should not happen if enabled=True
                elif "[No Inventory" in formatted_inventory_string_for_status: inv_status = "EMPTY"
                status_message += f" Inv={inv_status}"
            else:
                 status_message += " Inv=OFF"

            # Emit status *before* final call/return
            await self._emit_status(event_emitter, session_id, status_message, done=False) # Done=False here

            # --- 10. Execute Final LLM or Prepare Output ---
            # This now receives the calculated status message, but might emit its own final one
            final_result = await self._execute_or_prepare_output(
                session_id=session_id, body=body, final_llm_payload_contents=final_llm_payload_contents,
                event_emitter=event_emitter, status_message=status_message, final_payload_tokens=final_payload_tokens
            )

            # --- 11. Post-Turn Inventory Update (Reverted Position and Status Emission) ---
            inventory_update_attempted = False
            inventory_update_succeeded = False
            if inventory_enabled and self._update_inventories_func:
                inventory_update_attempted = True
                # Only run if previous steps didn't error and didn't return a stream
                if isinstance(final_result, dict) and "error" in final_result:
                    self.logger.warning(f"[{session_id}] Skipping post-turn inventory update due to upstream error: {final_result.get('error')}")
                elif isinstance(final_result, AsyncGenerator):
                    self.logger.warning(f"[{session_id}] Skipping post-turn inventory update: Streaming response detected.")
                elif isinstance(final_result, str): # Only run if final_result is the response string
                    self.logger.info(f"[{session_id}] Performing post-turn inventory update...")
                    # Emit its own status
                    await self._emit_status(event_emitter, session_id, "Status: Updating inventory state...", done=False)
                    try:
                        # Prepare config (copied from previous attempt)
                        inv_llm_url = getattr(self.config, 'inv_llm_api_url', None)
                        inv_llm_key = getattr(self.config, 'inv_llm_api_key', None)
                        inv_llm_temp = getattr(self.config, 'inv_llm_temperature', 0.3)
                        inv_llm_prompt_template = getattr(self.config, 'inv_llm_prompt_template', None)
                        template_seems_valid = inv_llm_prompt_template and isinstance(inv_llm_prompt_template, str) and len(inv_llm_prompt_template) > 50

                        if not inv_llm_url or not inv_llm_key or not template_seems_valid:
                            self.logger.error(f"[{session_id}] Inventory LLM config missing or invalid. Cannot perform update.")
                            await self._emit_status(event_emitter, session_id, "Status: Inventory update skipped (config error).", done=True) # Use Done=True
                        else:
                            inv_llm_config = {"url": inv_llm_url, "key": inv_llm_key, "temp": inv_llm_temp, "prompt_template": inv_llm_prompt_template}
                            history_for_inv_update_list = self._get_recent_turns_func(current_active_history, 4, exclude_last=False)
                            history_for_inv_update_str = self._format_history_func(history_for_inv_update_list)

                            # Create a new cursor for inventory updates
                            if not self.sqlite_cursor or not self.sqlite_cursor.connection:
                                 self.logger.error(f"[{session_id}] Cannot update inventory: SQLite cursor or connection is invalid.")
                                 await self._emit_status(event_emitter, session_id, "Status: Inventory update skipped (DB error).", done=True)
                            else:
                                 new_cursor = self.sqlite_cursor.connection.cursor()
                                 try:
                                     update_success = await self._update_inventories_func(
                                         cursor=new_cursor,
                                         session_id=session_id,
                                         main_llm_response=final_result, # Use the final response string
                                         user_query=latest_user_query_str,
                                         recent_history_str=history_for_inv_update_str,
                                         llm_call_func=self._async_llm_call_wrapper,
                                         db_get_inventory_func=self._get_char_inventory_db_func,
                                         db_update_inventory_func=self._update_char_inventory_db_func,
                                         inventory_llm_config=inv_llm_config
                                     )
                                     inventory_update_succeeded = update_success
                                     if update_success:
                                         self.logger.info(f"[{session_id}] Post-turn inventory update successful.")
                                         # Emit its own final status
                                         await self._emit_status(event_emitter, session_id, "Status: Inventory update complete.", done=True)
                                     else:
                                         self.logger.warning(f"[{session_id}] Post-turn inventory update function returned False.")
                                         await self._emit_status(event_emitter, session_id, "Status: Inventory update check finished.", done=True)
                                 except Exception as e_inv_call:
                                      self.logger.error(f"[{session_id}] Exception during inventory update function call: {e_inv_call}", exc_info=True)
                                      await self._emit_status(event_emitter, session_id, "Status: Error during inventory update.", done=True)
                                 finally:
                                      new_cursor.close()

                    except Exception as e_inv_update_outer:
                        self.logger.error(f"[{session_id}] Outer error during post-turn inventory update setup: {e_inv_update_outer}", exc_info=True)
                        await self._emit_status(event_emitter, session_id, "Status: Error during inventory update setup.", done=True)
                # Handle cases where inventory wasn't run (e.g., streaming response, error, LLM off)
                elif isinstance(final_result, dict) and final_result.get("messages") == final_llm_payload_contents:
                    self.logger.info(f"[{session_id}] Skipping post-turn inventory update: Final LLM call was disabled or payload unchanged.")
                    # If inventory was enabled, emit a final status clarifying why it was skipped
                    if inventory_enabled:
                        # Use the previously calculated main status message and append clarification
                        await self._emit_status(event_emitter, session_id, f"{status_message} (Inv Check Skipped - Final LLM Off)", done=True)
                else: # Other cases (error dict, stream)
                    # If inventory was enabled, emit a final status clarifying why it was skipped
                    if inventory_enabled:
                         await self._emit_status(event_emitter, session_id, f"{status_message} (Inv Check Skipped - Upstream Err/Stream)", done=True)

            elif not inventory_enabled:
                 # If inventory is globally disabled, emit the main status message with done=True
                 self.logger.debug(f"[{session_id}] Skipping post-turn inventory update: Disabled by valve.")
                 await self._emit_status(event_emitter, session_id, status_message, done=True) # Main status, done=True
            else: # Inventory enabled but function missing
                 self.logger.warning(f"[{session_id}] Skipping post-turn inventory update: Update function missing/unavailable.")
                 await self._emit_status(event_emitter, session_id, "Status: Inventory update skipped (function missing).", done=True)

            # --- 12. Log Final Payload (if enabled) ---
            # Log payload if debug enabled *AND* final LLM was *not* triggered (i.e., returning payload dict)
            if getattr(self.config, 'debug_log_final_payload', False) and isinstance(final_result, dict) and "messages" in final_result:
                # Check if final LLM was triggered - if not, final_result *is* the payload dict
                final_url = getattr(self.config, 'final_llm_api_url', None); final_key = getattr(self.config, 'final_llm_api_key', None)
                final_llm_triggered = bool(final_url and final_key)
                if not final_llm_triggered:
                    self.logger.info(f"[{session_id}] Logging final payload dict due to debug valve (Final LLM Off).")
                    self._log_debug_final_payload(session_id, final_result)
                else:
                    self.logger.debug(f"[{session_id}] Skipping final payload log: Final LLM was triggered.")


            # --- 13. Return Final Result ---
            pipe_end_time_iso = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"Orchestrator process_turn [{session_id}]: Finished at {pipe_end_time_iso}")
            if final_result is None:
                self.logger.error(f"[{session_id}] Final result is None at end of processing. Returning error.")
                # A final status should have already been emitted in the block where final_result could become None
                return {"error": "Internal processing error, final result was None.", "status_code": 500}
            # Important: If inventory ran, the final status was already emitted.
            # If inventory did *not* run (disabled or skipped), the final status was emitted above.
            # So, just return the result.
            return final_result

        # --- Exception Handling ---
        except asyncio.CancelledError:
             self.logger.info(f"[{session_id or 'unknown'}] Orchestrator process_turn cancelled.")
             await self._emit_status(event_emitter, session_id or 'unknown', "Status: Processing cancelled.", done=True)
             raise
        except Exception as e_orch:
            session_id_for_log = session_id if 'session_id' in locals() and session_id != "uninitialized_session" else 'unknown'
            self.logger.critical(f"[{session_id_for_log}] Orchestrator UNHANDLED EXCEPTION in process_turn: {e_orch}", exc_info=True)
            await self._emit_status(event_emitter, session_id_for_log, f"ERROR: Orchestrator Failed ({type(e_orch).__name__})", done=True)
            return {"error": f"Orchestrator failed: {type(e_orch).__name__}", "status_code": 500}
# [[END MODIFIED orchestration.py - FULL]]

# === END OF FILE i4_llm_agent/orchestration.py ===