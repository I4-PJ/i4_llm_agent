# [[START MODIFIED context_processor.py - Use Snapshot v0.2.4]]
# i4_llm_agent/context_processor.py

import logging
import asyncio
import sqlite3
import json
import re
from typing import (
    Tuple, Union, List, Dict, Optional, Any, Callable, Coroutine, Sequence
)

# --- i4_llm_agent Imports ---
from . import database # Import the whole module to access functions
from .history import select_turns_for_t0, format_history_for_llm, DIALOGUE_ROLES
from .prompting import (
    process_system_prompt, # Base prompt/context extraction
    combine_background_context, # Combining various sources
    construct_final_llm_payload, # Final payload structure
    generate_rag_query, # T2 query generation (if enabled)
    DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT, # Cache Maintainer Template
)
from .cache import update_rag_cache_maintainer # Cache Maintainer Function

# Import Inventory functions if available
try:
    from .inventory import format_inventory_for_prompt as format_inv_func
    INVENTORY_MODULE_AVAILABLE = True
except ImportError:
    INVENTORY_MODULE_AVAILABLE = False
    def format_inv_func(*args, **kwargs): return "[Inventory Module Not Found]"

from .utils import TIKTOKEN_AVAILABLE, count_tokens # Token counting

# Initialize logger for this module
logger = logging.getLogger(__name__) # i4_llm_agent.context_processor
CTX_PROC_VERSION = "0.2.4" # <<< Version Updated


async def process_context_and_prepare_payload(
    session_id: str,
    body: Dict,
    user_valves: Any, # Pipe.UserValves object
    current_active_history: List[Dict], # Full history including latest query
    history_for_processing: List[Dict], # History slice before query
    latest_user_query_str: str, # Query determined by orchestrator
    # --- State Info (Passed from Orchestrator) ---
    current_scene_state_dict: Dict[str, Any],
    current_world_state_dict: Dict[str, Any],
    generated_event_hint_text: Optional[str], # Might be None if regen skipped in orchestrator
    generated_weather_proposal: Optional[Dict[str, Optional[str]]], # Might be default if regen skipped
    # --- Config & Utilities (Passed from Orchestrator) ---
    config: Any, # Pipe.Valves object
    logger: logging.Logger, # Use passed-in logger
    sqlite_cursor: Optional[sqlite3.Cursor],
    chroma_client: Optional[Any], # ChromaDB client instance
    chroma_embed_wrapper: Optional[Any], # ChromaDB embedding function wrapper
    embedding_func: Optional[Callable[[Sequence[str]], List[List[float]]]], # OWI Embedding function
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # Async LLM wrapper
    tokenizer: Optional[Callable], # Tiktoken tokenizer instance
    event_emitter: Optional[Callable],
    orchestrator_debug_path_getter: Optional[Callable[[str], Optional[str]]],
    dialogue_roles: List[str],
    session_period_setting: Optional[str] = None,
    # --- Database Function Aliases (Passed from Orchestrator) ---
    db_get_recent_t1_summaries_func: Optional[Callable[..., Coroutine[Any, Any, List[Tuple[str, Dict]]]]] = None,
    db_get_recent_aged_summaries_func: Optional[Callable[..., Coroutine[Any, Any, List[Tuple[str, Dict]]]]] = None,
    # --- Regeneration Handling ---
    is_regeneration_heuristic: bool = False,
    # <<< NEW: Snapshot Components (passed by orchestrator if regen and snapshot exists) >>>
    snapshot_base_prompt: Optional[str] = None,
    snapshot_t0_history: Optional[List[Dict]] = None,
    snapshot_combined_context: Optional[str] = None,
    snapshot_latest_query: Optional[str] = None, # Query used for the original turn
    snapshot_event_hint: Optional[str] = None, # Hint used for the original turn

) -> Tuple[Optional[List[Dict]], Dict[str, Any], Optional[str], Optional[List[Dict]], Optional[str]]:
    """
    Processes context sources or uses a provided snapshot for regeneration.
    Version: 0.2.4

    If is_regeneration_heuristic is True and snapshot components are provided,
    it uses the snapshot directly to build the final payload, skipping most steps.
    Otherwise, it processes context normally (skipping certain steps if regenerating
    without a snapshot) and returns the payload, status, and key context components.

    Returns:
        Tuple[Optional[List[Dict]], Dict[str, Any], Optional[str], Optional[List[Dict]], Optional[str]]:
            - Final LLM payload contents list (or None on error).
            - Dictionary containing status information about context processing.
            - Base system prompt text used/calculated (for snapshotting).
            - T0 history slice used/calculated (for snapshotting).
            - Combined background context string used/calculated (for snapshotting).
    """
    func_logger = logger
    func_logger.debug(f"[{session_id}] ContextProcessor v{CTX_PROC_VERSION}: Entered (Regen Flag: {is_regeneration_heuristic}). Snapshot provided: {bool(snapshot_base_prompt is not None)}")

    # Initialize return values for context components (used for snapshotting)
    final_base_prompt: Optional[str] = None
    final_t0_history: Optional[List[Dict]] = None
    final_combined_context: Optional[str] = None

    context_status_info = {
        "t1_retrieved_count": 0,
        "aged_retrieved_count": 0,
        "t2_retrieved_count": 0,
        "t0_dialogue_tokens": -1,
        "initial_owi_context_tokens": -1,
        "cache_maintenance_performed": False,
        "cache_retrieved_on_regen": False,
        "used_context_snapshot": False, # <<< New flag
        "final_context_tokens": -1,
        "error": None,
    }

    # --- Helper Functions (Unchanged) ---
    async def _emit_status(description: str, done: bool = False):
        if event_emitter and callable(event_emitter) and getattr(config, 'emit_status_updates', True):
            try:
                status_data = { "type": "status", "data": {"description": str(description), "done": bool(done)} }
                if asyncio.iscoroutinefunction(event_emitter): await event_emitter(status_data)
                else: event_emitter(status_data)
            except Exception as e_emit: func_logger.warning(f"[{session_id}] ContextProcessor failed emit status '{description}': {e_emit}")
        else: func_logger.debug(f"[{session_id}] ContextProcessor status update (not emitted): '{description}' (Done: {done})")

    def _count_tokens_safe(text: Optional[str]) -> int:
        if not text or not tokenizer or not TIKTOKEN_AVAILABLE: return 0
        try: return count_tokens(text, tokenizer)
        except Exception as e_tok: func_logger.error(f"[{session_id}] Token count failed: {e_tok}"); return 0

    # --- Prerequisites Check (Unchanged) ---
    if not sqlite_cursor:
        context_status_info["error"] = "SQLite cursor unavailable."
        func_logger.critical(f"[{session_id}] ContextProcessor CRITICAL: SQLite cursor unavailable.")
        # Ensure correct return signature on early exit
        return None, context_status_info, None, None, None
    if not db_get_recent_t1_summaries_func:
        func_logger.warning(f"[{session_id}] ContextProcessor Warning: db_get_recent_t1_summaries_func not provided.")
    if not db_get_recent_aged_summaries_func:
        func_logger.warning(f"[{session_id}] ContextProcessor Warning: db_get_recent_aged_summaries_func not provided.")


    # --- <<< NEW: Regeneration Path using Snapshot >>> ---
    if is_regeneration_heuristic and snapshot_base_prompt is not None \
       and snapshot_t0_history is not None and snapshot_combined_context is not None \
       and snapshot_latest_query is not None:

        func_logger.info(f"[{session_id}] Regen: Using provided context snapshot.")
        context_status_info["used_context_snapshot"] = True
        await _emit_status("Status: Using previous context snapshot...")

        # Use snapshot components directly
        final_base_prompt = snapshot_base_prompt
        final_t0_history = snapshot_t0_history
        final_combined_context = snapshot_combined_context
        query_to_use = snapshot_latest_query # Use the query from the snapshot
        hint_to_use = snapshot_event_hint # Use the hint from the snapshot

        # Calculate token counts from snapshot data for status
        context_status_info["t0_dialogue_tokens"] = _count_tokens_safe(
            format_history_for_llm(final_t0_history)
        )
        context_status_info["final_context_tokens"] = _count_tokens_safe(final_combined_context)

        # Skip steps 1-7 entirely

        # Step 8: Construct Final LLM Payload using snapshot data
        await _emit_status("Status: Constructing final payload from snapshot...")
        final_llm_payload = construct_final_llm_payload(
             system_prompt=final_base_prompt,
             history=final_t0_history,
             context=final_combined_context,
             query=query_to_use, # Use query from snapshot
             long_term_goal=getattr(user_valves, 'long_term_goal', None),
             event_hint=hint_to_use, # Use hint from snapshot
             period_setting=session_period_setting,
             include_ack_turns=getattr(config, 'include_ack_turns', True)
        )
        if isinstance(final_llm_payload, dict) and "error" in final_llm_payload:
            context_status_info["error"] = f"Snapshot Payload construction failed: {final_llm_payload['error']}"
            func_logger.error(f"[{session_id}] Snapshot Payload construction failed: {final_llm_payload['error']}")
            return None, context_status_info, final_base_prompt, final_t0_history, final_combined_context
        elif isinstance(final_llm_payload, dict) and "contents" in final_llm_payload:
            func_logger.info(f"[{session_id}] Final payload constructed successfully from snapshot ({len(final_llm_payload['contents'])} turns).")
            await _emit_status("Status: Payload construction complete.")
            # Return payload, status, and the snapshot components themselves
            return final_llm_payload["contents"], context_status_info, final_base_prompt, final_t0_history, final_combined_context
        else:
            context_status_info["error"] = "Snapshot Payload construction returned unexpected format."
            func_logger.error(f"[{session_id}] Snapshot Payload construction failed: Unexpected format {type(final_llm_payload)}.")
            return None, context_status_info, final_base_prompt, final_t0_history, final_combined_context

    # --- <<< END Regeneration Path using Snapshot >>> ---


    # --- Normal Turn Processing OR Regen Fallback (If snapshot was missing) ---
    if is_regeneration_heuristic:
        # This block only runs if is_regeneration_heuristic was True BUT snapshot was missing
        func_logger.warning(f"[{session_id}] Regen: Snapshot data missing or incomplete. Falling back to standard regen context processing (skipping LLM calls/DB queries).")
        await _emit_status("Status: Processing context (Regen Fallback)...")
        context_status_info["used_context_snapshot"] = False # Explicitly set


    # --- Step 1: Select T0 History Slice ---
    # (Run always, needed for both normal and regen fallback)
    await _emit_status("Status: Selecting dialogue history...")
    t0_active_history_token_limit = getattr(config, 't0_active_history_token_limit', 4000)
    # Use history_for_processing passed from orchestrator (already regen-aware)
    t0_history_slice, t0_dialogue_tokens, _ = select_turns_for_t0(
        history_for_processing,
        target_tokens=t0_active_history_token_limit,
        tokenizer=tokenizer,
        dialogue_only_roles=dialogue_roles
    )
    context_status_info["t0_dialogue_tokens"] = t0_dialogue_tokens
    func_logger.info(f"[{session_id}] T0 History: Selected {len(t0_history_slice)} turns ({t0_dialogue_tokens} tokens).")
    final_t0_history = t0_history_slice # Store for snapshot return


    # --- Step 2: Process OWI RAG (Extract Only) ---
    # (Run always, base prompt needed for both normal and regen fallback)
    await _emit_status("Status: Processing OWI context...")
    raw_owi_system_prompt = ""
    for msg in body.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            raw_owi_system_prompt = msg.get("content", "")
            break
    base_system_prompt_text, extracted_owi_context_str = process_system_prompt([{"role": "system", "content": raw_owi_system_prompt}])
    initial_owi_context_tokens = _count_tokens_safe(extracted_owi_context_str)
    context_status_info["initial_owi_context_tokens"] = initial_owi_context_tokens
    func_logger.info(f"[{session_id}] OWI Context: Extracted {initial_owi_context_tokens} tokens from <source> tags.")
    func_logger.debug(f"[{session_id}] Initial base_system_prompt_text length: {len(base_system_prompt_text)}")

    # Apply User Valve Text Removal (Run always)
    text_to_remove = getattr(user_valves, 'text_block_to_remove', '').strip()
    if text_to_remove and base_system_prompt_text:
        normalized_text_to_remove = text_to_remove.replace('\r\n', '\n')
        normalized_base_prompt_text = base_system_prompt_text.replace('\r\n', '\n')
        modified_base_prompt_text = normalized_base_prompt_text.replace(normalized_text_to_remove, "")
        if len(modified_base_prompt_text) < len(normalized_base_prompt_text):
            func_logger.info(f"[{session_id}] Removed specified text block from BASE system prompt (using normalized comparison).")
            base_system_prompt_text = modified_base_prompt_text
            func_logger.debug(f"[{session_id}] Modified base_system_prompt_text length: {len(base_system_prompt_text)}")
        else:
            func_logger.warning(f"[{session_id}] Specified text block to remove was not found in BASE system prompt (using normalized comparison).")
    final_base_prompt = base_system_prompt_text # Store for snapshot return

    # Check if OWI Context should be processed (Run always)
    process_owi_rag_flag = getattr(user_valves, 'process_owi_rag', True)
    effective_owi_context = extracted_owi_context_str if process_owi_rag_flag else None
    if not process_owi_rag_flag:
        func_logger.info(f"[{session_id}] Skipping OWI RAG processing and Cache Maintenance as per user valve 'process_owi_rag'.")
    else:
         func_logger.debug(f"[{session_id}] OWI RAG context processing enabled by user valve.")


    # --- Step 3: Handle RAG Cache (Maintain or Retrieve - Regen Fallback Logic) ---
    maintained_cache_text = ""
    enable_cache_maintainer = getattr(config, 'enable_rag_cache_maintainer', False)
    cache_llm_url = getattr(config, 'refiner_llm_api_url', None); cache_llm_key = getattr(config, 'refiner_llm_api_key', None)
    cache_llm_temp = getattr(config, 'refiner_llm_temperature', 0.3); cache_history_count = getattr(config, 'refiner_history_count', 6)

    # This block now handles Normal Turns AND Regen Fallback (snapshot failed)
    if is_regeneration_heuristic: # Regen Fallback: Retrieve cache from DB
        func_logger.info(f"[{session_id}] Regen Fallback: Skipping Cache Maintainer LLM call. Attempting to retrieve last known cache state.")
        await _emit_status("Status: Retrieving cached context (Regen Fallback)...")
        try:
            retrieved_cache = await database.get_rag_cache(session_id, sqlite_cursor)
            if retrieved_cache is not None:
                maintained_cache_text = retrieved_cache
                context_status_info["cache_retrieved_on_regen"] = True
                func_logger.info(f"[{session_id}] Regen Fallback: Successfully retrieved last cache state (Len: {len(maintained_cache_text)}).")
            else:
                maintained_cache_text = effective_owi_context or ""
                func_logger.warning(f"[{session_id}] Regen Fallback: No previous cache found in DB. Using current OWI context as fallback (Len: {len(maintained_cache_text)}).")
        except Exception as e_get_cache_regen:
            func_logger.error(f"[{session_id}] Regen Fallback: Error retrieving cache state: {e_get_cache_regen}. Using current OWI context.", exc_info=True)
            maintained_cache_text = effective_owi_context or ""
        context_status_info["cache_maintenance_performed"] = False

    elif enable_cache_maintainer and process_owi_rag_flag and cache_llm_url and cache_llm_key: # Normal Turn: Maintain Cache
        await _emit_status("Status: Maintaining session cache...")
        func_logger.info(f"[{session_id}] Applying RAG Cache Maintainer logic...")
        cache_maintainer_llm_config = { "url": cache_llm_url, "key": cache_llm_key, "temp": cache_llm_temp, "prompt_template": DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT }
        try:
            maintained_cache_text = await update_rag_cache_maintainer(
                session_id=session_id, current_owi_context=effective_owi_context,
                history_messages=history_for_processing, latest_user_query=latest_user_query_str,
                llm_call_func=llm_call_func, sqlite_cursor=sqlite_cursor,
                cache_maintainer_llm_config=cache_maintainer_llm_config, history_count=cache_history_count,
                dialogue_only_roles=dialogue_roles, caller_info=f"CtxProc_CacheMaint_{session_id}",
                debug_log_path_getter=orchestrator_debug_path_getter
            )
            context_status_info["cache_maintenance_performed"] = True
            func_logger.info(f"[{session_id}] Cache Maintainer finished. Resulting cache length: {len(maintained_cache_text)}")
            await _emit_status("Status: Cache maintenance complete.")
        except Exception as e_maintain:
            func_logger.error(f"[{session_id}] Error during Cache Maintainer call: {e_maintain}", exc_info=True)
            context_status_info["error"] = f"Cache Maintainer Error: {e_maintain}"
            try: # Fallback to previous cache on error
                 fallback_cache = await database.get_rag_cache(session_id, sqlite_cursor)
                 maintained_cache_text = fallback_cache if fallback_cache is not None else (effective_owi_context or "")
                 func_logger.warning(f"[{session_id}] Falling back to previous cache/OWI state (len: {len(maintained_cache_text)}) due to maintainer error.")
            except Exception as e_fallback:
                 func_logger.error(f"[{session_id}] Error retrieving fallback cache state: {e_fallback}. Using OWI/empty.", exc_info=True)
                 maintained_cache_text = effective_owi_context or ""

    else: # Normal Turn, but Cache Maintainer disabled or config missing
        if not enable_cache_maintainer: func_logger.info(f"[{session_id}] RAG Cache Maintainer is disabled by global config.")
        elif not process_owi_rag_flag: func_logger.info(f"[{session_id}] RAG Cache Maintainer skipped as OWI processing is disabled.")
        else: func_logger.warning(f"[{session_id}] RAG Cache Maintainer enabled, but URL/Key missing. Skipping.")
        maintained_cache_text = effective_owi_context or ""
        context_status_info["cache_maintenance_performed"] = False


    # --- Step 4: Fetch T1 and Aged Summaries ---
    # (Run always)
    await _emit_status("Status: Fetching recent summaries...")
    recent_t1_summaries_data: List[Tuple[str, Dict]] = []
    recent_aged_summaries_data: List[Tuple[str, Dict]] = []
    max_t1_blocks = getattr(config, 'max_stored_summary_blocks', 20)
    if db_get_recent_t1_summaries_func:
        try:
            recent_t1_summaries_data = await db_get_recent_t1_summaries_func( cursor=sqlite_cursor, session_id=session_id, limit=max_t1_blocks )
            context_status_info["t1_retrieved_count"] = len(recent_t1_summaries_data)
            func_logger.info(f"[{session_id}] Fetched {len(recent_t1_summaries_data)} T1 summaries via DB func.")
        except Exception as e_get_t1: func_logger.error(f"[{session_id}] Error fetching T1: {e_get_t1}", exc_info=True); context_status_info["t1_retrieved_count"] = 0
    else: func_logger.warning(f"[{session_id}] Cannot fetch T1: DB function alias missing.")
    if db_get_recent_aged_summaries_func:
        try:
            aged_limit = max(1, max_t1_blocks // 2)
            recent_aged_summaries_data = await db_get_recent_aged_summaries_func( cursor=sqlite_cursor, session_id=session_id, limit=aged_limit )
            context_status_info["aged_retrieved_count"] = len(recent_aged_summaries_data)
            func_logger.info(f"[{session_id}] Fetched {len(recent_aged_summaries_data)} Aged summaries via DB func.")
        except Exception as e_get_aged:
            if "no such table: aged_summaries" in str(e_get_aged): func_logger.info(f"[{session_id}] Table 'aged_summaries' not found.")
            else: func_logger.error(f"[{session_id}] Error fetching Aged: {e_get_aged}", exc_info=True)
            context_status_info["aged_retrieved_count"] = 0
    else: func_logger.warning(f"[{session_id}] Cannot fetch Aged: DB function alias missing.")

    # --- Step 5: T2 RAG Query & Retrieval (Skip if regenerating - fallback or snapshot) ---
    t2_rag_results: Optional[List[str]] = None
    if is_regeneration_heuristic: # Covers both snapshot and fallback regen cases
        func_logger.info(f"[{session_id}] Regen: Skipping T2 RAG query and retrieval.")
        await _emit_status("Status: Skipping long-term memory query (Regen)...")
        context_status_info["t2_retrieved_count"] = 0
    else: # Normal Turn: Perform T2 RAG
        await _emit_status("Status: Checking long-term memory...")
        ragq_llm_url = getattr(config, 'ragq_llm_api_url', None); ragq_llm_key = getattr(config, 'ragq_llm_api_key', None)
        can_rag = all([ database.CHROMADB_AVAILABLE, chroma_client, chroma_embed_wrapper, embedding_func, ragq_llm_url, ragq_llm_key, latest_user_query_str ])
        if can_rag:
            func_logger.info(f"[{session_id}] T2 RAG: Performing query...")
            await _emit_status("Status: Querying long-term memory...")
            try:
                t0_history_str = format_history_for_llm(t0_history_slice)
                rag_query = await generate_rag_query( latest_message_str=latest_user_query_str, dialogue_context_str=t0_history_str, llm_call_func=llm_call_func, api_url=ragq_llm_url, api_key=ragq_llm_key, temperature=getattr(config, 'ragq_llm_temperature', 0.3), caller_info=f"CtxProc_RAGQ_{session_id}" )
                if not rag_query or rag_query.startswith("[Error"): func_logger.warning(f"[{session_id}] T2 RAG: Query generation failed: {rag_query}")
                else:
                    rag_results_count = getattr(config, 'rag_summary_results_count', 3)
                    base_prefix = getattr(config, 'summary_collection_prefix', 'sm_t2_'); safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]; tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
                    tier2_collection = await database.get_or_create_chroma_collection( chroma_client, tier2_collection_name, chroma_embed_wrapper )
                    if not tier2_collection: func_logger.error(f"[{session_id}] T2 RAG: Failed get/create collection '{tier2_collection_name}'.")
                    else:
                        query_embedding_list = None
                        try:
                            embedding_result = await asyncio.to_thread(embedding_func, [rag_query])
                            if isinstance(embedding_result, list) and len(embedding_result) == 1 and isinstance(embedding_result[0], list): query_embedding_list = embedding_result; func_logger.debug(f"[{session_id}] T2 RAG: Generated query embedding.")
                            else: func_logger.error(f"[{session_id}] T2 RAG: Embedding func unexpected format: {type(embedding_result)}")
                        except Exception as e_embed: func_logger.error(f"[{session_id}] T2 RAG: Exception query embedding: {e_embed}", exc_info=True)
                        if query_embedding_list:
                            chroma_results_dict = await database.query_chroma_collection( collection=tier2_collection, query_embeddings=query_embedding_list, n_results=rag_results_count, include=["documents", "distances", "metadatas"] )
                            if chroma_results_dict and isinstance(chroma_results_dict.get('documents'), list):
                                if len(chroma_results_dict['documents']) > 0 and isinstance(chroma_results_dict['documents'][0], list):
                                    t2_rag_results = chroma_results_dict['documents'][0]
                                    if all(isinstance(doc, str) for doc in t2_rag_results): context_status_info["t2_retrieved_count"] = len(t2_rag_results); func_logger.info(f"[{session_id}] T2 RAG: Retrieved {len(t2_rag_results)} results.")
                                    else: func_logger.error(f"[{session_id}] T2 RAG: Expected list of strings, got non-strings. Resetting."); t2_rag_results = None
                                else: func_logger.warning(f"[{session_id}] T2 RAG: 'documents' not list of lists. Resetting."); t2_rag_results = None
                            else: func_logger.info(f"[{session_id}] T2 RAG: No results/invalid format for '{rag_query}'."); t2_rag_results = None
            except Exception as e_rag: func_logger.error(f"[{session_id}] T2 RAG: Error query/retrieval: {e_rag}", exc_info=True); context_status_info["error"] = f"T2 RAG Error: {e_rag}"
        else: missing_rag_prereqs = [p for p, v in {"chroma_lib": database.CHROMADB_AVAILABLE, "chroma_client": chroma_client, "wrapper": chroma_embed_wrapper, "embed_f": embedding_func, "url": ragq_llm_url, "key": ragq_llm_key, "query": bool(latest_user_query_str)}.items() if not v]; func_logger.debug(f"[{session_id}] T2 RAG: Skipping, missing: {', '.join(missing_rag_prereqs)}")

    # --- Step 6: Format Inventory Context (Skip if regenerating - fallback or snapshot) ---
    inventory_context_str: Optional[str] = None
    inventory_enabled = getattr(config, 'enable_inventory_management', False)
    if is_regeneration_heuristic: # Covers both snapshot and fallback regen cases
        func_logger.info(f"[{session_id}] Regen: Skipping Inventory context formatting.")
        await _emit_status("Status: Skipping inventory formatting (Regen)...")
        inventory_context_str = "[Inventory Formatting Skipped on Regen]"
    elif inventory_enabled and INVENTORY_MODULE_AVAILABLE: # Normal Turn: Format Inventory
        await _emit_status("Status: Formatting inventory context...")
        try:
            all_inventories = await database.get_all_inventories_for_session(sqlite_cursor, session_id)
            inventory_context_str = format_inv_func(all_inventories)
            func_logger.info(f"[{session_id}] Inventory context formatted (Len: {len(inventory_context_str or '')}).")
        except Exception as e_inv_fmt: func_logger.error(f"[{session_id}] Error formatting inventory: {e_inv_fmt}", exc_info=True); inventory_context_str = "[Inventory Formatting Error]"
    elif inventory_enabled and not INVENTORY_MODULE_AVAILABLE: func_logger.error(f"[{session_id}] Inventory enabled but module unavailable."); inventory_context_str = "[Inventory Module Error]"
    else: inventory_context_str = "[Inventory Disabled]"; func_logger.debug(f"[{session_id}] Inventory disabled.")

    # --- Step 7: Combine All Background Context ---
    # (Run always, uses results from previous steps which are now regen-aware)
    await _emit_status("Status: Combining background context...")
    scene_desc_for_combine = current_scene_state_dict.get('description')
    current_day_for_combine = current_world_state_dict.get('day')
    current_time_for_combine = current_world_state_dict.get('time_of_day')
    current_season_for_combine = current_world_state_dict.get('season')
    current_weather_for_combine = current_world_state_dict.get('weather')

    combined_background_context = combine_background_context(
        final_selected_context=maintained_cache_text, # Cache text from step 3
        t1_summaries=recent_t1_summaries_data,
        aged_summaries=recent_aged_summaries_data,
        t2_rag_results=t2_rag_results, # Will be None if regen skipped
        scene_description=scene_desc_for_combine,
        inventory_context=inventory_context_str, # Will be placeholder if regen skipped
        current_day=current_day_for_combine,
        current_time_of_day=current_time_for_combine,
        current_season=current_season_for_combine,
        current_weather=current_weather_for_combine,
        weather_proposal=generated_weather_proposal,
        event_hint_text=generated_event_hint_text # Will be None if regen skipped in orchestrator
    )
    final_combined_context = combined_background_context # Store for snapshot return

    final_context_tokens = _count_tokens_safe(final_combined_context)
    context_status_info["final_context_tokens"] = final_context_tokens
    func_logger.info(f"[{session_id}] Combined background context length: {len(final_combined_context)} ({final_context_tokens} tokens)")

    # --- Step 8: Construct Final LLM Payload ---
    # (Run always, uses combined context)
    await _emit_status("Status: Constructing final payload...")
    # Use latest_user_query_str determined by orchestrator for normal/regen-fallback
    query_for_payload = latest_user_query_str
    # Use hint passed from orchestrator (will be None if regen)
    hint_for_payload = generated_event_hint_text

    final_llm_payload = construct_final_llm_payload(
         system_prompt=final_base_prompt, # Use base prompt determined earlier
         history=final_t0_history, # Use T0 slice determined earlier
         context=final_combined_context, # Use combined context determined earlier
         query=query_for_payload,
         long_term_goal=getattr(user_valves, 'long_term_goal', None),
         event_hint=hint_for_payload,
         period_setting=session_period_setting,
         include_ack_turns=getattr(config, 'include_ack_turns', True)
    )
    if isinstance(final_llm_payload, dict) and "error" in final_llm_payload:
        context_status_info["error"] = f"Payload construction failed: {final_llm_payload['error']}"
        func_logger.error(f"[{session_id}] Payload construction failed: {final_llm_payload['error']}")
        # Return None payload, status, and calculated context parts
        return None, context_status_info, final_base_prompt, final_t0_history, final_combined_context
    elif isinstance(final_llm_payload, dict) and "contents" in final_llm_payload:
        func_logger.info(f"[{session_id}] Final payload constructed successfully ({len(final_llm_payload['contents'])} turns).")
        await _emit_status("Status: Payload construction complete.")
        # Return payload, status, and calculated context parts
        return final_llm_payload["contents"], context_status_info, final_base_prompt, final_t0_history, final_combined_context
    else:
        context_status_info["error"] = "Payload construction returned unexpected format."
        func_logger.error(f"[{session_id}] Payload construction failed: Unexpected format {type(final_llm_payload)}.")
        # Return None payload, status, and calculated context parts
        return None, context_status_info, final_base_prompt, final_t0_history, final_combined_context

# [[END MODIFIED context_processor.py - Use Snapshot v0.2.4]]