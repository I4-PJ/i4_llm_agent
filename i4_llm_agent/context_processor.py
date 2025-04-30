# === START MODIFIED FILE: i4_llm_agent/context_processor.py ===
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
# <<< Import specific DB functions needed >>>
from . import database # Import the whole module to access functions
from .history import select_turns_for_t0, format_history_for_llm, DIALOGUE_ROLES
from .prompting import (
    process_system_prompt, combine_background_context, construct_final_llm_payload,
    refine_external_context, # Stateless refinement orchestrator
    generate_rag_query, # T2 query generation
    # Default templates used by refinement/cache functions
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
)
# <<< Import cache functions directly >>>
from .cache import update_rag_cache, select_final_context # RAG Cache orchestrators
from .utils import TIKTOKEN_AVAILABLE, count_tokens # Token counting

# Initialize logger for this module
logger = logging.getLogger(__name__) # i4_llm_agent.context_processor


async def process_context_and_prepare_payload(
    session_id: str,
    body: Dict,
    user_valves: Any, # Pipe.UserValves object
    current_active_history: List[Dict],
    history_for_processing: List[Dict], # History slice before query
    latest_user_query_str: str,
    # --- State Info (Passed from Orchestrator) ---
    current_scene_state_dict: Dict[str, Any],
    current_world_state_dict: Dict[str, Any],
    generated_event_hint_text: Optional[str],
    generated_weather_proposal: Optional[Dict[str, Optional[str]]],
    # --- Config & Utilities (Passed from Orchestrator) ---
    config: Any, # Pipe.Valves object
    logger: logging.Logger,
    sqlite_cursor: Optional[sqlite3.Cursor],
    chroma_client: Optional[Any], # ChromaDB client instance
    chroma_embed_wrapper: Optional[Any], # ChromaDB embedding function wrapper (used for collection creation)
    embedding_func: Optional[Callable[[Sequence[str]], List[List[float]]]], # OWI Embedding function (takes list, returns list of lists)
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # Async LLM wrapper
    tokenizer: Optional[Callable], # Tiktoken tokenizer instance
    event_emitter: Optional[Callable],
    orchestrator_debug_path_getter: Optional[Callable[[str], Optional[str]]],
    dialogue_roles: List[str],
    session_period_setting: Optional[str] = None,
    # --- Database Function Aliases (Passed from Orchestrator) ---
    db_get_recent_t1_summaries_func: Optional[Callable[..., Coroutine[Any, Any, List[Tuple[str, Dict]]]]] = None,
    db_get_recent_aged_summaries_func: Optional[Callable[..., Coroutine[Any, Any, List[Tuple[str, Dict]]]]] = None,
) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
    """
    Processes context sources (OWI RAG, T1/Aged Summaries, T2 RAG, Inventory, State)
    and prepares the final LLM payload ('contents' format).

    Handles:
    - T0 history slicing.
    - OWI RAG processing (conditional).
    - Stateless Refinement (conditional).
    - RAG Cache (conditional).
    - T2 RAG query generation and execution (conditional).
    - Fetching T1 and Aged summaries via DB functions.
    - Formatting inventory context.
    - Combining all context sources.
    - Constructing the final Gemini payload.

    Returns:
        Tuple[Optional[List[Dict]], Dict[str, Any]]:
            - The final LLM payload contents list (or None on error).
            - A dictionary containing status information about context processing.
    """
    func_logger = logger
    func_logger.debug(f"[{session_id}] ContextProcessor: Entered process_context_and_prepare_payload.")

    context_status_info = {
        "t1_retrieved_count": 0,
        "aged_retrieved_count": 0,
        "t2_retrieved_count": 0,
        "t0_dialogue_tokens": -1,
        "initial_owi_context_tokens": -1,
        "refined_context_tokens": -1,
        "final_context_selection_performed": False,
        "stateless_refinement_performed": False,
        "error": None,
    }

    # --- Helper Functions ---
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

    # --- Prerequisites Check ---
    if not sqlite_cursor:
        context_status_info["error"] = "SQLite cursor unavailable."
        func_logger.critical(f"[{session_id}] ContextProcessor CRITICAL: SQLite cursor unavailable.")
        return None, context_status_info
    if not db_get_recent_t1_summaries_func:
        func_logger.warning(f"[{session_id}] ContextProcessor Warning: db_get_recent_t1_summaries_func not provided.")
    if not db_get_recent_aged_summaries_func:
        func_logger.warning(f"[{session_id}] ContextProcessor Warning: db_get_recent_aged_summaries_func not provided.")


    # --- 1. Select T0 History Slice ---
    await _emit_status("Status: Selecting dialogue history...")
    t0_active_history_token_limit = getattr(config, 't0_active_history_token_limit', 4000)
    # Call should now work after history.py fix
    t0_history_slice, t0_dialogue_tokens, _ = select_turns_for_t0(
        history_for_processing, t0_active_history_token_limit, tokenizer, dialogue_roles
    )
    context_status_info["t0_dialogue_tokens"] = t0_dialogue_tokens
    func_logger.info(f"[{session_id}] T0 History: Selected {len(t0_history_slice)} turns ({t0_dialogue_tokens} tokens).")

    # --- 2. Process OWI RAG & Apply Refinements ---
    await _emit_status("Status: Processing OWI context...")
    raw_owi_system_prompt = ""
    for msg in body.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            raw_owi_system_prompt = msg.get("content", "")
            break
    base_system_prompt_text, extracted_owi_context_str = process_system_prompt([{"role": "system", "content": raw_owi_system_prompt}])
    initial_owi_context_tokens = _count_tokens_safe(extracted_owi_context_str)
    context_status_info["initial_owi_context_tokens"] = initial_owi_context_tokens
    func_logger.info(f"[{session_id}] OWI Context: Extracted {initial_owi_context_tokens} tokens.")

    # --- Apply User Valve Text Removal ---
    text_to_remove = getattr(user_valves, 'text_block_to_remove', '').strip()
    if text_to_remove and extracted_owi_context_str:
        original_len = len(extracted_owi_context_str)
        extracted_owi_context_str = extracted_owi_context_str.replace(text_to_remove, "")
        if len(extracted_owi_context_str) < original_len:
            func_logger.info(f"[{session_id}] Removed specified text block from OWI context.")
        else:
            func_logger.warning(f"[{session_id}] Specified text block to remove was not found in OWI context.")

    processed_owi_rag_context = extracted_owi_context_str # Start with potentially modified OWI context
    process_owi_rag_flag = getattr(user_valves, 'process_owi_rag', True)
    if not process_owi_rag_flag:
        func_logger.info(f"[{session_id}] Skipping OWI RAG processing as per user valve.")
        processed_owi_rag_context = None # Effectively disable OWI context if flag is false

    # --- Apply Refinements (Stateless or Cache) ---
    final_selected_context: Optional[str] = None
    refined_context_tokens = -1 # Default if refinement doesn't run or fails

    # --- Refinement Configuration ---
    enable_rag_cache = getattr(config, 'enable_rag_cache', False)
    enable_stateless_refinement = getattr(config, 'enable_stateless_refinement', False)
    refiner_url = getattr(config, 'refiner_llm_api_url', None); refiner_key = getattr(config, 'refiner_llm_api_key', None)
    refiner_temp = getattr(config, 'refiner_llm_temperature', 0.3)
    refiner_history_count = getattr(config, 'refiner_history_count', 6)
    stateless_skip_threshold = getattr(config, 'stateless_refiner_skip_threshold', 500)
    # RAG Cache specific config (now used by update_rag_cache directly)
    # cache_update_skip_owi_threshold = getattr(config, 'CACHE_UPDATE_SKIP_OWI_THRESHOLD', 50)
    # cache_update_similarity_threshold = getattr(config, 'CACHE_UPDATE_SIMILARITY_THRESHOLD', 0.9)

    refiner_llm_config = { "url": refiner_url, "key": refiner_key, "temp": refiner_temp, "prompt_template": DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE }
    cache_update_llm_config = { "url": refiner_url, "key": refiner_key, "temp": refiner_temp, "prompt_template": DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT }
    final_select_llm_config = { "url": refiner_url, "key": refiner_key, "temp": refiner_temp, "prompt_template": DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT }

    # --- Select Refinement Strategy ---
    if enable_rag_cache and processed_owi_rag_context and refiner_url and refiner_key:
        func_logger.info(f"[{session_id}] Applying RAG Cache (Two-Step Refinement)...")
        await _emit_status("Status: Applying RAG cache...")
        try:
            # Step 1: Update Cache
            # <<< MODIFIED CALL: Removed previous_cache argument >>>
            updated_cache_text = await update_rag_cache(
                session_id=session_id,
                current_owi_context=processed_owi_rag_context,
                history_messages=history_for_processing,
                latest_user_query=latest_user_query_str,
                llm_call_func=llm_call_func,
                sqlite_cursor=sqlite_cursor, # Pass cursor here
                cache_update_llm_config=cache_update_llm_config,
                history_count=refiner_history_count, # Pass history count needed by update_rag_cache
                dialogue_only_roles=dialogue_roles,
                caller_info=f"CtxProc_CacheUpdate_{session_id}"
            )

            # Step 2: Select Final Context (Pass debug path getter)
            final_selected_context = await select_final_context(
                updated_cache_text=updated_cache_text,
                current_owi_context=processed_owi_rag_context,
                history_messages=history_for_processing,
                latest_user_query=latest_user_query_str,
                llm_call_func=llm_call_func,
                context_selection_llm_config=final_select_llm_config,
                history_count=refiner_history_count, # Pass history count needed by select_final_context
                dialogue_only_roles=dialogue_roles,
                caller_info=f"CtxProc_CtxSelect_{session_id}",
                debug_log_path_getter=orchestrator_debug_path_getter # Pass the getter
            )
            context_status_info["final_context_selection_performed"] = True
            refined_context_tokens = _count_tokens_safe(final_selected_context)
            func_logger.info(f"[{session_id}] RAG Cache processing complete. Final context tokens: {refined_context_tokens}")
        except Exception as e_cache:
            func_logger.error(f"[{session_id}] Error during RAG Cache processing: {e_cache}", exc_info=True)
            context_status_info["error"] = f"RAG Cache Error: {e_cache}" # Add error info
            final_selected_context = processed_owi_rag_context # Fallback to processed OWI
            refined_context_tokens = _count_tokens_safe(final_selected_context)

    elif enable_stateless_refinement and processed_owi_rag_context and refiner_url and refiner_key:
        func_logger.info(f"[{session_id}] Applying Stateless Refinement...")
        await _emit_status("Status: Applying stateless refinement...")
        try:
            # Ensure refiner_llm_config has the stateless template
            refiner_llm_config['prompt_template'] = DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE
            final_selected_context = await refine_external_context(
                external_context=processed_owi_rag_context, history_messages=history_for_processing,
                latest_user_query=latest_user_query_str, llm_call_func=llm_call_func,
                refiner_llm_config=refiner_llm_config, skip_threshold=stateless_skip_threshold,
                history_count=refiner_history_count, dialogue_only_roles=dialogue_roles,
                caller_info=f"CtxProc_Stateless_{session_id}"
            )
            context_status_info["stateless_refinement_performed"] = True
            refined_context_tokens = _count_tokens_safe(final_selected_context)
            func_logger.info(f"[{session_id}] Stateless refinement complete. Final context tokens: {refined_context_tokens}")
        except Exception as e_stateless:
            func_logger.error(f"[{session_id}] Error during Stateless Refinement: {e_stateless}", exc_info=True)
            context_status_info["error"] = f"Stateless Refinement Error: {e_stateless}" # Add error info
            final_selected_context = processed_owi_rag_context # Fallback
            refined_context_tokens = _count_tokens_safe(final_selected_context)

    else:
        func_logger.info(f"[{session_id}] No refinement applied. Using processed OWI RAG context directly.")
        final_selected_context = processed_owi_rag_context
        refined_context_tokens = _count_tokens_safe(final_selected_context)

    context_status_info["refined_context_tokens"] = refined_context_tokens

    # --- 3. Fetch T1 and Aged Summaries (via DB functions) ---
    await _emit_status("Status: Fetching recent summaries...")
    recent_t1_summaries_data: List[Tuple[str, Dict]] = []
    recent_aged_summaries_data: List[Tuple[str, Dict]] = []
    max_t1_blocks = getattr(config, 'max_stored_summary_blocks', 20) # Use same limit as orchestrator for fetch

    if db_get_recent_t1_summaries_func:
        try:
            recent_t1_summaries_data = await db_get_recent_t1_summaries_func(
                cursor=sqlite_cursor, session_id=session_id, limit=max_t1_blocks
            )
            context_status_info["t1_retrieved_count"] = len(recent_t1_summaries_data)
            func_logger.info(f"[{session_id}] Fetched {len(recent_t1_summaries_data)} T1 summaries via DB func.")
        except Exception as e_get_t1:
            func_logger.error(f"[{session_id}] Error fetching T1 summaries via DB func: {e_get_t1}", exc_info=True)
            context_status_info["t1_retrieved_count"] = 0
    else:
         func_logger.warning(f"[{session_id}] Cannot fetch T1 summaries: DB function alias missing.")

    if db_get_recent_aged_summaries_func:
        try:
            # Determine a reasonable limit for aged summaries, e.g., half of T1 limit?
            aged_limit = max(1, max_t1_blocks // 2)
            recent_aged_summaries_data = await db_get_recent_aged_summaries_func(
                cursor=sqlite_cursor, session_id=session_id, limit=aged_limit
            )
            context_status_info["aged_retrieved_count"] = len(recent_aged_summaries_data)
            func_logger.info(f"[{session_id}] Fetched {len(recent_aged_summaries_data)} Aged summaries via DB func.")
        except Exception as e_get_aged:
            # Check if it's the missing table error specifically
            if "no such table: aged_summaries" in str(e_get_aged):
                 func_logger.error(f"[{session_id}] CRITICAL DB Error: Table 'aged_summaries' not found. Please ensure it is created in the database.")
                 context_status_info["error"] = "Missing 'aged_summaries' table in database."
            else:
                 func_logger.error(f"[{session_id}] Error fetching Aged summaries via DB func: {e_get_aged}", exc_info=True)
            context_status_info["aged_retrieved_count"] = 0
    else:
         func_logger.warning(f"[{session_id}] Cannot fetch Aged summaries: DB function alias missing.")


    # --- 4. T2 RAG Query & Retrieval ---
    await _emit_status("Status: Checking long-term memory...")
    t2_rag_results: Optional[List[str]] = None # Expecting list of strings now
    ragq_llm_url = getattr(config, 'ragq_llm_api_url', None); ragq_llm_key = getattr(config, 'ragq_llm_api_key', None)
    # <<< MODIFIED can_rag check: Needs chroma_client, chroma_embed_wrapper (for collection), embedding_func (for query) >>>
    can_rag = all([ database.CHROMADB_AVAILABLE, chroma_client, chroma_embed_wrapper, embedding_func, ragq_llm_url, ragq_llm_key, latest_user_query_str ])
    if can_rag:
        func_logger.info(f"[{session_id}] T2 RAG: Performing query...")
        await _emit_status("Status: Querying long-term memory...")
        try:
            # Generate Query
            t0_history_str = format_history_for_llm(t0_history_slice) # Use T0 slice for RAGQ context
            rag_query = await generate_rag_query(
                latest_message_str=latest_user_query_str, dialogue_context_str=t0_history_str,
                llm_call_func=llm_call_func, api_url=ragq_llm_url, api_key=ragq_llm_key,
                temperature=getattr(config, 'ragq_llm_temperature', 0.3),
                caller_info=f"CtxProc_RAGQ_{session_id}"
            )
            if not rag_query or rag_query.startswith("[Error"):
                func_logger.warning(f"[{session_id}] T2 RAG: Query generation failed: {rag_query}")
            else:
                # <<< MODIFIED T2 RAG Block START >>>
                rag_results_count = getattr(config, 'rag_summary_results_count', 3)
                base_prefix = getattr(config, 'summary_collection_prefix', 'sm_t2_')
                safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]
                tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]

                # 1. Get Chroma Collection Object
                tier2_collection = await database.get_or_create_chroma_collection(
                    chroma_client, tier2_collection_name, chroma_embed_wrapper # Pass wrapper here for creation
                )

                if not tier2_collection:
                    func_logger.error(f"[{session_id}] T2 RAG: Failed to get/create collection '{tier2_collection_name}'.")
                else:
                    # 2. Get Query Embedding
                    query_embedding_list = None
                    try:
                        # Assuming embedding_func is sync, needs thread. Takes list[str], returns list[list[float]]
                        embedding_result = await asyncio.to_thread(embedding_func, [rag_query])
                        if isinstance(embedding_result, list) and len(embedding_result) == 1 and isinstance(embedding_result[0], list):
                            query_embedding_list = embedding_result
                            func_logger.debug(f"[{session_id}] T2 RAG: Successfully generated query embedding.")
                        else:
                             func_logger.error(f"[{session_id}] T2 RAG: Embedding function returned unexpected format: {type(embedding_result)}")
                    except Exception as e_embed:
                        func_logger.error(f"[{session_id}] T2 RAG: Exception during query embedding: {e_embed}", exc_info=True)

                    if query_embedding_list:
                        # 3. Call query_chroma_collection with correct args
                        chroma_results_dict = await database.query_chroma_collection(
                            collection=tier2_collection, # Pass collection object
                            query_embeddings=query_embedding_list, # Pass embedding list
                            n_results=rag_results_count,
                            include=["documents", "distances", "metadatas"] # Keep include standard
                        )

                        if chroma_results_dict and isinstance(chroma_results_dict.get('documents'), list):
                            t2_rag_results = chroma_results_dict['documents'][0] # Documents is list of lists
                            if isinstance(t2_rag_results, list): # Ensure it's a list of strings
                                context_status_info["t2_retrieved_count"] = len(t2_rag_results)
                                func_logger.info(f"[{session_id}] T2 RAG: Retrieved {len(t2_rag_results)} results.")
                            else:
                                func_logger.error(f"[{session_id}] T2 RAG: Expected list of documents, got {type(t2_rag_results)}. Resetting results.")
                                t2_rag_results = None
                        else:
                            func_logger.info(f"[{session_id}] T2 RAG: No results found or invalid format from query for '{rag_query}'.")
                            t2_rag_results = None
                # <<< MODIFIED T2 RAG Block END >>>

        except Exception as e_rag:
            func_logger.error(f"[{session_id}] T2 RAG: Error during query/retrieval: {e_rag}", exc_info=True)
            context_status_info["error"] = f"T2 RAG Error: {e_rag}" # Add error info
    else:
        missing_rag_prereqs = [p for p, v in {"chroma_lib": database.CHROMADB_AVAILABLE, "chroma_client": chroma_client, "wrapper": chroma_embed_wrapper, "embed_f": embedding_func, "url": ragq_llm_url, "key": ragq_llm_key, "query": bool(latest_user_query_str)}.items() if not v]
        func_logger.debug(f"[{session_id}] T2 RAG: Skipping due to missing prerequisites: {', '.join(missing_rag_prereqs)}")


    # --- 5. Format Inventory Context ---
    await _emit_status("Status: Formatting inventory context...")
    inventory_context_str: Optional[str] = None
    inventory_enabled = getattr(config, 'enable_inventory_management', False)
    if inventory_enabled: # Check module availability later
        try:
            # Try importing formatting/DB functions specific to this step
            from .inventory import format_inventory_for_prompt as format_inv_func
            # from .database import get_all_inventories_for_session # Use database. prefix now
            all_inventories = await database.get_all_inventories_for_session(sqlite_cursor, session_id)
            inventory_context_str = format_inv_func(all_inventories)
            func_logger.info(f"[{session_id}] Inventory context formatted (Length: {len(inventory_context_str or '')}).")
        except ImportError:
            func_logger.error(f"[{session_id}] Failed to import inventory formatting/DB functions.")
            inventory_context_str = "[Inventory Module/DB Error]"
        except Exception as e_inv_fmt:
            func_logger.error(f"[{session_id}] Error formatting inventory: {e_inv_fmt}", exc_info=True)
            inventory_context_str = "[Inventory Formatting Error]"
    else:
        inventory_context_str = "[Inventory Disabled]"
        func_logger.debug(f"[{session_id}] Inventory disabled by config.")


    # --- 6. Combine All Background Context ---
    await _emit_status("Status: Combining background context...")
    # Use pre-assessed state passed from orchestrator
    scene_desc_for_combine = current_scene_state_dict.get('description')
    current_day_for_combine = current_world_state_dict.get('day')
    current_time_for_combine = current_world_state_dict.get('time_of_day')
    current_season_for_combine = current_world_state_dict.get('season')
    current_weather_for_combine = current_world_state_dict.get('weather')

    combined_background_context = combine_background_context(
        final_selected_context=final_selected_context,
        t1_summaries=recent_t1_summaries_data, # Pass list of tuples
        aged_summaries=recent_aged_summaries_data, # Pass list of tuples
        t2_rag_results=t2_rag_results, # Pass list of strings
        scene_description=scene_desc_for_combine,
        inventory_context=inventory_context_str,
        current_day=current_day_for_combine,
        current_time_of_day=current_time_for_combine,
        current_season=current_season_for_combine,
        current_weather=current_weather_for_combine,
        weather_proposal=generated_weather_proposal,
    )
    func_logger.info(f"[{session_id}] Combined background context length: {len(combined_background_context)}")

    # --- 7. Construct Final LLM Payload ---
    await _emit_status("Status: Constructing final payload...")
    final_llm_payload = construct_final_llm_payload(
        system_prompt=base_system_prompt_text,
        history=t0_history_slice, # Use T0 slice
        context=combined_background_context,
        query=latest_user_query_str,
        long_term_goal=getattr(user_valves, 'long_term_goal', None),
        event_hint=generated_event_hint_text,
        period_setting=session_period_setting, # Pass period setting
        include_ack_turns=getattr(config, 'include_ack_turns', True)
    )

    if isinstance(final_llm_payload, dict) and "error" in final_llm_payload:
        context_status_info["error"] = final_llm_payload["error"]
        func_logger.error(f"[{session_id}] Payload construction failed: {final_llm_payload['error']}")
        return None, context_status_info
    elif isinstance(final_llm_payload, dict) and "contents" in final_llm_payload:
        func_logger.info(f"[{session_id}] Final payload constructed successfully ({len(final_llm_payload['contents'])} turns).")
        await _emit_status("Status: Payload construction complete.")
        return final_llm_payload["contents"], context_status_info
    else:
        context_status_info["error"] = "Payload construction returned unexpected format."
        func_logger.error(f"[{session_id}] Payload construction failed: Unexpected format {type(final_llm_payload)}.")
        return None, context_status_info

# === END MODIFIED FILE: i4_llm_agent/context_processor.py ===