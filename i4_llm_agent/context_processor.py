# === START OF FILE i4_llm_agent/context_processor.py ===
# i4_llm_agent/context_processor.py

"""
Handles processing of background context (T2 RAG, OWI Context Refinement, Inventory)
and construction of the final LLM payload.
"""

import logging
import asyncio
import re
import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import (
    Tuple, Union, List, Dict, Optional, Any, Callable, Coroutine, Sequence
)

# --- Library Imports ---
# (Imports remain the same as the previous version of this file)
from .database import (
    get_rag_cache,
    get_all_inventories_for_session,
    get_or_create_chroma_collection,
    add_to_chroma_collection,
    query_chroma_collection,
    get_chroma_collection_count,
    CHROMADB_AVAILABLE, ChromaEmbeddingFunction, ChromaCollectionType,
    InvalidDimensionException,
)
from .history import (
    format_history_for_llm, get_recent_turns # DIALOGUE_ROLES passed in
)
from .cache import update_rag_cache, select_final_context
from .api_client import call_google_llm_api # Potentially used if not passed directly

try:
    from .utils import TIKTOKEN_AVAILABLE, count_tokens, calculate_string_similarity
except ImportError:
    TIKTOKEN_AVAILABLE = False
    def count_tokens(*args, **kwargs): return 0
    def calculate_string_similarity(*args, **kwargs): return 0.0
    logging.getLogger(__name__).warning("ContextProcessor: Failed to import utils (tiktoken?). Token counting/similarity may be affected.")

from .prompting import (
    # format functions
    format_inventory_update_prompt,
    format_stateless_refiner_prompt,
    format_cache_update_prompt,
    format_final_context_selection_prompt,
    # standalone functions
    construct_final_llm_payload,
    clean_context_tags, generate_rag_query,
    combine_background_context, process_system_prompt,
    refine_external_context,
    # Default templates
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
    DEFAULT_RAGQ_LLM_PROMPT,
)

# Inventory Module Import
try:
    from .inventory import (
        format_inventory_for_prompt as _real_format_inventory_func,
    )
    _INVENTORY_MODULE_AVAILABLE = True
    _dummy_format_inventory = None
except ImportError:
    _INVENTORY_MODULE_AVAILABLE = False
    _real_format_inventory_func = None
    def _dummy_format_inventory(*args, **kwargs): return "[Inventory Module Unavailable]"
    logging.getLogger(__name__).warning(
        "ContextProcessor: Inventory module not found. Inventory formatting disabled."
        )


# --- Internal Helper Functions (Moved/Adapted from Orchestrator) ---
# _get_t2_rag_results_internal, _prepare_and_refine_background_internal, _emit_status_internal
# (Implementations remain the same as the previous version of this file)

async def _get_t2_rag_results_internal(
    session_id: str,
    history_for_processing: List[Dict],
    latest_user_query_str: str,
    config: object,
    logger: logging.Logger,
    chroma_client: Optional[Any],
    chroma_embed_wrapper: Optional[Any],
    embedding_func: Optional[Callable[[Sequence[str], str, Optional[Dict]], List[List[float]]]], # More specific type hint
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    event_emitter: Optional[Callable],
    dialogue_roles: List[str]
) -> Tuple[List[str], int]:
    """ Internal helper to retrieve T2 RAG results. (Implementation unchanged) """
    # [[ Implementation from previous correct version ]]
    await _emit_status_internal(event_emitter, session_id, logger, "Status: Searching long-term memory...")
    retrieved_rag_summaries = []; t2_retrieved_count = 0; tier2_collection = None;
    n_results_t2 = getattr(config, 'rag_summary_results_count', 0)
    ragq_url = getattr(config, 'ragq_llm_api_url', None)
    ragq_key = getattr(config, 'ragq_llm_api_key', None)
    ragq_temp = getattr(config, 'ragq_llm_temperature', 0.3)
    _generate_rag_query_func = generate_rag_query

    can_rag = all([
        chroma_client is not None, chroma_embed_wrapper is not None, latest_user_query_str,
        embedding_func is not None, _generate_rag_query_func is not None, llm_call_func is not None,
        ragq_url, ragq_key, n_results_t2 > 0
    ])

    if not can_rag:
         missing_prereqs = [p for p, v in {
             "chroma": chroma_client is not None, "chroma_wrapper": chroma_embed_wrapper is not None,
             "query": latest_user_query_str, "embed_func": embedding_func is not None,
             "gen_ragq_func": _generate_rag_query_func is not None, "llm_call_func": llm_call_func is not None,
             "ragq_url": ragq_url, "ragq_key": ragq_key, "n_results": n_results_t2 > 0
             }.items() if not v]
         logger.debug(f"[{session_id}] Skipping T2 RAG check: Prerequisites not met: {', '.join(missing_prereqs)} (RAG Results Count: {n_results_t2}).")
         return [], 0

    try: base_prefix = getattr(config, 'summary_collection_prefix', 'sm_t2_'); safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[:50]; tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
    except Exception as e_name: logger.error(f"[{session_id}] Error creating T2 collection name: {e_name}"); return [], 0
    try: tier2_collection = await get_or_create_chroma_collection(chroma_client, tier2_collection_name, chroma_embed_wrapper)
    except Exception as e_get_coll_rag: logger.error(f"[{session_id}] Error getting T2 collection for RAG: {e_get_coll_rag}. Skipping.", exc_info=True); return [], 0
    if not tier2_collection: logger.error(f"[{session_id}] Failed get/create T2 collection '{tier2_collection_name}'. Skipping RAG."); return [], 0
    try: t2_doc_count = await get_chroma_collection_count(tier2_collection)
    except Exception as e_count: logger.error(f"[{session_id}] Error checking T2 collection count: {e_count}. Skipping RAG.", exc_info=True); return [], 0
    if t2_doc_count <= 0: logger.debug(f"[{session_id}] Skipping T2 RAG: Collection '{tier2_collection.name}' is empty ({t2_doc_count})."); return [], 0

    try:
        await _emit_status_internal(event_emitter, session_id, logger, "Status: Generating search query...")
        context_messages_for_ragq = get_recent_turns( history_for_processing, count=getattr(config, 'refiner_history_count', 6), exclude_last=False, roles=dialogue_roles)
        dialogue_context_str = format_history_for_llm(context_messages_for_ragq) if context_messages_for_ragq else "[No recent history]"

        logger.debug(f"[{session_id}] ContextProc: Calling generate_rag_query (uses library default prompt)...")
        rag_query = await _generate_rag_query_func(
             latest_message_str=latest_user_query_str, dialogue_context_str=dialogue_context_str,
             llm_call_func=llm_call_func, api_url=ragq_url, api_key=ragq_key,
             temperature=ragq_temp, caller_info=f"CtxProc_RAGQ_{session_id}",
        )

        if not (rag_query and isinstance(rag_query, str) and not rag_query.startswith("[Error:") and rag_query.strip()): logger.error(f"[{session_id}] RAG Query Generation failed: '{rag_query}'. Skipping RAG."); return [], 0
        logger.debug(f"[{session_id}] Generated RAG Query: '{rag_query[:100]}...'")

        await _emit_status_internal(event_emitter, session_id, logger, "Status: Embedding search query...")
        query_embedding = None; query_embedding_successful = False
        try:
            if not callable(embedding_func): logger.error(f"[{session_id}] Cannot embed RAG query: OWI Embedding function invalid."); return [], 0
            from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX
            query_embedding_list = await asyncio.to_thread(embedding_func, [rag_query], prefix=RAG_EMBEDDING_QUERY_PREFIX)
            if isinstance(query_embedding_list, list) and len(query_embedding_list) == 1 and isinstance(query_embedding_list[0], list) and len(query_embedding_list[0]) > 0:
                query_embedding = query_embedding_list[0]; query_embedding_successful = True; logger.debug(f"[{session_id}] RAG query embedding successful (dim: {len(query_embedding)}).")
            else: logger.error(f"[{session_id}] RAG query embed invalid structure: {type(query_embedding_list)}.")
        except Exception as embed_e: logger.error(f"[{session_id}] EXCEPTION during RAG query embedding: {embed_e}", exc_info=True)
        if not (query_embedding_successful and query_embedding): logger.error(f"[{session_id}] Skipping T2 ChromaDB query: RAG query embedding failed."); return [], 0

        await _emit_status_internal(event_emitter, session_id, logger, f"Status: Searching vector store (top {n_results_t2})...")
        rag_results_dict = await query_chroma_collection( tier2_collection, query_embeddings=[query_embedding], n_results=n_results_t2, include=["documents", "distances", "metadatas"])
        if rag_results_dict and isinstance(rag_results_dict.get("documents"), list) and rag_results_dict["documents"] and isinstance(rag_results_dict["documents"][0], list):
              retrieved_docs = rag_results_dict["documents"][0]
              if retrieved_docs:
                   retrieved_rag_summaries = retrieved_docs; t2_retrieved_count = len(retrieved_docs)
                   distances = rag_results_dict.get("distances", [[None]])[0]; ids = rag_results_dict.get("ids", [["N/A"]])[0]; dist_str = [f"{d:.4f}" for d in distances if d is not None]
                   logger.info(f"[{session_id}] Retrieved {t2_retrieved_count} docs from T2 RAG. IDs: {ids}, Dist: {dist_str}")
              else: logger.info(f"[{session_id}] T2 RAG query executed but returned no documents.")
        else: logger.info(f"[{session_id}] T2 RAG query returned no matches or unexpected structure: {type(rag_results_dict)}")
    except Exception as e_rag_outer: logger.error(f"[{session_id}] Unexpected error during outer T2 RAG processing: {e_rag_outer}", exc_info=True); retrieved_rag_summaries = []; t2_retrieved_count = 0
    return retrieved_rag_summaries, t2_retrieved_count

async def _prepare_and_refine_background_internal(
    session_id: str,
    body: Dict,
    user_valves: Any,
    retrieved_t1_summaries: List[str],
    retrieved_rag_summaries: List[str], # T2 results from _get_t2_rag_results_internal
    current_active_history: List[Dict], # Full history needed for cache/refinement
    latest_user_query_str: str,
    config: object,
    logger: logging.Logger,
    sqlite_cursor: sqlite3.Cursor,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    tokenizer: Optional[Any],
    event_emitter: Optional[Callable],
    dialogue_roles: List[str],
    orchestrator_debug_path_getter: Optional[Callable[[str], Optional[str]]] = None
) -> Tuple[str, str, int, int, bool, bool, bool, bool, str]:
    """ Internal helper to prepare base prompt, format inventory, and refine context. (Implementation unchanged) """
    # [[ Implementation from previous correct version ]]
    await _emit_status_internal(event_emitter, session_id, logger, "Status: Preparing context...")
    base_system_prompt_text = "You are helpful."; extracted_owi_context = None; initial_owi_context_tokens = -1; current_output_messages = body.get("messages", [])

    logger.debug("Step 1: Processing system prompt...")
    if process_system_prompt: # Use imported function
         try: base_system_prompt_text, extracted_owi_context = process_system_prompt(current_output_messages)
         except Exception as e_proc_sys: logger.error(f"[{session_id}] Error process_system_prompt: {e_proc_sys}.", exc_info=True); extracted_owi_context = None
    else: logger.error(f"[{session_id}] process_system_prompt unavailable."); base_system_prompt_text = "You are helpful."

    if extracted_owi_context and count_tokens and tokenizer:
         try: initial_owi_context_tokens = count_tokens(extracted_owi_context, tokenizer)
         except Exception: initial_owi_context_tokens = -1
    elif not extracted_owi_context: logger.debug(f"[{session_id}] No OWI <context> tag found.")
    logger.debug(f"Extracted OWI Context Length: {len(extracted_owi_context) if extracted_owi_context else 0}, Tokens: {initial_owi_context_tokens}")

    if not base_system_prompt_text: base_system_prompt_text = "You are helpful."; logger.warning(f"[{session_id}] System prompt empty after clean. Using default.")

    session_text_block_to_remove = getattr(user_valves, 'text_block_to_remove', '') if user_valves else ''
    if session_text_block_to_remove:
        original_len = len(base_system_prompt_text); temp_prompt = base_system_prompt_text.replace(session_text_block_to_remove, "")
        if len(temp_prompt) < original_len: base_system_prompt_text = temp_prompt; logger.debug(f"Removed text block from system prompt ({original_len - len(temp_prompt)} chars).")
        else: logger.debug(f"Text block for removal NOT FOUND: '{session_text_block_to_remove[:50]}...'")

    session_process_owi_rag = bool(getattr(user_valves, 'process_owi_rag', True))
    if not session_process_owi_rag:
        logger.debug("Session valve 'process_owi_rag=False'. Discarding OWI context.")
        extracted_owi_context = None; initial_owi_context_tokens = 0

    logger.debug("Step 2: Fetching and formatting inventory...")
    formatted_inventory_string = "[Inventory Management Disabled]"; raw_session_inventories = {};
    inventory_enabled = getattr(config, 'enable_inventory_management', False)
    _format_inventory_func = _real_format_inventory_func if _INVENTORY_MODULE_AVAILABLE else _dummy_format_inventory

    if inventory_enabled and _INVENTORY_MODULE_AVAILABLE and get_all_inventories_for_session and _format_inventory_func and sqlite_cursor:
        logger.debug("Inventory enabled, fetching data...")
        try:
            raw_session_inventories = await get_all_inventories_for_session(sqlite_cursor, session_id)
            if raw_session_inventories:
                logger.debug(f"Retrieved inventory data for {len(raw_session_inventories)} characters.")
                try: formatted_inventory_string = _format_inventory_func(raw_session_inventories); logger.debug(f"Formatted inventory string generated (len: {len(formatted_inventory_string)}).")
                except Exception as e_fmt_inv: logger.error(f"[{session_id}] Failed to format inventory string: {e_fmt_inv}", exc_info=True); formatted_inventory_string = "[Error Formatting Inventory]"; logger.error(f"ERROR formatting inventory string: {e_fmt_inv}")
            else: logger.debug("No inventory data found in DB for this session."); formatted_inventory_string = "[No Inventory Data Available]"
        except Exception as e_get_inv: logger.error(f"[{session_id}] Error retrieving inventory data from DB: {e_get_inv}", exc_info=True); formatted_inventory_string = "[Error Retrieving Inventory]"; logger.error(f"ERROR retrieving inventory data from DB: {e_get_inv}")
    elif not inventory_enabled: logger.debug("Skipping inventory fetch: Feature disabled by global valve.")
    elif inventory_enabled and not _INVENTORY_MODULE_AVAILABLE: logger.debug("Skipping inventory fetch: Module unavailable (Import failed).")
    else: missing_inv_funcs = [f for f, fn in {"db_get": get_all_inventories_for_session, "formatter": _format_inventory_func, "cursor": sqlite_cursor}.items() if not fn]; logger.debug(f"Skipping inventory fetch: Missing prerequisites: {missing_inv_funcs}"); formatted_inventory_string = "[Inventory Init/Config Error]"

    logger.debug("Step 3: Context Refinement Logic...")
    refined_context_str = extracted_owi_context or "";
    refined_context_tokens = initial_owi_context_tokens;
    cache_update_performed = False; cache_update_skipped = False; final_context_selection_performed = False; stateless_refinement_performed = False; updated_cache_text_intermediate = "[Cache not initialized or updated]"
    enable_rag_cache_global = getattr(config, 'enable_rag_cache', False); enable_stateless_refin_global = getattr(config, 'enable_stateless_refinement', False)
    logger.debug(f"RAG Cache Enabled: {enable_rag_cache_global}, Stateless Refinement Enabled: {enable_stateless_refin_global}")

    if enable_rag_cache_global and update_rag_cache and select_final_context and get_rag_cache and sqlite_cursor:
        logger.debug("RAG Cache Path Selected.")
        run_step1 = False; previous_cache_text = "";
        try: cache_result = await get_rag_cache(sqlite_cursor, session_id); previous_cache_text = cache_result if cache_result is not None else ""
        except Exception as e_get_cache: logger.error(f"[{session_id}] Error retrieving previous cache: {e_get_cache}", exc_info=True)
        logger.debug(f"Previous cache length: {len(previous_cache_text)}")

        if not session_process_owi_rag: logger.debug("Skipping RAG Cache Step 1 (session valve 'process_owi_rag=False')."); cache_update_skipped = True; run_step1 = False; updated_cache_text_intermediate = previous_cache_text
        else:
             skip_len = False
             skip_sim = False
             owi_content_for_check = extracted_owi_context or "";
             len_thresh = getattr(config, 'CACHE_UPDATE_SKIP_OWI_THRESHOLD', 50)

             if len(owi_content_for_check.strip()) < len_thresh:
                 skip_len = True
                 logger.debug(f"Cache S1 Skip: OWI len ({len(owi_content_for_check.strip())}) < {len_thresh}.")
             else:
                  if calculate_string_similarity and previous_cache_text:
                      logger.debug("Cache S1 Skip: Similarity check DISABLED.")

             cache_update_skipped = skip_len or skip_sim
             run_step1 = not cache_update_skipped
             if cache_update_skipped:
                 await _emit_status_internal(event_emitter, session_id, logger, "Status: Skipping cache update (short OWI).")
                 updated_cache_text_intermediate = previous_cache_text
                 logger.debug("Cache Step 1 SKIPPED (OWI length or disabled similarity).")

        cache_update_llm_config = { "url": getattr(config, 'refiner_llm_api_url', None), "key": getattr(config, 'refiner_llm_api_key', None), "temp": getattr(config, 'refiner_llm_temperature', 0.3), "prompt_template": DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,}
        final_select_llm_config = { "url": getattr(config, 'refiner_llm_api_url', None), "key": getattr(config, 'refiner_llm_api_key', None), "temp": getattr(config, 'refiner_llm_temperature', 0.3), "prompt_template": DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,}
        configs_ok_step1 = all([cache_update_llm_config["url"], cache_update_llm_config["key"], cache_update_llm_config["prompt_template"] != "[Default Cache Prompt Load Failed]"])
        configs_ok_step2 = all([final_select_llm_config["url"], final_select_llm_config["key"], final_select_llm_config["prompt_template"] != "[Default Select Prompt Load Failed]"])

        if not (configs_ok_step1 and configs_ok_step2):
            logger.error("ERROR: RAG Cache Refiner config incomplete or default prompt failed load. Cannot proceed.");
            await _emit_status_internal(event_emitter, session_id, logger, "ERROR: RAG Cache Refiner config incomplete.", done=False);
            updated_cache_text_intermediate = previous_cache_text;
            run_step1 = False
        else:
             if run_step1:
                  await _emit_status_internal(event_emitter, session_id, logger, "Status: Updating background cache...");
                  logger.debug("Executing RAG Cache Step 1 (Update)...")
                  try:
                      updated_cache_text_intermediate = await update_rag_cache(
                          session_id=session_id, current_owi_context=extracted_owi_context,
                          history_messages=current_active_history, latest_user_query=latest_user_query_str,
                          llm_call_func=llm_call_func, sqlite_cursor=sqlite_cursor,
                          cache_update_llm_config=cache_update_llm_config, history_count=getattr(config, 'refiner_history_count', 6),
                          dialogue_only_roles=dialogue_roles, caller_info=f"CtxProc_CacheUpdate_{session_id}",
                      )
                      cache_update_performed = True;
                      logger.debug(f"RAG Cache Step 1 (Update) completed. Updated cache length: {len(updated_cache_text_intermediate)}")
                  except Exception as e_cache_update:
                      logger.error(f"[{session_id}] EXCEPTION during RAG Cache Step 1 (Update): {e_cache_update}", exc_info=True);
                      updated_cache_text_intermediate = previous_cache_text;
                      logger.error(f"EXCEPTION during RAG Cache Step 1: {e_cache_update}")

             await _emit_status_internal(event_emitter, session_id, logger, "Status: Selecting relevant context...");
             base_owi_context_for_selection = extracted_owi_context or "";
             logger.debug("Executing RAG Cache Step 2 (Select)...");
             logger.debug(f"Step 2 Input Cache Length: {len(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else '')}");
             logger.debug(f"Step 2 Input OWI Length: {len(base_owi_context_for_selection)}")

             final_selected_context = await select_final_context(
                 updated_cache_text=(updated_cache_text_intermediate if isinstance(updated_cache_text_intermediate, str) else ""),
                 current_owi_context=base_owi_context_for_selection, history_messages=current_active_history,
                 latest_user_query=latest_user_query_str, llm_call_func=llm_call_func,
                 context_selection_llm_config=final_select_llm_config, history_count=getattr(config, 'refiner_history_count', 6),
                 dialogue_only_roles=dialogue_roles, caller_info=f"CtxProc_CtxSelect_{session_id}",
                 debug_log_path_getter=orchestrator_debug_path_getter
             )

             final_context_selection_performed = True;
             refined_context_str = final_selected_context;
             log_step1_status = "Performed" if cache_update_performed else ("Skipped" if cache_update_skipped else "Not Run");
             logger.debug(f"RAG Cache Step 2 complete. Selected context length: {len(refined_context_str)}. Step 1: {log_step1_status}")
             await _emit_status_internal(event_emitter, session_id, logger, "Status: Context selection complete.", done=False)

    elif enable_stateless_refin_global and refine_external_context: # Use imported function
        logger.debug("Stateless Refinement Path Selected.")
        await _emit_status_internal(event_emitter, session_id, logger, "Status: Refining OWI context (stateless)...")
        if not extracted_owi_context: logger.debug("Skipping stateless refinement: No OWI context.")
        elif not latest_user_query_str: logger.debug("Skipping stateless refinement: Query empty.")
        else:
             stateless_refiner_config = { "url": getattr(config, 'refiner_llm_api_url', None), "key": getattr(config, 'refiner_llm_api_key', None), "temp": getattr(config, 'refiner_llm_temperature', 0.3), "prompt_template": DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,}
             if not stateless_refiner_config["url"] or not stateless_refiner_config["key"] or stateless_refiner_config["prompt_template"] == "[Default Stateless Prompt Load Failed]":
                 logger.warning("Skipping stateless refinement: Refiner URL/Key missing or default prompt failed load."); await _emit_status_internal(event_emitter, session_id, logger, "ERROR: Stateless Refiner config incomplete.", done=False)
             else:
                  try:
                      refined_stateless_context = await refine_external_context(
                          external_context=extracted_owi_context, history_messages=current_active_history,
                          latest_user_query=latest_user_query_str, llm_call_func=llm_call_func,
                          refiner_llm_config=stateless_refiner_config, skip_threshold=getattr(config, 'stateless_refiner_skip_threshold', 500),
                          history_count=getattr(config, 'refiner_history_count', 6), dialogue_only_roles=dialogue_roles,
                          caller_info=f"CtxProc_StatelessRef_{session_id}",
                      )
                      if refined_stateless_context != extracted_owi_context:
                          refined_context_str = refined_stateless_context;
                          stateless_refinement_performed = True;
                          logger.debug(f"Stateless refinement successful. Refined length: {len(refined_context_str)}.");
                          await _emit_status_internal(event_emitter, session_id, logger, "Status: OWI context refined (stateless).", done=False)
                      else:
                          logger.debug("Stateless refinement resulted in no change or was skipped by length.")
                  except Exception as e_refine_stateless:
                      logger.error(f"[{session_id}] EXCEPTION during stateless refinement: {e_refine_stateless}", exc_info=True);
                      logger.error(f"EXCEPTION during stateless refinement: {e_refine_stateless}")
    else:
        logger.debug("No context refinement feature (RAG Cache or Stateless) is enabled.")

    logger.debug("Step 4: Calculating refined context tokens...")
    if count_tokens and tokenizer:
        try: refined_context_tokens = count_tokens(refined_context_str, tokenizer) if refined_context_str else 0
        except Exception as e_tok_ref: refined_context_tokens = -1; logger.error(f"[{session_id}] Error calculating refined tokens: {e_tok_ref}")
    else: refined_context_tokens = -1
    logger.debug(f"Final refined_context_str tokens: {refined_context_tokens}")

    return (
        refined_context_str, base_system_prompt_text, initial_owi_context_tokens,
        refined_context_tokens, cache_update_performed, cache_update_skipped,
        final_context_selection_performed, stateless_refinement_performed,
        formatted_inventory_string
    )

async def _emit_status_internal(
    event_emitter: Optional[Callable],
    session_id: str,
    logger_instance: logging.Logger,
    description: str,
    done: bool = False
):
    """Internal status emitter helper. (Implementation unchanged)"""
    # [[ Implementation from previous correct version ]]
    if event_emitter and callable(event_emitter):
        try:
            status_data = { "type": "status", "data": {"description": str(description), "done": bool(done)} }
            if asyncio.iscoroutinefunction(event_emitter): await event_emitter(status_data)
            else: event_emitter(status_data)
        except Exception as e_emit: logger_instance.warning(f"[{session_id}] ContextProcessor failed to emit status '{description}': {e_emit}")
    else: logger_instance.debug(f"[{session_id}] ContextProcessor status update (not emitted): '{description}' (Done: {done})")

# --- Main Processing Function (MODIFIED Signature and Logic) ---

async def process_context_and_prepare_payload(
    # Identifiers & Core Data
    session_id: str,
    body: Dict,
    user_valves: Any,
    current_active_history: List[Dict], # Full history up to *before* current query
    history_for_processing: List[Dict], # History slice for T2/Refinement (pre-query)
    # === NEW Parameter ===
    t0_history_slice: List[Dict],       # Token-limited slice for final payload
    # ===================
    latest_user_query_str: str,
    # Pre-fetched Memory/State
    recent_t1_summaries: List[str],
    current_scene_state_dict: Dict[str, Any],
    current_world_state_dict: Dict[str, Any], # Includes day, time, weather, season
    generated_event_hint_text: Optional[str],
    generated_weather_proposal: Optional[Dict[str, Optional[str]]], # Added for combine_context
    # Configuration & Services
    config: object,
    logger: logging.Logger,
    sqlite_cursor: sqlite3.Cursor,
    chroma_client: Optional[Any],
    chroma_embed_wrapper: Optional[Any],
    embedding_func: Optional[Callable[[Sequence[str], str, Optional[Dict]], List[List[float]]]],
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    tokenizer: Optional[Any], # e.g., tiktoken tokenizer instance
    # Callbacks & Helpers
    event_emitter: Optional[Callable],
    orchestrator_debug_path_getter: Optional[Callable[[str], Optional[str]]],
    # Constants/Settings
    dialogue_roles: List[str],
    session_period_setting: str, # From user valves
) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
    """
    Orchestrates context processing: T2 RAG, background refinement,
    inventory formatting, context combination, and final payload construction.
    Uses the explicitly provided t0_history_slice for the final payload.

    Returns:
        A tuple containing:
        1. The final LLM payload contents (List[Dict]) or None if error.
        2. A dictionary containing status information for logging/reporting.
    """
    status_info = {
        "t2_retrieved_count": 0,
        "initial_owi_context_tokens": -1,
        "refined_context_tokens": -1,
        "cache_update_performed": False,
        "cache_update_skipped": False,
        "final_context_selection_performed": False,
        "stateless_refinement_performed": False,
        "error": None,
    }
    final_llm_payload_contents: Optional[List[Dict]] = None

    try:
        # --- T2 RAG Retrieval ---
        # Uses history_for_processing (pre-query slice)
        retrieved_rag_summaries, t2_retrieved_count = await _get_t2_rag_results_internal(
            session_id=session_id,
            history_for_processing=history_for_processing, # Use pre-query slice
            latest_user_query_str=latest_user_query_str,
            config=config, logger=logger,
            chroma_client=chroma_client, chroma_embed_wrapper=chroma_embed_wrapper,
            embedding_func=embedding_func, llm_call_func=llm_call_func,
            event_emitter=event_emitter, dialogue_roles=dialogue_roles
        )
        status_info["t2_retrieved_count"] = t2_retrieved_count

        # --- Prepare Refined Context & Base System Prompt ---
        # Uses current_active_history (full history) for refinement context
        (
            refined_owi_cache_context,
            base_system_prompt_text,
            initial_owi_context_tokens,
            refined_context_tokens,
            cache_update_performed,
            cache_update_skipped,
            final_context_selection_performed,
            stateless_refinement_performed,
            formatted_inventory_string # Get formatted inventory string here
        ) = await _prepare_and_refine_background_internal(
            session_id=session_id, body=body, user_valves=user_valves,
            retrieved_t1_summaries=recent_t1_summaries,
            retrieved_rag_summaries=retrieved_rag_summaries,
            current_active_history=current_active_history, # Use full history for refinement
            latest_user_query_str=latest_user_query_str,
            config=config, logger=logger, sqlite_cursor=sqlite_cursor,
            llm_call_func=llm_call_func, tokenizer=tokenizer,
            event_emitter=event_emitter, dialogue_roles=dialogue_roles,
            orchestrator_debug_path_getter=orchestrator_debug_path_getter
        )
        status_info.update({
            "initial_owi_context_tokens": initial_owi_context_tokens,
            "refined_context_tokens": refined_context_tokens,
            "cache_update_performed": cache_update_performed,
            "cache_update_skipped": cache_update_skipped,
            "final_context_selection_performed": final_context_selection_performed,
            "stateless_refinement_performed": stateless_refinement_performed,
        })

        # --- Combine Final Background Context ---
        # Uses the world/scene state *as they were loaded* at the start of the turn
        combined_context_string = combine_background_context( # Use imported function
            final_selected_context=refined_owi_cache_context,
            t1_summaries=recent_t1_summaries,
            t2_rag_results=retrieved_rag_summaries,
            scene_description=current_scene_state_dict.get("description", ""),
            inventory_context=formatted_inventory_string, # Use generated string
            current_day=current_world_state_dict.get("day"),
            current_time_of_day=current_world_state_dict.get("time_of_day"),
            current_season=current_world_state_dict.get("season"),
            current_weather=current_world_state_dict.get("weather"),
            weather_proposal=generated_weather_proposal # Pass proposal
        )

        # --- Construct Final Payload ---
        await _emit_status_internal(event_emitter, session_id, logger, "Status: Constructing final request...")
        # === MODIFIED: Use t0_history_slice ===
        payload_dict_or_error = construct_final_llm_payload( # Use imported function
             system_prompt=base_system_prompt_text,
             history=t0_history_slice, # Use the CORRECT token-limited slice
             context=combined_context_string,
             query=latest_user_query_str,
             long_term_goal=getattr(user_valves, 'long_term_goal', ''),
             event_hint=generated_event_hint_text,
             period_setting=session_period_setting, # Use passed value
             strategy="standard", # Or make configurable
             include_ack_turns=getattr(config, 'include_ack_turns', True),
        )
        # === END MODIFICATION ===

        if isinstance(payload_dict_or_error, dict) and "contents" in payload_dict_or_error:
            final_llm_payload_contents = payload_dict_or_error["contents"]
            # Log the number of turns in the *correctly limited* slice
            logger.debug(f"[{session_id}] Constructed final payload using T0 slice ({len(t0_history_slice)} history turns).")
        else:
            error_msg = payload_dict_or_error.get("error", "Unknown payload construction error") if isinstance(payload_dict_or_error, dict) else "Invalid return type"
            logger.error(f"[{session_id}] Payload constructor failed: {error_msg}")
            status_info["error"] = f"Payload construction failed: {error_msg}"
            final_llm_payload_contents = None

    except Exception as e:
        logger.error(f"[{session_id}] Unhandled exception in process_context_and_prepare_payload: {e}", exc_info=True)
        status_info["error"] = f"Context processor exception: {type(e).__name__}"
        final_llm_payload_contents = None

    return final_llm_payload_contents, status_info

# === END OF FILE i4_llm_agent/context_processor.py ===