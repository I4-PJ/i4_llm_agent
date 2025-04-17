# i4_llm_agent/memory.py

import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple

# --- Internal Dependencies ---
try:
    from .history import format_history_for_llm
    HISTORY_FORMATTER_AVAILABLE = True
except ImportError:
    HISTORY_FORMATTER_AVAILABLE = False
    def format_history_for_llm(history_chunk: List[Dict], max_messages: Optional[int] = None) -> str:
        logging.getLogger(__name__).warning("CRITICAL: Using fallback history formatter in memory_manager.")
        if not history_chunk: return ""
        lines = []
        start_idx = -max_messages if max_messages is not None and max_messages > 0 else 0
        chunk_to_format = history_chunk[start_idx:]
        for msg in chunk_to_format:
            role = msg.get('role', 'unk').capitalize()
            content = msg.get('content', '').strip()
            if content: lines.append(f"{role}: {content}")
        return "\n".join(lines)

# --- Logger ---
logger = logging.getLogger(__name__) # 'i4_llm_agent.memory'

# --- Helper Function: Select History by Tokens ---
# (Keep internal to this module)
def _select_history_slice_by_tokens(
    messages: List[Dict],
    target_tokens: int,
    tokenizer: Any, # Expects a tokenizer with an .encode() method
    include_last: bool = True,
    fallback_turns: int = 10 # Fallback if no tokenizer
) -> List[Dict]:
    """Selects recent messages aiming for a token limit, iterating backwards."""
    func_logger = logging.getLogger(__name__ + '._select_history_slice_by_tokens')
    if not messages: return []
    source_list = messages if include_last else messages[:-1]
    if not source_list: return []

    if not tokenizer:
        func_logger.warning(f"Tokenizer unavailable. Falling back to last {fallback_turns} turns.")
        start_idx = max(0, len(source_list) - fallback_turns)
        return source_list[start_idx:]

    selected_history: List[Dict] = []
    current_tokens: int = 0
    stop_adding = False
    try:
        for i in range(len(source_list) - 1, -1, -1):
            if stop_adding: break
            msg = source_list[i]
            msg_content = msg.get("content", "")
            if not msg_content: continue
            msg_tokens = 0
            try:
                msg_tokens = len(tokenizer.encode(msg_content))
            except Exception as e:
                func_logger.error(f"Tokenizer error on msg index {i}: {e}. Skipping msg.")
                continue
            if selected_history and (current_tokens + msg_tokens > target_tokens):
                stop_adding = True
                continue
            selected_history.insert(0, msg)
            current_tokens += msg_tokens
    except Exception as e:
        func_logger.error(f"Unexpected error during token selection loop: {e}", exc_info=True)
        return selected_history if selected_history else []
    # func_logger.debug(f"Selected {len(selected_history)} msgs, ~{current_tokens} tokens for slice.")
    return selected_history


# --- Main Memory Management Function ---
# <<< Updated Return Signature: Added t0_end_index_at_summary >>>
async def manage_tier1_summarization(
    session_state: Dict,
    active_history: List[Dict],
    t0_token_limit: int,
    t1_chunk_size_target: int,
    tokenizer: Any,
    llm_call_func: Callable[..., Coroutine[Any, Any, Optional[str]]],
    llm_config: Dict[str, Any],
    add_t1_summary_func: Callable[..., Coroutine[Any, Any, bool]],
    session_id: str,
    user_id: str
) -> Tuple[bool, Optional[str], int, int, int]: # Success, Summary Text, New T1 End Index, Prompt Tokens, T0 End Index
    """
    Manages T1 summarization: checks trigger, identifies chunks, calls LLM, saves via callback.

    Returns:
        Tuple containing:
        - bool: True if summarization was successfully performed and saved.
        - Optional[str]: The generated summary text if successful.
        - int: The new last_summary_turn_index (end index of the T1 chunk).
        - int: Tokens used in the summarizer prompt (-1 if unavailable/failed).
        - int: Index of the last message in active_history *at the time of summarization* (-1 if not performed).
    """
    logger.debug("Entering manage_tier1_summarization function...")
    original_last_summary_index = session_state.get("last_summary_turn_index", -1)
    summarization_performed = False
    generated_summary = None
    new_last_summary_index = original_last_summary_index
    summarizer_prompt_tokens = -1
    t0_end_index_at_summary = -1 # <<< Initialize new return value

    # --- Prerequisites Check ---
    # (Keep checks for tokenizer, llm_call_func, add_t1_summary_func, llm_config, HISTORY_FORMATTER_AVAILABLE)
    if not tokenizer: logger.warning("Tokenizer unavailable."); return False, None, new_last_summary_index, -1, -1
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func): logger.error("Async LLM func invalid."); return False, None, new_last_summary_index, -1, -1
    if not add_t1_summary_func or not asyncio.iscoroutinefunction(add_t1_summary_func): logger.error("Async Add T1 func invalid."); return False, None, new_last_summary_index, -1, -1
    required_llm_keys = ['url', 'key', 'temp', 'sys_prompt']
    if not all(key in llm_config for key in required_llm_keys): logger.error(f"LLM Config missing keys: {[k for k in required_llm_keys if k not in llm_config]}"); return False, None, new_last_summary_index, -1, -1
    if not HISTORY_FORMATTER_AVAILABLE: logger.error("History formatter unavailable."); return False, None, new_last_summary_index, -1, -1

    # --- Calculate current unsummarized tokens ---
    current_unsummarized_messages = []
    if original_last_summary_index < len(active_history) - 1:
        current_unsummarized_messages = active_history[original_last_summary_index + 1 :]
    else:
         logger.debug("No new messages since last summary index.")
         return False, None, new_last_summary_index, -1, -1
    if not current_unsummarized_messages:
         logger.debug("Unsummarized message slice is empty.")
         return False, None, new_last_summary_index, -1, -1

    total_unsummarized_tokens = 0
    try:
        combined_text = " ".join([msg.get("content", "") for msg in current_unsummarized_messages if msg.get("content")])
        if combined_text: total_unsummarized_tokens = len(tokenizer.encode(combined_text))
        else: total_unsummarized_tokens = 0
        logger.debug(f"Estimated total unsummarized tokens: {total_unsummarized_tokens}")
    except Exception as e:
        logger.error(f"Tokenizer error calculating total unsummarized tokens: {e}", exc_info=True)
        return False, None, new_last_summary_index, -1, -1

    # --- Check Trigger Condition ---
    summarization_trigger_threshold = t0_token_limit + t1_chunk_size_target
    should_summarize = (total_unsummarized_tokens > summarization_trigger_threshold)
    if not should_summarize:
        # logger.debug(f"Summarization not triggered ({total_unsummarized_tokens} <= {summarization_trigger_threshold}).")
        return False, None, new_last_summary_index, -1, -1
    logger.info(f"Summarization triggered ({total_unsummarized_tokens} > {summarization_trigger_threshold}).")

    # --- Identify T0 and T1 Chunks ---
    # logger.debug(f"Identifying T0 slice within unsummarized block (target: {t0_token_limit} tokens)...")
    t0_messages_slice = _select_history_slice_by_tokens(
        messages=current_unsummarized_messages, target_tokens=t0_token_limit, tokenizer=tokenizer, include_last=True
    )
    if not t0_messages_slice:
        logger.error("Failed to select T0 history slice. Aborting summarization cycle.")
        return False, None, new_last_summary_index, -1, -1

    first_t0_message = t0_messages_slice[0]
    t1_chunk_messages = []
    t1_chunk_end_index_relative = -1
    try:
        t1_chunk_end_index_relative = current_unsummarized_messages.index(first_t0_message)
        if t1_chunk_end_index_relative > 0:
            t1_chunk_messages = current_unsummarized_messages[0:t1_chunk_end_index_relative]
            logger.info(f"Identified T1 chunk: {len(t1_chunk_messages)} messages to summarize.")
        else:
            logger.info("T0 slice includes all unsummarized messages. No separate T1 chunk yet.")
            return False, None, new_last_summary_index, -1, -1
    except ValueError:
        logger.error("CRITICAL: Could not find start of T0 slice within unsummarized messages. Aborting.", exc_info=True)
        return False, None, new_last_summary_index, -1, -1
    except Exception as e:
        logger.error(f"Error determining T1 chunk indices: {e}", exc_info=True)
        return False, None, new_last_summary_index, -1, -1
    if not t1_chunk_messages:
        logger.error("Identified T1 chunk is empty. Aborting.")
        return False, None, new_last_summary_index, -1, -1

    # --- Perform Summarization ---
    try:
        t1_chunk_start_index_absolute = original_last_summary_index + 1
        t1_chunk_end_index_absolute = original_last_summary_index + t1_chunk_end_index_relative
        # <<< Capture T0 End Index HERE >>>
        current_t0_end_index = len(active_history) - 1 # Index of last msg *before* potential new ones arrive

        logger.info(f"Summarizing T1 chunk (Abs Indices {t1_chunk_start_index_absolute} to {t1_chunk_end_index_absolute}). Current T0 ends at index {current_t0_end_index}.")
        formatted_t1_chunk = format_history_for_llm(t1_chunk_messages)
        if not formatted_t1_chunk or not formatted_t1_chunk.strip():
             logger.warning("Formatted T1 chunk is empty. Skipping LLM call.")
             return False, None, new_last_summary_index, -1, -1

        summarizer_sys_prompt = llm_config.get('sys_prompt', "Summarize this dialogue.")
        prompt = f"{summarizer_sys_prompt}\\n\\n--- Dialogue History Chunk ---\\n{formatted_t1_chunk}\\n\\n--- End Dialogue History Chunk ---\\n\\nConcise Summary:"

        try:
            summarizer_prompt_tokens = len(tokenizer.encode(prompt))
            logger.debug(f"Summarizer Payload tokens: {summarizer_prompt_tokens}")
        except Exception as e_tok:
            logger.error(f"Tokenizer error on Summarizer Prompt: {e_tok}", exc_info=True)
            summarizer_prompt_tokens = -1

        summ_payload = {"contents": [{"parts": [{"text": prompt}]}]}
        logger.info("Calling Summarizer LLM via async llm_call_func...")
        summary_result_text = await llm_call_func(
             api_url=llm_config['url'], api_key=llm_config['key'], payload=summ_payload,
             temperature=llm_config['temp'], timeout=120, caller_info="i4_llm_agent_Summarizer",
        )

        # --- Process Summarization Result ---
        if (isinstance(summary_result_text, str) and not summary_result_text.startswith("[Error:") and summary_result_text.strip()):
            logger.info("Summary generated successfully by LLM.")
            generated_summary = summary_result_text.strip()
            summary_id = f"t1_sum_{uuid.uuid4()}"
            now = datetime.now(timezone.utc)
            metadata = {
                "session_id": session_id, "user_id": user_id,
                "timestamp_utc": now.timestamp(), "timestamp_iso": now.isoformat(),
                "turn_start_index": t1_chunk_start_index_absolute,
                "turn_end_index": t1_chunk_end_index_absolute,
                "char_length": len(generated_summary), "doc_type": "llm_summary",
                "config_t0_token_limit": t0_token_limit,
                "config_t1_chunk_target": t1_chunk_size_target,
                "calculated_prompt_tokens": summarizer_prompt_tokens,
                "t0_end_index_at_summary": current_t0_end_index, # <<< Add the captured T0 end index
            }

            logger.info(f"Attempting to save T1 summary {summary_id} via callback...")
            save_successful = await add_t1_summary_func(
                summary_id=summary_id, session_id=session_id, user_id=user_id,
                summary_text=generated_summary, metadata=metadata # Pass updated metadata
            )
            if save_successful:
                logger.info(f"T1 summary {summary_id} saved successfully.")
                summarization_performed = True
                new_last_summary_index = t1_chunk_end_index_absolute
                t0_end_index_at_summary = current_t0_end_index # Set return value on success
            else:
                logger.error(f"Failed to save T1 summary {summary_id} via callback.")
        elif isinstance(summary_result_text, str) and summary_result_text.startswith("[Error:"):
            logger.error(f"Summarizer LLM call failed: {summary_result_text}")
        else:
            logger.error(f"Summarization failed (LLM returned None or empty content). Result: '{summary_result_text}'")
    except Exception as e:
        logger.error(f"Unexpected error during manage_tier1_summarization execution block: {e}", exc_info=True)

    logger.debug(f"Exiting manage_tier1_summarization. Success: {summarization_performed}, New T1 Idx: {new_last_summary_index}, Prompt Tok: {summarizer_prompt_tokens}, T0 End Idx: {t0_end_index_at_summary}")
    # <<< RETURN UPDATED 5-TUPLE >>>
    return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens, t0_end_index_at_summary