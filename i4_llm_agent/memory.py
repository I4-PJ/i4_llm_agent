# [[START MODIFIED memory.py - REMOVE DEBUG LOGS]]
# i4_llm_agent/memory.py

import logging
import uuid
import asyncio
import sqlite3 # <<< ADDED for type hinting
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple, Union

# --- Internal Dependencies & Constants ---
try:
    from .history import format_history_for_llm, DIALOGUE_ROLES
    HISTORY_FORMATTER_AVAILABLE = True
except ImportError:
    HISTORY_FORMATTER_AVAILABLE = False
    DIALOGUE_ROLES = ["user", "assistant"]
    def format_history_for_llm(history_chunk: List[Dict], max_messages: Optional[int] = None, allowed_roles: Optional[List[str]] = None) -> str:
        logging.getLogger(__name__).warning("CRITICAL: Using fallback history formatter in memory_manager.")
        if not history_chunk: return ""
        lines = []; start_idx = -max_messages if max_messages is not None and max_messages > 0 else 0
        chunk_to_format = history_chunk[start_idx:]
        for msg in chunk_to_format:
            role = msg.get('role', 'unk').capitalize(); content = msg.get('content', '').strip()
            if content: lines.append(f"{role}: {content}")
        return "\n".join(lines)

# <<< ADDED Import for DB Check >>>
try:
    from .database import check_t1_summary_exists
except ImportError:
    check_t1_summary_exists = None # Handle case where it cannot be imported
    logging.getLogger(__name__).error("Failed to import check_t1_summary_exists from database module.")

# <<< NEW: Import Summarizer Prompt Constants >>>
try:
    from .prompting import DEFAULT_SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_DIALOGUE_CHUNK_PLACEHOLDER
    SUMMARIZER_PROMPT_LOADED = True
except ImportError:
    DEFAULT_SUMMARIZER_SYSTEM_PROMPT = "[Default Summarizer Prompt Load Failed]"
    SUMMARIZER_DIALOGUE_CHUNK_PLACEHOLDER = "{dialogue_chunk}" # Fallback placeholder
    SUMMARIZER_PROMPT_LOADED = False
    logging.getLogger(__name__).critical("Failed to import default summarizer prompt from prompting module.")
# <<< END NEW IMPORT >>>


# --- Logger ---
logger = logging.getLogger(__name__) # 'i4_llm_agent.memory'

# --- Helper Function: Select History by Tokens (FIXED: Ignores System Messages) ---
def _select_history_slice_by_tokens(
    messages: List[Dict],
    target_tokens: int,
    tokenizer: Any, # Expects a tokenizer with an .encode() method
    include_last: bool = True,
    fallback_turns: int = 10, # Fallback if no tokenizer
    dialogue_only_roles: List[str] = DIALOGUE_ROLES # Roles to consider
) -> List[Dict]:
    """
    Selects recent dialogue messages (user/assistant) aiming for a token limit.
    Iterates backwards through the filtered dialogue history.
    """
    func_logger = logging.getLogger(__name__ + '._select_history_slice_by_tokens')
    if not messages: return []
    source_list_full = messages if include_last else messages[:-1]
    dialogue_messages = [ msg for msg in source_list_full if isinstance(msg, dict) and msg.get("role") in dialogue_only_roles ]
    if not dialogue_messages: func_logger.debug("No dialogue messages found in the provided slice."); return []
    if not tokenizer:
        func_logger.warning(f"Tokenizer unavailable. Falling back to last {fallback_turns} dialogue turns.")
        start_idx = max(0, len(dialogue_messages) - fallback_turns); return dialogue_messages[start_idx:]
    selected_history: List[Dict] = []; current_tokens: int = 0
    try:
        for i in range(len(dialogue_messages) - 1, -1, -1):
            msg = dialogue_messages[i]; msg_content = msg.get("content", "")
            if not msg_content: continue
            msg_tokens = 0
            try: msg_tokens = len(tokenizer.encode(msg_content))
            except Exception as e: func_logger.error(f"Tokenizer error on dialogue msg index {i}: {e}. Skipping."); continue
            if selected_history and (current_tokens + msg_tokens > target_tokens): break
            selected_history.insert(0, msg); current_tokens += msg_tokens
    except Exception as e: func_logger.error(f"Unexpected error during token selection loop: {e}", exc_info=True); return selected_history
    return selected_history


# --- Main Memory Management Function (MODIFIED: Removed verbose DEBUG logs) ---
async def manage_tier1_summarization(
    # --- MODIFIED SIGNATURE ---
    current_last_summary_index: int,
    active_history: List[Dict],
    t0_token_limit: int,
    t1_chunk_size_target: int,
    tokenizer: Any,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    # MODIFIED: No longer requires 'sys_prompt' in llm_config
    llm_config: Dict[str, Any], # Should contain url, key, temp
    add_t1_summary_func: Callable[..., Coroutine[Any, Any, bool]],
    session_id: str,
    user_id: str,
    cursor: sqlite3.Cursor,
    is_regeneration: bool = False,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES
) -> Tuple[bool, Optional[str], int, int, int]: # Success, Summary Text, New T1 End Index, Prompt Tokens, T0 End Index
    """
    Manages T1 summarization based on dialogue history.
    Checks trigger based on dialogue tokens, identifies T1 chunk from dialogue,
    calls LLM using the library's default summarization prompt, and saves via callback.
    Includes a check during regeneration to prevent duplicate summaries.

    Args:
        current_last_summary_index: Index in active_history of the last message
                                    included in the *previous* summary (from DB).
        active_history: The full list of message dictionaries for the session.
        t0_token_limit: Token limit for the T0 active window.
        t1_chunk_size_target: Target token size for chunks to summarize (T1).
        tokenizer: Tokenizer instance with .encode().
        llm_call_func: Async function to call the summarizer LLM.
        llm_config: Config dict for the summarizer LLM (url, key, temp).
        add_t1_summary_func: Async callback function to save the generated summary.
        session_id: ID of the current session.
        user_id: ID of the current user.
        cursor: Active SQLite database cursor for DB checks.
        is_regeneration: Flag indicating if this is a regeneration request.
        dialogue_only_roles: List of roles considered dialogue history.

    Returns:
        Tuple containing:
        - bool: True if summarization was successfully performed and saved.
        - Optional[str]: The generated summary text if successful.
        - int: The new last_summary_turn_index to be saved by the caller.
        - int: Tokens used in the summarizer prompt (-1 if unavailable/failed).
        - int: Index in original active_history of the last message when summary triggered.
    """
    func_logger = logging.getLogger(__name__ + '.manage_tier1_summarization')
    func_logger.debug(f"Entering manage_tier1_summarization (Regen={is_regeneration})...")
    # === START REMOVED DEBUG LOGS ===
    # func_logger.debug(f"  Input: current_last_summary_index = {current_last_summary_index}")
    # func_logger.debug(f"  Input: active_history length = {len(active_history)}")
    # func_logger.debug(f"  Input: t0_token_limit = {t0_token_limit}")
    # func_logger.debug(f"  Input: t1_chunk_size_target = {t1_chunk_size_target}")
    # === END REMOVED DEBUG LOGS ===

    original_last_summary_index = current_last_summary_index
    summarization_performed = False
    generated_summary = None
    new_last_summary_index = original_last_summary_index
    summarizer_prompt_tokens = -1
    t0_end_index_at_summary = -1

    # --- Prerequisites Check ---
    if not cursor: func_logger.error("SQLite cursor unavailable."); return False, None, new_last_summary_index, -1, -1
    if not tokenizer: func_logger.warning("Tokenizer unavailable."); return False, None, new_last_summary_index, -1, -1
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func): func_logger.error("Async LLM func invalid."); return False, None, new_last_summary_index, -1, -1
    if not add_t1_summary_func or not asyncio.iscoroutinefunction(add_t1_summary_func): func_logger.error("Async Add T1 func invalid."); return False, None, new_last_summary_index, -1, -1
    required_llm_keys = ['url', 'key', 'temp']
    if not llm_config or not all(key in llm_config for key in required_llm_keys): func_logger.error(f"LLM Config missing keys: {[k for k in required_llm_keys if k not in (llm_config or {})]}"); return False, None, new_last_summary_index, -1, -1
    if not HISTORY_FORMATTER_AVAILABLE: func_logger.error("History formatter unavailable."); return False, None, new_last_summary_index, -1, -1
    if is_regeneration and not check_t1_summary_exists: func_logger.error("Regeneration check requires 'check_t1_summary_exists' function.");
    if not SUMMARIZER_PROMPT_LOADED: func_logger.critical("Summarizer Prompt constant failed to load. Cannot proceed."); return False, None, new_last_summary_index, -1, -1


    # --- Identify Unsummarized Dialogue ---
    unsummarized_full_slice = []
    unsummarized_start_index_absolute = original_last_summary_index + 1
    if unsummarized_start_index_absolute < len(active_history):
        unsummarized_full_slice = active_history[unsummarized_start_index_absolute :]
        func_logger.debug(f"Full unsummarized slice contains {len(unsummarized_full_slice)} messages (Abs Indices: {unsummarized_start_index_absolute} to {len(active_history) - 1}).")
    else:
         func_logger.debug("No new messages in active_history since last summary index.")
         return False, None, new_last_summary_index, -1, -1
    if not unsummarized_full_slice:
         func_logger.debug("Unsummarized message slice is empty after slicing.")
         return False, None, new_last_summary_index, -1, -1

    unsummarized_dialogue_messages = [ msg for msg in unsummarized_full_slice if isinstance(msg, dict) and msg.get("role") in dialogue_only_roles ]
    if not unsummarized_dialogue_messages:
        func_logger.debug(f"Unsummarized slice contains no dialogue messages. No trigger check needed.")
        return False, None, new_last_summary_index, -1, -1
    func_logger.debug(f"Filtered unsummarized slice to {len(unsummarized_dialogue_messages)} dialogue messages.")
    # === START REMOVED DEBUG LOGS ===
    # if unsummarized_dialogue_messages:
    #     first_unsum_msg = unsummarized_dialogue_messages[0]
    #     last_unsum_msg = unsummarized_dialogue_messages[-1]
    #     try:
    #         first_unsum_idx_abs = active_history.index(first_unsum_msg)
    #         last_unsum_idx_abs = active_history.index(last_unsum_msg)
    #         func_logger.debug(f"  Unsummarized Dialogue: First message (Abs Index {first_unsum_idx_abs}): '{str(first_unsum_msg.get('content', ''))[:50]}...'")
    #         func_logger.debug(f"  Unsummarized Dialogue: Last message (Abs Index {last_unsum_idx_abs}): '{str(last_unsum_msg.get('content', ''))[:50]}...'")
    #     except ValueError:
    #         func_logger.error("  Could not find unsummarized dialogue msg in active_history for debug logging.")
    # === END REMOVED DEBUG LOGS ===

    total_unsummarized_dialogue_tokens = 0
    try:
        combined_dialogue_text = " ".join([msg.get("content", "") for msg in unsummarized_dialogue_messages if msg.get("content")])
        if combined_dialogue_text: total_unsummarized_dialogue_tokens = len(tokenizer.encode(combined_dialogue_text))
        func_logger.debug(f"Estimated total unsummarized DIALOGUE tokens: {total_unsummarized_dialogue_tokens}")
    except Exception as e:
        func_logger.error(f"Tokenizer error calculating total unsummarized dialogue tokens: {e}", exc_info=True)
        return False, None, new_last_summary_index, -1, -1


    # --- Check Trigger Condition ---
    summarization_trigger_threshold = t0_token_limit + t1_chunk_size_target
    should_summarize = (total_unsummarized_dialogue_tokens > summarization_trigger_threshold)
    func_logger.info(f"Summarization Trigger Check: Dialogue Tokens={total_unsummarized_dialogue_tokens}, Threshold={summarization_trigger_threshold}, Triggered={should_summarize}")

    if not should_summarize:
        func_logger.debug(f"Summarization not triggered.")
        return False, None, new_last_summary_index, -1, -1

    func_logger.info(f"Summarization triggered.")
    t0_end_index_at_summary = len(active_history) - 1
    # === START REMOVED DEBUG LOGS ===
    # func_logger.debug(f"  T0 End Index determined at summary trigger: {t0_end_index_at_summary}")
    # === END REMOVED DEBUG LOGS ===


    # --- Identify T0 and T1 Chunks (From Dialogue Messages) ---
    func_logger.debug(f"Identifying T0 slice within unsummarized DIALOGUE block (target: {t0_token_limit} tokens)...")
    t0_dialogue_slice = _select_history_slice_by_tokens(
        messages=unsummarized_dialogue_messages, target_tokens=t0_token_limit,
        tokenizer=tokenizer, include_last=True, dialogue_only_roles=dialogue_only_roles
    )
    func_logger.debug(f"Identified {len(t0_dialogue_slice)} dialogue messages for T0 slice.")
    # === START REMOVED DEBUG LOGS ===
    # if t0_dialogue_slice:
    #     first_t0_msg = t0_dialogue_slice[0]
    #     last_t0_msg = t0_dialogue_slice[-1]
    #     try:
    #         first_t0_idx_abs = active_history.index(first_t0_msg)
    #         last_t0_idx_abs = active_history.index(last_t0_msg)
    #         func_logger.debug(f"  T0 Dialogue Slice: First message (Abs Index {first_t0_idx_abs}): '{str(first_t0_msg.get('content', ''))[:50]}...'")
    #         func_logger.debug(f"  T0 Dialogue Slice: Last message (Abs Index {last_t0_idx_abs}): '{str(last_t0_msg.get('content', ''))[:50]}...'")
    #     except ValueError:
    #          func_logger.error("  Could not find T0 dialogue msg in active_history for debug logging.")
    # === END REMOVED DEBUG LOGS ===

    t1_chunk_dialogue_messages = []
    if not t0_dialogue_slice:
        func_logger.warning("Could not select T0 dialogue slice, using full unsummarized dialogue as T1 chunk.")
        t1_chunk_dialogue_messages = unsummarized_dialogue_messages
    else:
        first_t0_message = t0_dialogue_slice[0]
        try:
            # Find the index of the first T0 message WITHIN the unsummarized dialogue block
            t1_chunk_end_index_relative_dialogue = unsummarized_dialogue_messages.index(first_t0_message)
            func_logger.debug(f"First T0 msg found at relative index {t1_chunk_end_index_relative_dialogue} within unsummarized dialogue.")
            if t1_chunk_end_index_relative_dialogue > 0:
                 # The T1 chunk consists of messages BEFORE the first T0 message in the unsummarized block
                 t1_chunk_dialogue_messages = unsummarized_dialogue_messages[:t1_chunk_end_index_relative_dialogue]
                 func_logger.info(f"Identified T1 chunk: {len(t1_chunk_dialogue_messages)} dialogue messages.")
            else:
                 func_logger.info(f"T0 started at beginning of unsummarized dialogue. No T1 chunk yet.")
                 return False, None, new_last_summary_index, -1, t0_end_index_at_summary
        except ValueError:
             func_logger.error("CRITICAL: Could not find start of T0 dialogue slice within unsummarized dialogue. Aborting.", exc_info=True)
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary
        except Exception as e:
             func_logger.error(f"Error determining T1 dialogue chunk indices: {e}", exc_info=True)
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary

    if not t1_chunk_dialogue_messages:
        func_logger.error("Identified T1 dialogue chunk is empty. Aborting summarization.")
        return False, None, new_last_summary_index, -1, t0_end_index_at_summary

    # === START REMOVED DEBUG LOGS ===
    # first_t1_chunk_msg = t1_chunk_dialogue_messages[0]
    # last_t1_chunk_msg = t1_chunk_dialogue_messages[-1]
    # try:
    #     first_t1_chunk_idx_abs = active_history.index(first_t1_chunk_msg)
    #     last_t1_chunk_idx_abs = active_history.index(last_t1_chunk_msg)
    #     func_logger.debug(f"  Final T1 Chunk for Summarization: First message (Abs Index {first_t1_chunk_idx_abs}): '{str(first_t1_chunk_msg.get('content', ''))[:50]}...'")
    #     func_logger.debug(f"  Final T1 Chunk for Summarization: Last message (Abs Index {last_t1_chunk_idx_abs}): '{str(last_t1_chunk_msg.get('content', ''))[:50]}...'")
    #     func_logger.debug(f"  Final T1 Chunk for Summarization: Total messages = {len(t1_chunk_dialogue_messages)}")
    # except ValueError:
    #      func_logger.error("  Could not find T1 chunk dialogue msg in active_history for debug logging.")
    # === END REMOVED DEBUG LOGS ===


    # --- Perform Summarization (on T1 Dialogue Chunk) ---
    try:
        # Determine Absolute Indices for Metadata and DB Check
        last_msg_in_t1_chunk = t1_chunk_dialogue_messages[-1]
        t1_chunk_end_index_absolute = -1
        try:
            t1_chunk_end_index_absolute = active_history.index(last_msg_in_t1_chunk)
            t1_chunk_start_index_absolute = unsummarized_start_index_absolute # Should be correct now
            func_logger.info(f"Summarizing T1 dialogue chunk (Abs Indices: {t1_chunk_start_index_absolute} to {t1_chunk_end_index_absolute}).")
            # === START REMOVED DEBUG LOGS ===
            # func_logger.debug(f"  Calculated Absolute Indices for Metadata/DB: Start={t1_chunk_start_index_absolute}, End={t1_chunk_end_index_absolute}")
            # === END REMOVED DEBUG LOGS ===
        except ValueError:
             func_logger.error("CRITICAL: Could not map end of T1 chunk back to original history index. Aborting summary.", exc_info=True)
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary

        # --- Regeneration Check ---
        if is_regeneration and check_t1_summary_exists:
            try:
                summary_exists = await check_t1_summary_exists(
                    cursor, session_id, t1_chunk_start_index_absolute, t1_chunk_end_index_absolute
                )
                if summary_exists:
                    func_logger.info(f"[{session_id}] Regen detected & identical T1 block ({t1_chunk_start_index_absolute}-{t1_chunk_end_index_absolute}) exists. Skipping LLM.")
                    return False, None, original_last_summary_index, -1, t0_end_index_at_summary
                else:
                     func_logger.debug(f"[{session_id}] Regen detected, but no identical T1 block found. Proceeding.")
            except Exception as e_check:
                 func_logger.error(f"[{session_id}] Error checking T1 summary: {e_check}. Proceeding with summary.", exc_info=True)

        # --- Format and Call LLM ---
        formatted_t1_dialogue_chunk = format_history_for_llm(
            t1_chunk_dialogue_messages, allowed_roles=dialogue_only_roles
        )
        if not formatted_t1_dialogue_chunk or not formatted_t1_dialogue_chunk.strip():
             func_logger.warning("Formatted T1 dialogue chunk is empty. Skipping LLM call.")
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary

        try:
            prompt = DEFAULT_SUMMARIZER_SYSTEM_PROMPT.format(
                **{SUMMARIZER_DIALOGUE_CHUNK_PLACEHOLDER.strip('{}'): formatted_t1_dialogue_chunk}
            )
        except KeyError as e_fmt:
            func_logger.critical(f"Failed to format default summarizer prompt: Missing key {e_fmt}. Aborting.", exc_info=True)
            return False, None, new_last_summary_index, -1, t0_end_index_at_summary
        except Exception as e_fmt_gen:
             func_logger.critical(f"Failed to format default summarizer prompt: {e_fmt_gen}. Aborting.", exc_info=True)
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary

        summarizer_prompt_tokens = -1
        try: summarizer_prompt_tokens = len(tokenizer.encode(prompt)); func_logger.debug(f"Summarizer Payload tokens: {summarizer_prompt_tokens}")
        except Exception as e_tok: func_logger.error(f"Tokenizer error on Summarizer Prompt: {e_tok}", exc_info=False)

        summ_payload = {"contents": [{"parts": [{"text": prompt}]}]}
        func_logger.info("Calling Summarizer LLM via async llm_call_func...")
        success, result_or_error = await llm_call_func(
             api_url=llm_config['url'],
             api_key=llm_config['key'],
             payload=summ_payload,
             temperature=llm_config['temp'],
             timeout=120,
             caller_info="i4_llm_agent_Summarizer",
        )

        # --- Process Summarization Result ---
        if success and isinstance(result_or_error, str) and result_or_error.strip():
            summary_result_text = result_or_error
            func_logger.info("Summary generated successfully by LLM.")
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
                "t0_end_index_at_summary": t0_end_index_at_summary,
            }

            func_logger.info(f"Attempting to save T1 summary {summary_id} via callback...")
            save_successful = await add_t1_summary_func(
                summary_id=summary_id, session_id=session_id, user_id=user_id,
                summary_text=generated_summary, metadata=metadata
            )
            if save_successful:
                func_logger.info(f"T1 summary {summary_id} saved successfully.")
                summarization_performed = True
                # CRITICAL: Set the new index to be returned
                new_last_summary_index = t1_chunk_end_index_absolute
                # === START REMOVED DEBUG LOGS ===
                # func_logger.debug(f"  Set new_last_summary_index to T1 chunk end absolute index: {new_last_summary_index}")
                # === END REMOVED DEBUG LOGS ===
            else:
                func_logger.error(f"Failed to save T1 summary {summary_id} via callback.")
                # Do NOT update new_last_summary_index

        elif success and (not isinstance(result_or_error, str) or not result_or_error.strip()):
             func_logger.error(f"Summarization failed (LLM returned empty/invalid content). Type: {type(result_or_error)}")
        elif not success and isinstance(result_or_error, dict):
            error_type = result_or_error.get('error_type', 'Unknown'); error_msg = result_or_error.get('message', 'No details')
            func_logger.error(f"Summarizer LLM call failed: Type='{error_type}', Message='{error_msg}'.")
        else:
            func_logger.error(f"Summarization failed (LLM call failed). Result: '{result_or_error}'")

    except Exception as e:
        func_logger.error(f"Unexpected error during manage_tier1_summarization exec block: {e}", exc_info=True)
        summarization_performed = False; generated_summary = None

    # --- Return Results ---
    func_logger.debug( f"Exiting manage_tier1_summarization. Success: {summarization_performed}, New T1 Idx: {new_last_summary_index}, Prompt Tok: {summarizer_prompt_tokens}, T0 End Idx: {t0_end_index_at_summary}" )
    return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens, t0_end_index_at_summary
# [[END MODIFIED memory.py - REMOVE DEBUG LOGS]]