# i4_llm_agent/memory.py

import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple, Union

# --- Internal Dependencies & Constants ---
try:
    # Import history formatting and role constants
    from .history import format_history_for_llm, DIALOGUE_ROLES
    HISTORY_FORMATTER_AVAILABLE = True
except ImportError:
    HISTORY_FORMATTER_AVAILABLE = False
    DIALOGUE_ROLES = ["user", "assistant"] # Fallback if import fails
    def format_history_for_llm(history_chunk: List[Dict], max_messages: Optional[int] = None, allowed_roles: Optional[List[str]] = None) -> str:
        # Fallback formatter (doesn't use allowed_roles here for simplicity)
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

    # Filter for dialogue roles first
    source_list_full = messages if include_last else messages[:-1]
    dialogue_messages = [
        msg for msg in source_list_full
        if isinstance(msg, dict) and msg.get("role") in dialogue_only_roles
    ]

    if not dialogue_messages:
        func_logger.debug("No dialogue messages found in the provided slice.")
        return []

    # Fallback if no tokenizer
    if not tokenizer:
        func_logger.warning(f"Tokenizer unavailable. Falling back to last {fallback_turns} dialogue turns.")
        start_idx = max(0, len(dialogue_messages) - fallback_turns)
        return dialogue_messages[start_idx:]

    # Select from filtered dialogue messages
    selected_history: List[Dict] = []
    current_tokens: int = 0
    try:
        # Iterate backwards through the dialogue-only messages
        for i in range(len(dialogue_messages) - 1, -1, -1):
            msg = dialogue_messages[i]
            msg_content = msg.get("content", "")
            if not msg_content: continue # Skip empty dialogue messages

            msg_tokens = 0
            try:
                # Calculate tokens for this dialogue message
                msg_tokens = len(tokenizer.encode(msg_content))
            except Exception as e:
                func_logger.error(f"Tokenizer error on dialogue msg index {i}: {e}. Skipping.")
                continue # Skip if tokenization fails

            # Check if adding this message exceeds the target
            if selected_history and (current_tokens + msg_tokens > target_tokens):
                # func_logger.debug(f"Token limit {target_tokens} reached. Stopping before adding dialogue index {i}.")
                break # Stop *before* adding this message

            # Prepend the message and update tokens
            selected_history.insert(0, msg)
            current_tokens += msg_tokens

    except Exception as e:
        func_logger.error(f"Unexpected error during token selection loop: {e}", exc_info=True)
        # Return whatever was selected, even if loop failed mid-way
        return selected_history

    # func_logger.debug(f"Selected {len(selected_history)} dialogue msgs, ~{current_tokens} tokens for slice.")
    return selected_history


# --- Main Memory Management Function (FIXED: Ignores System Messages for Trigger/Chunk) ---
async def manage_tier1_summarization(
    # --- MODIFIED SIGNATURE ---
    # session_state: Dict, # REMOVED: No longer pass the whole state dict
    current_last_summary_index: int, # ADDED: Pass the *current* index directly
    # --- END MODIFICATION ---
    active_history: List[Dict], # The *full* active history including system messages
    t0_token_limit: int,
    t1_chunk_size_target: int,
    tokenizer: Any,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # <<< Corrected LLM call func type hint >>>
    llm_config: Dict[str, Any],
    add_t1_summary_func: Callable[..., Coroutine[Any, Any, bool]],
    session_id: str,
    user_id: str,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES # Roles for summarization trigger/content
) -> Tuple[bool, Optional[str], int, int, int]: # Success, Summary Text, New T1 End Index, Prompt Tokens, T0 End Index
    """
    Manages T1 summarization based on dialogue history (ignoring system messages).
    Checks trigger based on dialogue tokens, identifies T1 chunk from dialogue,
    calls LLM to summarize the dialogue chunk, and saves via callback.

    Args:
        current_last_summary_index (int): The index in active_history of the last message
                                          included in the *previous* summary.
        active_history: The full list of message dictionaries for the session.
        t0_token_limit: Token limit for the T0 active window.
        t1_chunk_size_target: Target token size for chunks to summarize (T1).
        tokenizer: Tokenizer instance with .encode().
        llm_call_func: Async function to call the summarizer LLM. Expects Tuple[bool, Union[str, Dict]] return.
        llm_config: Config dict for the summarizer LLM (url, key, temp, sys_prompt).
        add_t1_summary_func: Async callback function to save the generated summary.
        session_id: ID of the current session.
        user_id: ID of the current user.
        dialogue_only_roles: List of roles considered dialogue history.

    Returns:
        Tuple containing:
        - bool: True if summarization was successfully performed and saved.
        - Optional[str]: The generated summary text if successful.
        - int: The new last_summary_turn_index (index in original active_history) to be saved by the caller.
        - int: Tokens used in the summarizer prompt (-1 if unavailable/failed).
        - int: Index in original active_history of the last message when summary triggered.
    """
    func_logger = logging.getLogger(__name__ + '.manage_tier1_summarization')
    func_logger.debug("Entering manage_tier1_summarization function...")

    # Use the passed index directly
    original_last_summary_index = current_last_summary_index
    summarization_performed = False
    generated_summary = None
    # Initialize new index with the original one. It will only change if summarization is successful.
    new_last_summary_index = original_last_summary_index
    summarizer_prompt_tokens = -1
    t0_end_index_at_summary = -1 # Index in original history

    # --- Prerequisites Check ---
    if not tokenizer: func_logger.warning("Tokenizer unavailable."); return False, None, new_last_summary_index, -1, -1
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func): func_logger.error("Async LLM func invalid."); return False, None, new_last_summary_index, -1, -1
    if not add_t1_summary_func or not asyncio.iscoroutinefunction(add_t1_summary_func): func_logger.error("Async Add T1 func invalid."); return False, None, new_last_summary_index, -1, -1
    required_llm_keys = ['url', 'key', 'temp', 'sys_prompt']
    if not llm_config or not all(key in llm_config for key in required_llm_keys): func_logger.error(f"LLM Config missing keys: {[k for k in required_llm_keys if k not in llm_config]}"); return False, None, new_last_summary_index, -1, -1
    if not HISTORY_FORMATTER_AVAILABLE: func_logger.error("History formatter unavailable."); return False, None, new_last_summary_index, -1, -1


    # --- Identify Unsummarized Dialogue ---
    # 1. Slice the *full* history based on the last summary index
    unsummarized_full_slice = []
    if original_last_summary_index < len(active_history) - 1:
        unsummarized_full_slice = active_history[original_last_summary_index + 1 :]
        func_logger.debug(f"Full unsummarized slice contains {len(unsummarized_full_slice)} messages (from index {original_last_summary_index + 1}).")
    else:
         func_logger.debug("No new messages in active_history since last summary index.")
         return False, None, new_last_summary_index, -1, -1

    if not unsummarized_full_slice:
         # This shouldn't happen if the length check passed, but double-check
         func_logger.debug("Unsummarized message slice is empty after slicing.")
         return False, None, new_last_summary_index, -1, -1

    # 2. Filter this slice to get only dialogue messages for token counting
    unsummarized_dialogue_messages = [
        msg for msg in unsummarized_full_slice
        if isinstance(msg, dict) and msg.get("role") in dialogue_only_roles
    ]

    if not unsummarized_dialogue_messages:
        func_logger.debug(f"Unsummarized slice contains no dialogue messages (roles: {dialogue_only_roles}). No trigger check needed.")
        return False, None, new_last_summary_index, -1, -1

    func_logger.debug(f"Filtered unsummarized slice to {len(unsummarized_dialogue_messages)} dialogue messages.")

    # 3. Calculate token count ONLY on the dialogue messages
    total_unsummarized_dialogue_tokens = 0
    try:
        # Combine content of dialogue messages only
        combined_dialogue_text = " ".join([msg.get("content", "") for msg in unsummarized_dialogue_messages if msg.get("content")])
        if combined_dialogue_text:
            total_unsummarized_dialogue_tokens = len(tokenizer.encode(combined_dialogue_text))
        func_logger.debug(f"Estimated total unsummarized DIALOGUE tokens: {total_unsummarized_dialogue_tokens}")
    except Exception as e:
        func_logger.error(f"Tokenizer error calculating total unsummarized dialogue tokens: {e}", exc_info=True)
        return False, None, new_last_summary_index, -1, -1


    # --- Check Trigger Condition (Based on Dialogue Tokens) ---
    summarization_trigger_threshold = t0_token_limit + t1_chunk_size_target
    func_logger.info(f"DEBUG TRIGGER CHECK: Dialogue Tokens = {total_unsummarized_dialogue_tokens}, Threshold ({t0_token_limit}+{t1_chunk_size_target}) = {summarization_trigger_threshold}")
    should_summarize = (total_unsummarized_dialogue_tokens > summarization_trigger_threshold)
    func_logger.info(f"DEBUG TRIGGER CHECK: should_summarize = {should_summarize}")

    if not should_summarize:
        func_logger.debug(f"Summarization not triggered (Dialogue tokens {total_unsummarized_dialogue_tokens} <= threshold {summarization_trigger_threshold}).")
        return False, None, new_last_summary_index, -1, -1

    func_logger.info(f"Summarization triggered (Dialogue tokens {total_unsummarized_dialogue_tokens} > threshold {summarization_trigger_threshold}).")
    # Record the index of the last message in the full history *at the time of triggering*
    t0_end_index_at_summary = len(active_history) - 1


    # --- Identify T0 and T1 Chunks (From Dialogue Messages) ---
    func_logger.debug(f"Identifying T0 slice within unsummarized DIALOGUE block (target: {t0_token_limit} tokens)...")
    # Use the helper that now filters roles internally
    t0_dialogue_slice = _select_history_slice_by_tokens(
        messages=unsummarized_dialogue_messages, # Pass the filtered dialogue messages
        target_tokens=t0_token_limit,
        tokenizer=tokenizer,
        include_last=True, # T0 usually includes the latest relevant dialogue
        dialogue_only_roles=dialogue_only_roles # Pass roles just in case helper needs it
    )

    func_logger.debug(f"DEBUG T0 SLICE: Identified {len(t0_dialogue_slice)} dialogue messages for T0 slice.")
    if t0_dialogue_slice:
        t0_first_role = t0_dialogue_slice[0].get('role','?')
        t0_last_role = t0_dialogue_slice[-1].get('role','?')
        func_logger.debug(f"DEBUG T0 SLICE: First role='{t0_first_role}', Last role='{t0_last_role}'.")

    if not t0_dialogue_slice:
        # This might happen if the dialogue messages are fewer than target tokens,
        # but summarization was still triggered (e.g., large T1 target).
        # Or if _select_history_slice_by_tokens failed.
        func_logger.warning("Could not select T0 dialogue slice, but summarization was triggered. Proceeding to summarize potentially the whole unsummarized dialogue block.")
        # If T0 is empty, the T1 chunk becomes all the unsummarized dialogue
        t1_chunk_dialogue_messages = unsummarized_dialogue_messages
        func_logger.debug(f"DEBUG T1 CHUNK: T0 slice empty. Setting T1 chunk to all {len(t1_chunk_dialogue_messages)} unsummarized dialogue messages.")
    else:
        # Find the first message of the T0 slice within the unsummarized dialogue list
        first_t0_message = t0_dialogue_slice[0]
        try:
            # Find its index in the *dialogue-only* list
            t1_chunk_end_index_relative_dialogue = unsummarized_dialogue_messages.index(first_t0_message)
            func_logger.debug(f"DEBUG T1 CHUNK: First message of T0 slice found at relative index {t1_chunk_end_index_relative_dialogue} within unsummarized dialogue messages.")

            if t1_chunk_end_index_relative_dialogue > 0:
                 # The T1 chunk consists of dialogue messages *before* the T0 slice starts
                 t1_chunk_dialogue_messages = unsummarized_dialogue_messages[:t1_chunk_end_index_relative_dialogue]
                 func_logger.info(f"Identified T1 chunk: {len(t1_chunk_dialogue_messages)} dialogue messages to summarize (indices 0 to {t1_chunk_end_index_relative_dialogue-1} relative to unsummarized dialogue).")
            else:
                 # T0 slice started at the beginning of unsummarized dialogue. No separate T1 chunk.
                 func_logger.info(f"T0 dialogue slice started at the beginning (relative index {t1_chunk_end_index_relative_dialogue}) of unsummarized dialogue messages. No separate T1 chunk to summarize yet.")
                 return False, None, new_last_summary_index, -1, t0_end_index_at_summary # Not an error, just nothing to summarize *yet*
        except ValueError:
             # Should not happen if t0_dialogue_slice came from unsummarized_dialogue_messages
             func_logger.error("CRITICAL: Could not find start of T0 dialogue slice within unsummarized dialogue messages. Aborting.", exc_info=True)
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary
        except Exception as e:
             func_logger.error(f"Error determining T1 dialogue chunk indices: {e}", exc_info=True)
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary

    if not t1_chunk_dialogue_messages:
        # Should be caught above, but double check
        func_logger.error("Identified T1 dialogue chunk is empty. Aborting summarization.")
        return False, None, new_last_summary_index, -1, t0_end_index_at_summary


    # --- Perform Summarization (on T1 Dialogue Chunk) ---
    try:
        # --- Determine Absolute Index for Metadata ---
        # Find the actual message object corresponding to the last message of the T1 dialogue chunk
        # This requires searching the *original* full active_history
        last_msg_in_t1_chunk = t1_chunk_dialogue_messages[-1]
        t1_chunk_end_index_absolute = -1
        try:
            # Find the index of this last message within the *original* full history
            t1_chunk_end_index_absolute = active_history.index(last_msg_in_t1_chunk)
            # The start index is simply the index after the previous summary
            t1_chunk_start_index_absolute = original_last_summary_index + 1
            func_logger.info(f"Summarizing T1 dialogue chunk (Abs Indices in active_history: {t1_chunk_start_index_absolute} to {t1_chunk_end_index_absolute}).")
        except ValueError:
             func_logger.error("CRITICAL: Could not map end of T1 dialogue chunk back to original active_history index. Metadata will be incorrect. Aborting summary.", exc_info=True)
             # If we can't reliably get the end index, we shouldn't update last_summary_turn_index
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary

        # --- Format and Call LLM ---
        # Format only the dialogue messages identified for the T1 chunk
        formatted_t1_dialogue_chunk = format_history_for_llm(
            t1_chunk_dialogue_messages, allowed_roles=dialogue_only_roles # Ensure only dialogue is formatted
        )
        if not formatted_t1_dialogue_chunk or not formatted_t1_dialogue_chunk.strip():
             func_logger.warning("Formatted T1 dialogue chunk is empty after filtering/formatting. Skipping LLM call.")
             return False, None, new_last_summary_index, -1, t0_end_index_at_summary

        summarizer_sys_prompt = llm_config.get('sys_prompt', "Summarize this dialogue.")
        # Construct the prompt using the formatted dialogue chunk
        prompt = f"{summarizer_sys_prompt}\n\n--- Dialogue History Chunk ---\n{formatted_t1_dialogue_chunk}\n\n--- End Dialogue History Chunk ---\n\nConcise Summary:"

        # Calculate prompt tokens (optional but good for stats)
        summarizer_prompt_tokens = -1
        try:
            summarizer_prompt_tokens = len(tokenizer.encode(prompt))
            func_logger.debug(f"Summarizer Payload tokens: {summarizer_prompt_tokens}")
        except Exception as e_tok:
            func_logger.error(f"Tokenizer error on Summarizer Prompt: {e_tok}", exc_info=False)

        # Prepare payload and call LLM
        summ_payload = {"contents": [{"parts": [{"text": prompt}]}]}
        func_logger.info("Calling Summarizer LLM via async llm_call_func...")
        # Use the corrected signature from type hint
        success, result_or_error = await llm_call_func(
             api_url=llm_config['url'], api_key=llm_config['key'], payload=summ_payload,
             temperature=llm_config['temp'], timeout=120, caller_info="i4_llm_agent_Summarizer",
        )

        # --- Process Summarization Result ---
        if success and isinstance(result_or_error, str) and result_or_error.strip():
            summary_result_text = result_or_error
            func_logger.info("Summary generated successfully by LLM.")
            generated_summary = summary_result_text.strip() # Clean the summary
            summary_id = f"t1_sum_{uuid.uuid4()}"
            now = datetime.now(timezone.utc)
            # Prepare metadata using the *absolute* indices from the original active_history
            metadata = {
                "session_id": session_id, "user_id": user_id,
                "timestamp_utc": now.timestamp(), "timestamp_iso": now.isoformat(),
                "turn_start_index": t1_chunk_start_index_absolute, # Correct start index
                "turn_end_index": t1_chunk_end_index_absolute,     # Correct end index
                "char_length": len(generated_summary), "doc_type": "llm_summary",
                "config_t0_token_limit": t0_token_limit,
                "config_t1_chunk_target": t1_chunk_size_target,
                "calculated_prompt_tokens": summarizer_prompt_tokens,
                "t0_end_index_at_summary": t0_end_index_at_summary, # Index when trigger occurred
            }

            func_logger.info(f"Attempting to save T1 summary {summary_id} via callback...")
            # Call the provided async function to save the summary
            save_successful = await add_t1_summary_func(
                summary_id=summary_id, session_id=session_id, user_id=user_id,
                summary_text=generated_summary, metadata=metadata
            )
            if save_successful:
                func_logger.info(f"T1 summary {summary_id} saved successfully.")
                summarization_performed = True
                # CRITICAL: Set the new index to be *returned* to the caller
                new_last_summary_index = t1_chunk_end_index_absolute
                # REMOVED: Direct state update (caller must handle this)
                # session_state["last_summary_turn_index"] = new_last_summary_index
            else:
                func_logger.error(f"Failed to save T1 summary {summary_id} via callback. Last summary index not updated.")
                # Do NOT update new_last_summary_index if save failed

        elif success and (not isinstance(result_or_error, str) or not result_or_error.strip()):
             # Handle cases where LLM call succeeded but returned empty or wrong type
             func_logger.error(f"Summarization failed (LLM call succeeded but returned empty/invalid content). Result Type: {type(result_or_error)}")
        elif not success and isinstance(result_or_error, dict):
            # Log specific error dict from LLM call
            error_type = result_or_error.get('error_type', 'Unknown')
            error_msg = result_or_error.get('message', 'No details provided')
            func_logger.error(f"Summarizer LLM call failed: Type='{error_type}', Message='{error_msg}'.")
        else:
            # Log generic failure if result was None or unexpected type
            func_logger.error(f"Summarization failed (LLM call failed). Result: '{result_or_error}'")

    except Exception as e:
        func_logger.error(f"Unexpected error during manage_tier1_summarization execution block: {e}", exc_info=True)
        # Ensure we don't incorrectly report success
        summarization_performed = False
        generated_summary = None
        # Keep new_last_summary_index as it was before this failed attempt (it defaults to original_last_summary_index)

    # --- Return Results ---
    func_logger.debug(
        f"Exiting manage_tier1_summarization. "
        f"Success: {summarization_performed}, "
        f"New T1 Idx: {new_last_summary_index}, " # Return the potentially updated index
        f"Prompt Tok: {summarizer_prompt_tokens}, "
        f"T0 End Idx: {t0_end_index_at_summary}"
        )
    # Return the calculated new index (or the original one if summary failed/skipped).
    # The caller is responsible for updating the SessionManager state with this index.
    return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens, t0_end_index_at_summary