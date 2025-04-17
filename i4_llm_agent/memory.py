# i4_llm_agent/memory.py

import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Callable, Coroutine, Tuple

# --- Internal Dependencies ---
# Attempt to import history formatter from within the library
try:
    from .history import format_history_for_llm
    HISTORY_FORMATTER_AVAILABLE = True
except ImportError:
    HISTORY_FORMATTER_AVAILABLE = False
    # Define a fallback inline if needed, or raise error if critical
    def format_history_for_llm(history_chunk: List[Dict], max_messages: Optional[int] = None) -> str:
        # This fallback should ideally log a persistent warning or be removed if the dep is mandatory
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
logger = logging.getLogger(__name__) # Use library's logger: 'i4_llm_agent.memory'

# --- Helper Function: Select History by Tokens ---
# Keep this internal to the memory module for now
def _select_history_slice_by_tokens(
    messages: List[Dict],
    target_tokens: int,
    tokenizer: Any, # Expects a tokenizer with an .encode() method
    include_last: bool = True,
    fallback_turns: int = 10 # Fallback if no tokenizer
) -> List[Dict]:
    """
    Selects recent messages aiming for a token limit, iterating backwards.

    Args:
        messages: The list of message dictionaries.
        target_tokens: The desired approximate token count.
        tokenizer: A tokenizer instance with an .encode() method.
        include_last: Whether to consider the very last message in the input list.
        fallback_turns: How many turns to take if tokenizer is unavailable.

    Returns:
        A list of selected message dictionaries, preserving original order.
    """
    # Add basic logging within this helper
    func_logger = logging.getLogger(__name__ + '._select_history_slice_by_tokens')

    if not messages:
        return []

    source_list = messages if include_last else messages[:-1]
    if not source_list:
        return []

    if not tokenizer:
        func_logger.warning(
            f"Tokenizer unavailable. Falling back to last {fallback_turns} turns for slice selection."
        )
        start_idx = max(0, len(source_list) - fallback_turns)
        return source_list[start_idx:]

    selected_history: List[Dict] = []
    current_tokens: int = 0
    stop_adding = False

    try:
        # Iterate backwards through the source list
        for i in range(len(source_list) - 1, -1, -1):
            if stop_adding:
                break # Stop adding once limit is breached

            msg = source_list[i]
            msg_content = msg.get("content", "")
            if not msg_content: # Skip messages with no content
                continue

            msg_tokens = 0
            try:
                # Use the provided tokenizer's encode method
                msg_tokens = len(tokenizer.encode(msg_content))
            except Exception as e:
                func_logger.error(f"Tokenizer error during selection on msg index {i}: {e}. Skipping msg.")
                continue # Skip this message if tokenization fails

            # Check if adding this message would exceed the target
            # Add a small buffer (e.g., 10%) to target? Optional.
            # token_limit_check = target_tokens * 1.10
            if selected_history and (current_tokens + msg_tokens > target_tokens):
                # func_logger.debug(
                #     f"Token limit (~{target_tokens}) would be breached. "
                #     f"Current: {current_tokens}, NextMsg: {msg_tokens}. Stopping before index {i}."
                # )
                stop_adding = True
                continue # Don't add this message, but continue loop to potentially add earlier ones if needed (though unlikely with token growth)

            # Prepend the message to maintain order
            selected_history.insert(0, msg)
            current_tokens += msg_tokens

    except Exception as e:
        func_logger.error(f"Unexpected error during token selection loop: {e}", exc_info=True)
        # Return whatever was selected so far, or empty list
        return selected_history if selected_history else []

    func_logger.debug(
        f"Selected {len(selected_history)} messages, "
        f"approx {current_tokens} tokens (target: {target_tokens}) for slice."
    )
    return selected_history


# --- Main Memory Management Function ---
async def manage_tier1_summarization(
    session_state: Dict,
    active_history: List[Dict],
    t0_token_limit: int,
    t1_chunk_size_target: int,
    tokenizer: Any,  # Expects object with .encode()
    llm_call_func: Callable[..., Coroutine[Any, Any, Optional[str]]], # Expect async LLM call func now
    llm_config: Dict[str, Any], # Expects keys like url, key, temp, sys_prompt
    add_t1_summary_func: Callable[..., Coroutine[Any, Any, bool]], # The async DB save function
    session_id: str, # Needed for saving metadata
    user_id: str # Needed for saving metadata
) -> Tuple[bool, Optional[str], int, int]:
    """
    Checks if T1 summarization is needed based on token limits, performs
    summarization if required using an async LLM call, and saves the result
    via an async callback.

    Manages history slicing based on token limits to define T0 (kept active)
    and T1 (to be summarized) message chunks.

    Args:
        session_state: The current session state dictionary (must contain 'last_summary_turn_index').
        active_history: The full list of messages for the session.
        t0_token_limit: The target token limit for T0 history (recent turns kept active).
        t1_chunk_size_target: The target token size threshold that, when added to t0_token_limit,
                             triggers summarization of the oldest unsummarized messages.
        tokenizer: The Tiktoken (or compatible) tokenizer instance with .encode().
        llm_call_func: An ASYNCHRONOUS function to call the summarizer LLM API.
                       Expected signature: async def func(...) -> Optional[str]
                       Must handle API calls internally (e.g., using aiohttp or asyncio.to_thread).
        llm_config: Dict with necessary LLM parameters (url, key, temp, sys_prompt).
        add_t1_summary_func: ASYNCHRONOUS function to save the summary.
                             Expected signature: async def func(summary_id, session_id,
                                                    user_id, summary_text, metadata) -> bool
        session_id: The session ID for metadata.
        user_id: The user ID for metadata.

    Returns:
        Tuple containing:
        - bool: True if summarization was successfully performed and saved, False otherwise.
        - Optional[str]: The generated summary text if successful, None otherwise.
        - int: The new last_summary_turn_index to update session_state with.
               Returns the original index from session_state if no summarization occurred or failed.
        - int: Tokens used in the summarizer prompt (-1 if unavailable/failed).
    """
    logger.debug("Entering manage_tier1_summarization function...")
    original_last_summary_index = session_state.get("last_summary_turn_index", -1)
    summarization_performed = False
    generated_summary = None
    new_last_summary_index = original_last_summary_index
    summarizer_prompt_tokens = -1 # Initialize token counter

    # --- Prerequisites Check ---
    if not tokenizer:
        logger.warning("Tokenizer unavailable, cannot perform T1 summarization check.")
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
         logger.error("Async LLM Call Function unavailable or not async, cannot perform T1 summarization.")
         return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens
    if not add_t1_summary_func or not asyncio.iscoroutinefunction(add_t1_summary_func):
         logger.error("Async Add T1 Summary Function unavailable or not async, cannot save T1 summary.")
         return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens
    # Check specific keys expected in llm_config
    required_llm_keys = ['url', 'key', 'temp', 'sys_prompt']
    if not all(key in llm_config for key in required_llm_keys):
         missing_keys = [key for key in required_llm_keys if key not in llm_config]
         logger.error(f"LLM Config dictionary is missing required keys: {missing_keys}. Config provided: {llm_config}")
         return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens
    if not HISTORY_FORMATTER_AVAILABLE: # Check if formatter loaded
        logger.error("History formatter function unavailable, cannot format T1 chunk.")
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens

    # --- Calculate current unsummarized tokens ---
    current_unsummarized_messages = []
    if original_last_summary_index < len(active_history) - 1:
        # Get the slice of history *after* the last summarized message index
        current_unsummarized_messages = active_history[original_last_summary_index + 1 :]
    else:
         logger.debug("No new messages since last summary index.")
         return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens # Nothing to do

    if not current_unsummarized_messages:
         logger.debug("Unsummarized message slice is empty.")
         return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens # Nothing to do


    total_unsummarized_tokens = 0
    try:
        # Estimate token count based on combined content of the unsummarized part
        # This is an approximation; could tokenize message by message for accuracy
        combined_text = " ".join([msg.get("content", "") for msg in current_unsummarized_messages if msg.get("content")])
        if combined_text:
            total_unsummarized_tokens = len(tokenizer.encode(combined_text))
            logger.debug(f"Estimated total unsummarized tokens: {total_unsummarized_tokens}")
        else:
            logger.debug("No content in unsummarized messages to count tokens.")
            total_unsummarized_tokens = 0

    except Exception as e:
        logger.error(f"Tokenizer error calculating total unsummarized tokens: {e}", exc_info=True)
        # Cannot proceed if token count fails, as trigger condition relies on it
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens

    # --- Check Trigger Condition ---
    # Trigger ONLY if the total unsummarized tokens exceed the size needed for T0 PLUS the target size for a T1 chunk.
    # This ensures we have enough history to potentially split off a T1 chunk.
    summarization_trigger_threshold = t0_token_limit + t1_chunk_size_target
    should_summarize = (
        total_unsummarized_tokens > summarization_trigger_threshold
        # Optional: Add a minimum message count? e.g., len(current_unsummarized_messages) > 5
    )

    if not should_summarize:
        logger.debug(f"Summarization not triggered. Total tokens {total_unsummarized_tokens} <= Threshold {summarization_trigger_threshold}.")
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens

    logger.info(f"Summarization triggered ({total_unsummarized_tokens} > {summarization_trigger_threshold}).")

    # --- Identify T0 and T1 Chunks from the `current_unsummarized_messages` ---
    # Goal: Select the most recent messages for T0 (up to t0_token_limit).
    # The messages *before* that slice within the unsummarized block become T1.
    logger.debug(f"Identifying T0 slice within unsummarized block (target tokens: {t0_token_limit})...")
    t0_messages_slice = _select_history_slice_by_tokens(
        messages=current_unsummarized_messages,
        target_tokens=t0_token_limit,
        tokenizer=tokenizer,
        include_last=True, # T0 includes the very latest messages
    )

    if not t0_messages_slice:
        # This shouldn't happen if current_unsummarized_messages is not empty, but handle defensively
        logger.error("Failed to select T0 history slice even though trigger met. Aborting summarization cycle.")
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens

    # Find the first message of the T0 slice to determine the split point
    first_t0_message = t0_messages_slice[0]
    t1_chunk_messages = []
    t1_chunk_end_index_relative = -1 # Index within the current_unsummarized_messages list

    try:
        # Find where the T0 slice *starts* within the unsummarized block
        t1_chunk_end_index_relative = current_unsummarized_messages.index(first_t0_message)

        # The T1 chunk is everything *before* the start of the T0 slice
        if t1_chunk_end_index_relative > 0:
            t1_chunk_messages = current_unsummarized_messages[0:t1_chunk_end_index_relative]
            logger.info(f"Identified T1 chunk: {len(t1_chunk_messages)} messages to summarize.")
        else:
            # This means the T0 slice starts at the very beginning of the unsummarized block.
            # This implies there isn't a separate chunk *before* it large enough to summarize yet.
            # This might happen if t0_token_limit is very large or t1_chunk_size_target is small relative to message sizes.
            logger.info("T0 slice includes all unsummarized messages. No separate T1 chunk identifiable yet. Skipping summarization cycle.")
            return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens # Abort this attempt

    except ValueError:
        # This is a critical logic error if the message isn't found
        logger.error("CRITICAL: Could not find start of T0 slice within unsummarized messages list. Aborting summarization.", exc_info=True)
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens
    except Exception as e:
        logger.error(f"Error determining T1 chunk indices: {e}", exc_info=True)
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens

    if not t1_chunk_messages:
        # Should be caught by the logic above, but double-check
        logger.error("Identified T1 chunk is empty after index calculation. Aborting summarization cycle.")
        return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens

    # --- Perform Summarization ---
    try:
        # Calculate absolute indices in the *full active_history* for storing in DB metadata
        # The T1 chunk starts right after the previous summary ends
        t1_chunk_start_index_absolute = original_last_summary_index + 1
        # The end index is the index of the *last message included* in the T1 chunk (absolute)
        # original_last_summary_index + 1 (start of unsummarized) + t1_chunk_end_index_relative - 1 (end of T1 slice)
        t1_chunk_end_index_absolute = original_last_summary_index + t1_chunk_end_index_relative

        logger.info(
            f"Summarizing T1 chunk: {len(t1_chunk_messages)} messages "
            f"(Abs Indices {t1_chunk_start_index_absolute} to {t1_chunk_end_index_absolute})."
        )
        # Format the identified T1 chunk for the LLM
        formatted_t1_chunk = format_history_for_llm(t1_chunk_messages) # Use the library's formatter

        if not formatted_t1_chunk or not formatted_t1_chunk.strip():
             logger.warning("Formatted T1 chunk for summarization is empty. Skipping LLM call.")
             # Return 0 tokens? Or -1 as it wasn't calculated/used? Let's use -1.
             return summarization_performed, generated_summary, new_last_summary_index, -1

        # Construct the summarization prompt
        summarizer_sys_prompt = llm_config.get('sys_prompt', "You are an expert summarizer.") # Use configured sys prompt
        prompt = f"{summarizer_sys_prompt}\\n\\n--- Dialogue History Chunk ---\\n{formatted_t1_chunk}\\n\\n--- End Dialogue History Chunk ---\\n\\nConcise Summary:"

        # Calculate Summarizer Prompt Tokens BEFORE the call
        try:
            summarizer_prompt_tokens = len(tokenizer.encode(prompt))
            logger.debug(f"Summarizer Payload tokens calculated: {summarizer_prompt_tokens}")
        except Exception as e_tok:
            logger.error(f"Tokenizer error on Summarizer Prompt: {e_tok}", exc_info=True)
            summarizer_prompt_tokens = -1 # Indicate failure

        # Prepare payload for the LLM call
        summ_payload = {"contents": [{"parts": [{"text": prompt}]}]}
        # Temperature is handled by the call_google_llm_api function now

        logger.info("Calling Summarizer LLM via async llm_call_func...")
        # Use the provided async function directly
        summary_result_text = await llm_call_func(
             api_url=llm_config['url'],
             api_key=llm_config['key'],
             payload=summ_payload,
             temperature=llm_config['temp'],
             timeout=120, # Configurable? Default 120s for summarization
             caller_info="i4_llm_agent_Summarizer", # Origin Tag for LLM call
        )

        # --- Process Summarization Result ---
        if (
            isinstance(summary_result_text, str)
            and not summary_result_text.startswith("[Error:") # Check for API client errors
            and summary_result_text.strip() # Check for non-empty content
        ):
            logger.info("Summary generated successfully by LLM.")
            generated_summary = summary_result_text.strip() # Store the summary
            summary_id = f"t1_sum_{uuid.uuid4()}"
            now = datetime.now(timezone.utc)

            # Prepare metadata including token sizes from config/trigger logic
            metadata = {
                "session_id": session_id,
                "user_id": user_id,
                "timestamp_utc": now.timestamp(),
                "timestamp_iso": now.isoformat(),
                "turn_start_index": t1_chunk_start_index_absolute,
                "turn_end_index": t1_chunk_end_index_absolute, # Index of last message *in* the summarized chunk
                "char_length": len(generated_summary),
                "doc_type": "llm_summary",
                # Store config values used for this summary generation
                "config_t0_token_limit": t0_token_limit,
                "config_t1_chunk_target": t1_chunk_size_target,
                "calculated_prompt_tokens": summarizer_prompt_tokens, # Store calculated tokens
            }

            logger.info(f"Attempting to save T1 summary {summary_id} via async add_t1_summary_func callback...")
            # Call the ASYNC callback function provided by the Pipe to save the summary
            save_successful = await add_t1_summary_func(
                summary_id=summary_id,
                session_id=session_id,
                user_id=user_id,
                summary_text=generated_summary,
                metadata=metadata
            )

            if save_successful:
                logger.info(f"T1 summary {summary_id} saved successfully via callback.")
                summarization_performed = True # Mark success
                # Update the index to the end of the chunk we just successfully summarized and saved
                new_last_summary_index = t1_chunk_end_index_absolute
            else:
                logger.error(f"Failed to save T1 summary {summary_id} using the provided callback function. State not updated.")
                # Do NOT set summarization_performed = True or update index if save fails

        elif isinstance(summary_result_text, str) and summary_result_text.startswith("[Error:"):
            # LLM call function returned a specific error string
            logger.error(f"Summarizer LLM call failed: {summary_result_text}")
            # summarization_performed remains False
        else: # LLM returned None or unexpected type (e.g., empty string after stripping)
            logger.error(f"Summarization failed (LLM returned None or empty content). Result: '{summary_result_text}'")
            # summarization_performed remains False

    except Exception as e:
        logger.error(f"Unexpected error during manage_tier1_summarization execution block: {e}", exc_info=True)
        # Ensure default failure state is returned; prompt tokens might be -1 or calculated value

    logger.debug(f"Exiting manage_tier1_summarization. Success: {summarization_performed}, New Index: {new_last_summary_index}, Prompt Tokens: {summarizer_prompt_tokens}")
    # Return the results, including the new index and calculated prompt tokens
    return summarization_performed, generated_summary, new_last_summary_index, summarizer_prompt_tokens
