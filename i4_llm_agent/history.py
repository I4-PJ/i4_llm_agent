# i4_llm_agent/history.py
import logging
from typing import List, Dict, Optional, Any
import json # For pretty printing debug output

logger = logging.getLogger(__name__) # Gets logger named 'i4_llm_agent.history'

def format_history_for_llm(
    history_chunk: List[Dict], max_messages: Optional[int] = None
) -> str:
    """
    Formats a list of message dictionaries (like OWI's) into a simple
    'Role: Content' string, suitable for LLM context/prompts.

    Args:
        history_chunk: A list of message dictionaries, each expected
                       to have 'role' and 'content' keys.
        max_messages: If provided, only formats the last N messages.

    Returns:
        A single string with messages formatted and joined by newlines.
        Returns an empty string if the input chunk is empty or formatting results in nothing.
    """
    if not history_chunk:
        return ""

    lines = []
    start_idx = -max_messages if max_messages is not None and max_messages > 0 else 0
    chunk_to_format = history_chunk[start_idx:] # Slice the chunk if max_messages is set

    # logger.debug(f"Formatting {len(chunk_to_format)} messages (Max: {max_messages}).") # Less verbose

    for msg in chunk_to_format:
        role = msg.get('role', 'unk').capitalize() # Default to 'Unk' if role missing
        content = msg.get('content', '').strip()   # Default to empty string if content missing
        if not content: # Optionally skip empty messages
            continue
        lines.append(f"{role}: {content}")

    formatted_string = "\n".join(lines)
    return formatted_string


def get_recent_turns(
    messages: List[Dict],
    count: int,
    roles: List[str] = ["user", "assistant"],
    exclude_last: bool = True
) -> List[Dict]:
    """
    Extracts the most recent 'count' messages matching the specified roles.
    Optionally excludes the very last message in the list.

    Args:
        messages: The list of message dictionaries.
        count: The maximum number of messages to return.
        roles: A list of roles to include (e.g., ['user', 'assistant']).
        exclude_last: If True, the very last message in the original list is excluded
                      before selecting the count.

    Returns:
        A list containing the selected recent messages, preserving order.
    """
    # logger.debug(f"Extracting up to {count} recent turns (roles: {roles}, exclude_last: {exclude_last})...") # Less verbose
    if count <= 0 or not messages:
        return []

    source_list = messages[:-1] if exclude_last and len(messages) > 0 else messages
    filtered_messages = [
        msg for msg in source_list if isinstance(msg, dict) and msg.get("role") in roles
    ]

    if not filtered_messages:
        return []

    start_index = max(0, len(filtered_messages) - count)
    recent_history = filtered_messages[start_index:]

    # logger.debug(f"Extracted {len(recent_history)} recent turns.") # Less verbose
    return recent_history


def get_dialogue_history(
    messages: List[Dict],
    roles: List[str] = ["user", "assistant"],
    exclude_last: bool = True
) -> List[Dict]:
    """
    Extracts all messages matching the specified roles from the list.
    Optionally excludes the very last message.

    Args:
        messages: The list of message dictionaries.
        roles: A list of roles to include.
        exclude_last: If True, the very last message in the original list is excluded.

    Returns:
        A list containing all matching messages, preserving order.
    """
    # logger.debug(f"Extracting dialogue history (roles: {roles}, exclude_last: {exclude_last})...") # Less verbose
    if not messages:
        return []

    source_list = messages[:-1] if exclude_last and len(messages) > 0 else messages
    filtered_messages = [
        msg for msg in source_list if isinstance(msg, dict) and msg.get("role") in roles
    ]

    # logger.debug(f"Extracted {len(filtered_messages)} dialogue history messages.") # Less verbose
    return filtered_messages

# ==============================================================================
# === Turn-Aware T0 History Selection (with Debug Logging)                   ===
# ==============================================================================
def select_turns_for_t0(
    full_history: List[Dict],
    target_tokens: int,
    tokenizer: Any,
    max_overflow_ratio: float = 1.15,
    fallback_turns: int = 10
) -> List[Dict]:
    """
    Selects recent messages for T0 context, aiming for a token limit,
    but prioritizing inclusion of the preceding user turn for the earliest
    assistant turn if it fits within a defined overflow ratio. Includes detailed debug logging.
    """
    # Use specific logger for this function for targeted debugging
    func_logger = logging.getLogger(__name__ + '.select_turns_for_t0')
    # Ensure logger level is appropriate (e.g., set to DEBUG in pipe)
    is_debug_enabled = func_logger.isEnabledFor(logging.DEBUG)

    if not full_history:
        func_logger.debug("[T0 Select] Input history is empty, returning empty T0 slice.")
        return []

    # --- Fallback if no tokenizer ---
    if not tokenizer:
        func_logger.warning(
            f"[T0 Select] Tokenizer unavailable. Falling back to last {fallback_turns} turns."
        )
        start_idx = max(0, len(full_history) - fallback_turns)
        return full_history[start_idx:]

    # --- Log Initial Parameters ---
    if is_debug_enabled:
        func_logger.debug(f"[T0 Select Init] Received {len(full_history)} history messages.")
        func_logger.debug(f"[T0 Select Init] Target tokens: {target_tokens}")
        func_logger.debug(f"[T0 Select Init] Max overflow ratio: {max_overflow_ratio}")
        # Optionally log first/last message snippet of input history
        if full_history:
            first_msg_snippet = str(full_history[0].get('content', ''))[:50]
            last_msg_snippet = str(full_history[-1].get('content', ''))[:50]
            func_logger.debug(f"[T0 Select Init] Input History First Msg Snippet: '{first_msg_snippet}...'")
            func_logger.debug(f"[T0 Select Init] Input History Last Msg Snippet: '{last_msg_snippet}...'")

    # --- Main selection loop (iterating backwards) ---
    selected_history: List[Dict] = []
    current_tokens: int = 0
    earliest_selected_index: int = -1

    if is_debug_enabled: func_logger.debug("[T0 Select Loop] Starting backward iteration...")

    for i in range(len(full_history) - 1, -1, -1):
        msg = full_history[i]
        msg_content = msg.get("content", "")
        msg_role = msg.get("role", "unknown") # Get role for logging

        if not msg_content: # Skip messages with no content
            if is_debug_enabled: func_logger.debug(f"[T0 Select Loop] Skipping index {i} (role: {msg_role}): Empty content.")
            continue

        msg_tokens = 0
        try:
            # Use a temporary variable in case encode fails
            encoded_tokens = tokenizer.encode(msg_content)
            msg_tokens = len(encoded_tokens)
            if is_debug_enabled: func_logger.debug(f"[T0 Select Loop] Index {i} (role: {msg_role}): Calculated tokens = {msg_tokens}")
        except Exception as e:
            func_logger.error(f"[T0 Select Loop] Tokenizer error on msg index {i} (role: {msg_role}): {e}. Skipping msg.", exc_info=False) # Less verbose traceback
            continue # Skip this message if tokenization fails

        # Check token limit BEFORE adding the current message
        # If selected_history is not empty AND adding this msg would exceed limit, break
        if selected_history and (current_tokens + msg_tokens > target_tokens):
            if is_debug_enabled:
                func_logger.debug(
                    f"[T0 Select Loop] Token limit {target_tokens} would be exceeded by adding msg index {i} "
                    f"(current: {current_tokens}, msg: {msg_tokens}, total: {current_tokens + msg_tokens}). "
                    f"Stopping initial selection BEFORE adding index {i}."
                )
            break # Stop BEFORE adding this message

        # Prepend the message to maintain order and update state
        selected_history.insert(0, msg)
        current_tokens += msg_tokens
        earliest_selected_index = i
        if is_debug_enabled: func_logger.debug(f"[T0 Select Loop] Added index {i}. New total tokens: {current_tokens}. Current slice length: {len(selected_history)}")

    # --- Post-Loop Logging & Checks ---
    if not selected_history:
         func_logger.warning("[T0 Select PostLoop] Initial selection resulted in an empty slice.")
         return [] # Return empty if nothing was selected

    if is_debug_enabled:
        func_logger.debug(f"[T0 Select PostLoop] Initial selection finished.")
        func_logger.debug(f"[T0 Select PostLoop] Selected {len(selected_history)} messages.")
        func_logger.debug(f"[T0 Select PostLoop] Current tokens: {current_tokens}")
        func_logger.debug(f"[T0 Select PostLoop] Earliest selected index in full history: {earliest_selected_index}")
        # Log the content of the earliest selected message for context
        if selected_history:
            earliest_msg_role = selected_history[0].get('role', 'unknown')
            earliest_msg_snippet = str(selected_history[0].get('content', ''))[:100]
            func_logger.debug(f"[T0 Select PostLoop] Earliest selected message (role: {earliest_msg_role}): '{earliest_msg_snippet}...'")

    # --- Post-loop: Check for Turn Completion ---
    first_selected_msg = selected_history[0]
    first_selected_role = first_selected_msg.get("role")

    # Only check if the earliest message selected is 'assistant' and not the very first message overall
    if first_selected_role == "assistant" and earliest_selected_index > 0:
        if is_debug_enabled: func_logger.debug("[T0 Turn Check] Earliest selected msg is 'assistant', checking preceding turn.")
        preceding_index = earliest_selected_index - 1
        preceding_msg = full_history[preceding_index]
        preceding_role = preceding_msg.get("role")

        if preceding_role == "user":
            if is_debug_enabled: func_logger.debug(f"[T0 Turn Check] Found preceding 'user' message at index {preceding_index}.")
            preceding_content = preceding_msg.get("content", "")
            preceding_tokens = 0
            if preceding_content:
                try:
                    preceding_tokens = len(tokenizer.encode(preceding_content))
                except Exception as e:
                    func_logger.error(f"[T0 Turn Check] Tokenizer error on preceding user msg index {preceding_index}: {e}", exc_info=False)
                    preceding_tokens = -1 # Indicate error

            if preceding_tokens >= 0:
                overflow_limit = target_tokens * max_overflow_ratio
                projected_tokens = current_tokens + preceding_tokens

                if is_debug_enabled:
                    func_logger.debug(
                        f"[T0 Turn Check] Preceding user tokens: {preceding_tokens}. "
                        f"Current slice tokens: {current_tokens}. "
                        f"Projected total: {projected_tokens}. "
                        f"Overflow limit ({max_overflow_ratio:.2f}x): {overflow_limit:.0f}"
                    )

                if projected_tokens <= overflow_limit:
                    func_logger.info( # Log as INFO if we actually modify the list
                        f"[T0 Turn Check] Including preceding 'user' turn (index {preceding_index}) as it fits within overflow limit."
                    )
                    selected_history.insert(0, preceding_msg)
                    current_tokens = projected_tokens
                else:
                    if is_debug_enabled: func_logger.debug("[T0 Turn Check] Preceding 'user' turn excluded as it exceeds overflow limit.")
            else:
                if is_debug_enabled: func_logger.debug("[T0 Turn Check] Preceding 'user' turn tokenization failed, cannot check overflow.")
        else:
            if is_debug_enabled: func_logger.debug(f"[T0 Turn Check] Preceding message (index {preceding_index}) is not 'user' (role: {preceding_role}). No turn completion check needed.")
    elif first_selected_role != "assistant":
        if is_debug_enabled: func_logger.debug(f"[T0 Turn Check] Earliest selected msg is '{first_selected_role}', no preceding turn check needed.")
    else: # earliest_selected_index must be 0
         if is_debug_enabled: func_logger.debug("[T0 Turn Check] Earliest selected T0 msg is the first message in history, no preceding turn check needed.")


    # --- Final Logging and Return ---
    final_count = len(selected_history)
    func_logger.info(f"Final T0 slice selected: {final_count} messages, approx {current_tokens} tokens.")
    if is_debug_enabled and not selected_history:
         func_logger.debug("[T0 Select Return] Returning EMPTY list.") # Explicit log if empty
    elif is_debug_enabled:
         # Log final selection summary
         final_roles = [msg.get("role", "unk") for msg in selected_history]
         func_logger.debug(f"[T0 Select Return] Final slice roles: {final_roles}")
         # Optionally log full content if needed, but can be very verbose
         # func_logger.debug(f"[T0 Select Return] Final slice content:\n{json.dumps(selected_history, indent=2)}")

    return selected_history