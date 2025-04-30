# i4_llm_agent/history.py
import logging
from typing import List, Dict, Optional, Any
import json # For pretty printing debug output

logger = logging.getLogger(__name__) # Gets logger named 'i4_llm_agent.history'

# --- Constants ---
DIALOGUE_ROLES = ["user", "assistant"] # Roles considered part of the conversation history


# --- Formatting Function ---
def format_history_for_llm(
    history_chunk: List[Dict],
    max_messages: Optional[int] = None,
    allowed_roles: Optional[List[str]] = None # Optional: Filter roles during formatting
) -> str:
    """
    Formats a list of message dictionaries into a simple 'Role: Content' string.

    Args:
        history_chunk: A list of message dictionaries {'role': str, 'content': str}.
        max_messages: If provided, formats only the last N messages from the chunk.
        allowed_roles: If provided, only messages with roles in this list are included.
                       Defaults to None (include all roles in the chunk).

    Returns:
        A single string with messages formatted and joined by newlines.
        Returns an empty string if the input is empty or filtering removes all messages.
    """
    if not history_chunk:
        return ""

    # Apply role filtering if specified
    if allowed_roles:
        filtered_chunk = [
            msg for msg in history_chunk
            if isinstance(msg, dict) and msg.get("role") in allowed_roles
        ]
    else:
        filtered_chunk = history_chunk # Use the original chunk if no roles specified

    if not filtered_chunk:
        return "" # Return empty if filtering removed everything

    # Apply max_messages slicing
    start_idx = -max_messages if max_messages is not None and max_messages > 0 else 0
    chunk_to_format = filtered_chunk[start_idx:]

    lines = []
    for msg in chunk_to_format:
        # Already filtered by role if allowed_roles was provided
        role = msg.get('role', 'unk').capitalize()
        content = msg.get('content', '').strip()
        if content: # Only include messages with actual content
            lines.append(f"{role}: {content}")

    formatted_string = "\n".join(lines)
    return formatted_string


# --- Retrieval Functions ---
def get_recent_turns(
    messages: List[Dict],
    count: int,
    roles: List[str] = DIALOGUE_ROLES, # Default to dialogue roles
    exclude_last: bool = True
) -> List[Dict]:
    """
    Extracts the most recent 'count' messages matching the specified roles.
    Optionally excludes the very last message in the list.

    Args:
        messages: The list of message dictionaries.
        count: The maximum number of messages to return.
        roles: A list of roles to include (defaults to 'user', 'assistant').
        exclude_last: If True, the very last message in the original list is excluded
                      before selecting the count.

    Returns:
        A list containing the selected recent messages, preserving order.
    """
    if count <= 0 or not messages:
        return []

    source_list = messages[:-1] if exclude_last and len(messages) > 0 else messages
    # Filter messages based on the provided roles
    filtered_messages = [
        msg for msg in source_list if isinstance(msg, dict) and msg.get("role") in roles
    ]

    if not filtered_messages:
        return []

    # Slice the filtered list to get the most recent 'count'
    start_index = max(0, len(filtered_messages) - count)
    recent_history = filtered_messages[start_index:]

    return recent_history


def get_dialogue_history(
    messages: List[Dict],
    roles: List[str] = DIALOGUE_ROLES, # Default to dialogue roles
    exclude_last: bool = True
) -> List[Dict]:
    """
    Extracts all messages matching the specified roles from the list.
    Optionally excludes the very last message.

    Args:
        messages: The list of message dictionaries.
        roles: A list of roles to include (defaults to 'user', 'assistant').
        exclude_last: If True, the very last message in the original list is excluded.

    Returns:
        A list containing all matching messages, preserving order.
    """
    if not messages:
        return []

    source_list = messages[:-1] if exclude_last and len(messages) > 0 else messages
    # Filter messages based on the provided roles
    filtered_messages = [
        msg for msg in source_list if isinstance(msg, dict) and msg.get("role") in roles
    ]

    return filtered_messages

# ==============================================================================
# === Turn-Aware T0 History Selection (FIXED: Ignores System Messages)      ===
# ==============================================================================
def select_turns_for_t0(
    full_history: List[Dict],
    target_tokens: int,
    tokenizer: Any, # Expects .encode() method
    max_overflow_ratio: float = 1.15,
    fallback_turns: int = 10,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES # Roles to consider for T0
) -> List[Dict]:
    """
    Selects recent messages for T0 context from dialogue roles ('user'/'assistant'),
    aiming for a token limit, and prioritizing turn completion within an overflow ratio.
    Ignores system messages and other non-dialogue roles.

    Args:
        full_history: The complete list of message dictionaries (can include system msgs).
        target_tokens: The desired token limit for the T0 slice.
        tokenizer: A tokenizer instance with an .encode() method.
        max_overflow_ratio: Max allowed ratio over target_tokens for turn completion (e.g., 1.15).
        fallback_turns: How many turns to return if tokenizer is unavailable.
        dialogue_only_roles: List of message roles to consider part of the dialogue.

    Returns:
        A list of selected message dictionaries, containing only dialogue roles.
    """
    func_logger = logging.getLogger(__name__ + '.select_turns_for_t0')
    is_debug_enabled = func_logger.isEnabledFor(logging.DEBUG)

    if not full_history:
        func_logger.debug("[T0 Select] Input history is empty, returning empty T0 slice.")
        return []

    # --- Filter for Dialogue Roles FIRST ---
    dialogue_history = [
        msg for msg in full_history
        if isinstance(msg, dict) and msg.get("role") in dialogue_only_roles
    ]
    if not dialogue_history:
        func_logger.debug(f"[T0 Select] Input history contains no messages with roles in {dialogue_only_roles}. Returning empty T0 slice.")
        return []

    if is_debug_enabled:
        func_logger.debug(f"[T0 Select Init] Filtered for dialogue roles ({dialogue_only_roles}). Kept {len(dialogue_history)} out of {len(full_history)} messages.")

    # --- Fallback if no tokenizer (uses filtered dialogue history) ---
    if not tokenizer:
        func_logger.warning(
            f"[T0 Select] Tokenizer unavailable. Falling back to last {fallback_turns} turns from dialogue history."
        )
        start_idx = max(0, len(dialogue_history) - fallback_turns)
        return dialogue_history[start_idx:]

    # --- Log Initial Parameters (after filtering) ---
    if is_debug_enabled:
        func_logger.debug(f"[T0 Select Init] Target tokens: {target_tokens}")
        func_logger.debug(f"[T0 Select Init] Max overflow ratio: {max_overflow_ratio}")
        if dialogue_history:
            last_msg_snippet = str(dialogue_history[-1].get('content', ''))[:50]
            func_logger.debug(f"[T0 Select Init] Dialogue History Last Msg Snippet: '{last_msg_snippet}...'")

    # --- Main selection loop (iterating backwards through DIALOGUE history) ---
    selected_history: List[Dict] = []
    current_tokens: int = 0
    # Keep track of the index *within the dialogue_history list*
    earliest_selected_dialogue_index: int = -1

    if is_debug_enabled: func_logger.debug("[T0 Select Loop] Starting backward iteration through dialogue history...")

    for i in range(len(dialogue_history) - 1, -1, -1):
        msg = dialogue_history[i]
        msg_content = msg.get("content", "")
        msg_role = msg.get("role", "unknown") # Should be user/assistant here

        if not msg_content:
            if is_debug_enabled: func_logger.debug(f"[T0 Select Loop] Skipping dialogue index {i} (role: {msg_role}): Empty content.")
            continue

        msg_tokens = 0
        try:
            encoded_tokens = tokenizer.encode(msg_content)
            msg_tokens = len(encoded_tokens)
            if is_debug_enabled: func_logger.debug(f"[T0 Select Loop] Dialogue index {i} (role: {msg_role}): Calculated tokens = {msg_tokens}")
        except Exception as e:
            func_logger.error(f"[T0 Select Loop] Tokenizer error on dialogue msg index {i} (role: {msg_role}): {e}. Skipping msg.", exc_info=False)
            continue

        # Check token limit BEFORE adding the current message
        if selected_history and (current_tokens + msg_tokens > target_tokens):
            if is_debug_enabled:
                func_logger.debug(
                    f"[T0 Select Loop] Token limit {target_tokens} would be exceeded by adding dialogue index {i} "
                    f"(current: {current_tokens}, msg: {msg_tokens}, total: {current_tokens + msg_tokens}). "
                    f"Stopping selection BEFORE adding index {i}."
                )
            break # Stop BEFORE adding this message

        # Prepend the message to maintain order and update state
        selected_history.insert(0, msg)
        current_tokens += msg_tokens
        earliest_selected_dialogue_index = i
        if is_debug_enabled: func_logger.debug(f"[T0 Select Loop] Added dialogue index {i}. New total tokens: {current_tokens}. Current slice length: {len(selected_history)}")

    # --- Post-Loop Logging & Checks ---
    if not selected_history:
         func_logger.warning("[T0 Select PostLoop] Initial selection resulted in an empty slice (from dialogue history).")
         return []

    if is_debug_enabled:
        func_logger.debug(f"[T0 Select PostLoop] Initial selection finished.")
        func_logger.debug(f"[T0 Select PostLoop] Selected {len(selected_history)} messages.")
        func_logger.debug(f"[T0 Select PostLoop] Current tokens: {current_tokens}")
        func_logger.debug(f"[T0 Select PostLoop] Earliest selected index in dialogue history: {earliest_selected_dialogue_index}")
        if selected_history:
            earliest_msg_role = selected_history[0].get('role', 'unknown')
            earliest_msg_snippet = str(selected_history[0].get('content', ''))[:100]
            func_logger.debug(f"[T0 Select PostLoop] Earliest selected message (role: {earliest_msg_role}): '{earliest_msg_snippet}...'")

    # --- Post-loop: Check for Turn Completion (within dialogue history) ---
    first_selected_msg = selected_history[0]
    first_selected_role = first_selected_msg.get("role") # Should be user/assistant

    # Only check if the earliest is 'assistant' and it's not the very first dialogue message
    if first_selected_role == "assistant" and earliest_selected_dialogue_index > 0:
        if is_debug_enabled: func_logger.debug("[T0 Turn Check] Earliest selected msg is 'assistant', checking preceding dialogue turn.")
        # Get the preceding message *in the filtered dialogue_history list*
        preceding_dialogue_index = earliest_selected_dialogue_index - 1
        preceding_msg = dialogue_history[preceding_dialogue_index]
        preceding_role = preceding_msg.get("role") # Should be 'user' for a valid turn pair

        if preceding_role == "user":
            if is_debug_enabled: func_logger.debug(f"[T0 Turn Check] Found preceding 'user' message at dialogue index {preceding_dialogue_index}.")
            preceding_content = preceding_msg.get("content", "")
            preceding_tokens = 0
            if preceding_content:
                try:
                    preceding_tokens = len(tokenizer.encode(preceding_content))
                except Exception as e:
                    func_logger.error(f"[T0 Turn Check] Tokenizer error on preceding user msg dialogue index {preceding_dialogue_index}: {e}", exc_info=False)
                    preceding_tokens = -1 # Indicate error

            if preceding_tokens >= 0:
                # === START MODIFICATION ===
                # Explicitly recalculate overflow limit here for clarity and certainty
                calculated_overflow_limit = target_tokens * max_overflow_ratio
                projected_tokens = current_tokens + preceding_tokens

                if is_debug_enabled:
                    func_logger.debug(
                        f"[T0 Turn Check] Preceding user tokens: {preceding_tokens}. "
                        f"Current slice tokens: {current_tokens}. "
                        f"Projected total: {projected_tokens}. "
                        f"Target: {target_tokens}. Ratio: {max_overflow_ratio:.2f}. "
                        f"Calculated Overflow limit: {calculated_overflow_limit:.0f}" # Use calculated limit in log
                    )

                # Check if adding the preceding user turn fits within the explicitly calculated overflow budget
                if projected_tokens <= calculated_overflow_limit:
                # === END MODIFICATION ===
                    func_logger.info( # Changed to INFO to ensure visibility if added
                        f"[T0 Turn Check] Including preceding 'user' turn (dialogue index {preceding_dialogue_index}) "
                        f"as projected tokens ({projected_tokens}) <= overflow limit ({calculated_overflow_limit:.0f})."
                    )
                    selected_history.insert(0, preceding_msg) # Prepend the user message
                    current_tokens = projected_tokens # Update token count
                else:
                    if is_debug_enabled: func_logger.debug( # Keep this DEBUG unless problematic
                        f"[T0 Turn Check] Preceding 'user' turn excluded as projected tokens ({projected_tokens}) "
                        f"> overflow limit ({calculated_overflow_limit:.0f})."
                        )
            else:
                if is_debug_enabled: func_logger.debug("[T0 Turn Check] Preceding 'user' turn tokenization failed, cannot check overflow.")
        # This case should ideally not happen if history alternates user/assistant
        elif preceding_role == "assistant":
             if is_debug_enabled: func_logger.debug(f"[T0 Turn Check] Preceding message (dialogue index {preceding_dialogue_index}) is also 'assistant'. Turn completion check skipped.")
        else: # Should not happen with DIALOGUE_ROLES filter
             if is_debug_enabled: func_logger.debug(f"[T0 Turn Check] Preceding message (dialogue index {preceding_dialogue_index}) has unexpected role: {preceding_role}.")

    elif first_selected_role == "user":
        if is_debug_enabled: func_logger.debug(f"[T0 Turn Check] Earliest selected msg is 'user', no preceding turn check needed.")
    else: # earliest_selected_dialogue_index must be 0 if role is assistant
         if is_debug_enabled: func_logger.debug("[T0 Turn Check] Earliest selected dialogue msg is the first dialogue message, no preceding turn check needed.")


    # --- Final Logging and Return ---
    final_count = len(selected_history)
    func_logger.info(f"Final T0 slice selected: {final_count} dialogue messages, approx {current_tokens} tokens.")
    if is_debug_enabled and not selected_history:
         func_logger.debug("[T0 Select Return] Returning EMPTY list.")
    elif is_debug_enabled:
         final_roles = [msg.get("role", "unk") for msg in selected_history]
         func_logger.debug(f"[T0 Select Return] Final slice roles: {final_roles}")
         # Optionally log full content if needed
         # func_logger.debug(f"[T0 Select Return] Final slice content:\n{json.dumps(selected_history, indent=2)}")

    # Return the list containing only selected dialogue messages
    return selected_history