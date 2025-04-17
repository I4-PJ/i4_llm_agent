# i4_llm_agent/history.py
import logging
from typing import List, Dict, Optional, Any

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

    logger.debug(f"Formatting {len(chunk_to_format)} messages (Max: {max_messages}).")

    for msg in chunk_to_format:
        role = msg.get('role', 'unk').capitalize() # Default to 'Unk' if role missing
        content = msg.get('content', '').strip()   # Default to empty string if content missing
        if not content: # Optionally skip empty messages
            # logger.debug(f"Skipping empty message from role: {role}") # Can be noisy
            continue
        lines.append(f"{role}: {content}")

    formatted_string = "\n".join(lines)
    # logger.debug(f"Formatted history string length: {len(formatted_string)}") # Also potentially noisy
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
    logger.debug(f"Extracting up to {count} recent turns (roles: {roles}, exclude_last: {exclude_last})...")
    if count <= 0 or not messages:
        return []

    # Determine the effective list to filter from
    source_list = messages[:-1] if exclude_last and len(messages) > 0 else messages

    # Filter messages by role from the source list
    filtered_messages = [
        msg for msg in source_list if isinstance(msg, dict) and msg.get("role") in roles
    ]

    if not filtered_messages:
        return []

    # Get the last 'count' messages from the filtered list
    start_index = max(0, len(filtered_messages) - count)
    recent_history = filtered_messages[start_index:]

    logger.debug(f"Extracted {len(recent_history)} recent turns.")
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
    logger.debug(f"Extracting dialogue history (roles: {roles}, exclude_last: {exclude_last})...")
    if not messages:
        return []

    # Determine the effective list to filter from
    source_list = messages[:-1] if exclude_last and len(messages) > 0 else messages

    # Filter messages by role
    filtered_messages = [
        msg for msg in source_list if isinstance(msg, dict) and msg.get("role") in roles
    ]

    logger.debug(f"Extracted {len(filtered_messages)} dialogue history messages.")
    return filtered_messages




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
    assistant turn if it fits within a defined overflow ratio.

    Args:
        full_history: The list of message dictionaries (should exclude the
                      absolute latest user query being processed by the pipe).
        target_tokens: The desired approximate token count for the T0 slice.
        tokenizer: A tokenizer instance with an .encode() method.
        max_overflow_ratio: Maximum allowed token overflow factor
                           (e.g., 1.15 allows up to 15% overflow to complete a turn).
        fallback_turns: How many turns to take if tokenizer is unavailable.

    Returns:
        A list of selected message dictionaries for the T0 slice,
        preserving original order.
    """
    func_logger = logging.getLogger(__name__ + '.select_turns_for_t0')

    if not full_history:
        func_logger.debug("Input history is empty, returning empty T0 slice.")
        return []

    # --- Fallback if no tokenizer ---
    if not tokenizer:
        func_logger.warning(
            f"Tokenizer unavailable for T0 selection. Falling back to last {fallback_turns} turns."
        )
        start_idx = max(0, len(full_history) - fallback_turns)
        return full_history[start_idx:]

    # --- Main selection loop (iterating backwards) ---
    selected_history: List[Dict] = []
    current_tokens: int = 0
    earliest_selected_index: int = -1 # Track index in full_history

    func_logger.debug(f"Selecting T0 turns from {len(full_history)} messages, target tokens: {target_tokens}")

    for i in range(len(full_history) - 1, -1, -1):
        msg = full_history[i]
        msg_content = msg.get("content", "")
        if not msg_content: # Skip messages with no content
            continue

        msg_tokens = 0
        try:
            msg_tokens = len(tokenizer.encode(msg_content))
        except Exception as e:
            func_logger.error(f"Tokenizer error on msg index {i}: {e}. Skipping msg.")
            continue # Skip this message if tokenization fails

        # Check if adding this message would exceed the target
        if current_tokens + msg_tokens > target_tokens and selected_history:
            # We've hit the token limit *before* adding this message.
            # The loop stops here, selected_history contains messages up to index i+1
            func_logger.debug(
                f"Token limit {target_tokens} reached before adding msg index {i} "
                f"(current: {current_tokens}, msg: {msg_tokens}). Stopping initial selection."
            )
            break

        # Prepend the message to maintain order and update state
        selected_history.insert(0, msg)
        current_tokens += msg_tokens
        earliest_selected_index = i
    else:
        # Loop completed without breaking (all history fit or history was small)
        func_logger.debug("Selected all messages as they fit within the token limit.")
        earliest_selected_index = 0 # All messages selected

    if not selected_history:
         func_logger.warning("T0 selection resulted in an empty slice after initial token check.")
         return []

    func_logger.debug(f"Initial T0 selection: {len(selected_history)} msgs, {current_tokens} tokens. Earliest index: {earliest_selected_index}")

    # --- Post-loop: Check for Turn Completion ---
    first_selected_msg = selected_history[0]
    first_selected_role = first_selected_msg.get("role")

    # Only check if the earliest message selected is from the assistant
    # and if it's not the very first message in the entire history
    if first_selected_role == "assistant" and earliest_selected_index > 0:
        func_logger.debug("Earliest selected T0 msg is 'assistant', checking preceding 'user' turn.")
        preceding_index = earliest_selected_index - 1
        preceding_msg = full_history[preceding_index]
        preceding_role = preceding_msg.get("role")

        if preceding_role == "user":
            func_logger.debug(f"Found preceding 'user' message at index {preceding_index}.")
            preceding_content = preceding_msg.get("content", "")
            preceding_tokens = 0
            if preceding_content:
                try:
                    preceding_tokens = len(tokenizer.encode(preceding_content))
                except Exception as e:
                    func_logger.error(f"Tokenizer error on preceding user msg index {preceding_index}: {e}")
                    preceding_tokens = -1 # Indicate error

            if preceding_tokens >= 0: # If tokenization didn't fail
                overflow_limit = target_tokens * max_overflow_ratio
                projected_tokens = current_tokens + preceding_tokens

                func_logger.debug(
                    f"Preceding user tokens: {preceding_tokens}. "
                    f"Projected total: {projected_tokens}. Overflow limit: {overflow_limit:.0f}"
                )

                if projected_tokens <= overflow_limit:
                    func_logger.info(
                        f"Including preceding 'user' turn (index {preceding_index}) as it fits within overflow limit."
                    )
                    selected_history.insert(0, preceding_msg)
                    current_tokens = projected_tokens # Update token count
                else:
                    func_logger.debug(
                        "Preceding 'user' turn excluded as it exceeds overflow limit."
                    )
            # else: tokenizer error already logged
        else:
            func_logger.debug(f"Preceding message (index {preceding_index}) is not 'user' ({preceding_role}). No turn completion needed.")
    elif first_selected_role != "assistant":
        func_logger.debug("Earliest selected T0 msg is not 'assistant', no preceding turn check needed.")
    else: # earliest_selected_index must be 0
         func_logger.debug("Earliest selected T0 msg is the first message in history, no preceding turn check needed.")


    final_count = len(selected_history)
    func_logger.info(f"Final T0 slice: {final_count} messages, approx {current_tokens} tokens.")
    return selected_history