# i4_llm_agent/history.py
import logging
from typing import List, Dict, Optional

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
