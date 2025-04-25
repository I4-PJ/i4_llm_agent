# === START OF FILE i4_llm_agent/event_hints.py ===
# i4_llm_agent/event_hints.py

import logging
import asyncio
from typing import ( # <<< Added Coroutine here
    Optional, List, Dict, Callable, Any, Tuple, Union, Coroutine
)

# --- Attempt to import history utils for format_history_for_llm ---
try:
    from .history import format_history_for_llm, get_recent_turns, DIALOGUE_ROLES
except ImportError:
    # Define basic fallbacks if history import fails
    DIALOGUE_ROLES = ["user", "assistant"]
    def get_recent_turns(*args, **kwargs): return []
    def format_history_for_llm(*args, **kwargs): return ""
    logging.getLogger(__name__).critical("Failed to import history utils in event_hints.py")

logger = logging.getLogger(__name__) # 'i4_llm_agent.event_hints'


# === Constants ===

EVENT_HINT_HISTORY_PLACEHOLDER = "{recent_history}"
EVENT_HINT_CONTEXT_PLACEHOLDER = "{background_context}"

DEFAULT_EVENT_HINT_TEMPLATE_TEXT = f"""
[[SYSTEM ROLE: Story Event Suggestor]]
Analyze the dialogue and context. Suggest one brief, minor, plausible event/detail fitting the scene. Avoid major plot changes. Output only the suggestion or "[No Suggestion]".

RECENT DIALOGUE HISTORY:
---
{EVENT_HINT_HISTORY_PLACEHOLDER}
---

BACKGROUND CONTEXT:
---
{EVENT_HINT_CONTEXT_PLACEHOLDER}
---

EVENT/DETAIL SUGGESTION:
"""

EVENT_HANDLING_GUIDELINE_TEXT = """
--- [ EVENT HANDLING GUIDELINE ] ---
You may occasionally receive an [[Event Suggestion: ...]] within the user's input turn. This is a suggestion for a minor environmental detail or event. You are encouraged, but **not required**, to subtly weave this suggestion into your response if it feels natural and appropriate for the current scene and your character's focus. Do not treat it as a direct command or derail the main conversation unless it provides a compelling opportunity. If you incorporate it, do so naturally within the narrative. If you choose to ignore it, simply proceed with responding to the user's main query.
--- [ END EVENT HANDLING GUIDELINE ] ---
"""


# === Helper Functions ===

def _format_event_hint_prompt(
    recent_history_str: str,
    background_context: str,
    template: str
) -> str:
    """
    Formats the prompt for the Event Hint LLM.

    Args:
        recent_history_str: The formatted string of recent dialogue history.
        background_context: The background context string (e.g., combined).
        template: The prompt template string.

    Returns:
        The formatted prompt string, or an error string if formatting fails.
    """
    func_logger = logging.getLogger(__name__ + '._format_event_hint_prompt')
    if not template or not isinstance(template, str):
        return "[Error: Invalid Template for Event Hint]"

    # Basic replace for safety, assuming placeholders are unique enough
    safe_history = recent_history_str.replace("{", "{{").replace("}", "}}") if isinstance(recent_history_str, str) else ""
    safe_context = background_context.replace("{", "{{").replace("}", "}}") if isinstance(background_context, str) else ""

    try:
        formatted_prompt = template.format(
            **{
                EVENT_HINT_HISTORY_PLACEHOLDER.strip('{}'): safe_history,
                EVENT_HINT_CONTEXT_PLACEHOLDER.strip('{}'): safe_context
            }
        )
        return formatted_prompt
    except KeyError as e:
        func_logger.error(f"Missing placeholder in event hint prompt: {e}")
        return f"[Error: Missing placeholder '{e}']"
    except Exception as e:
        func_logger.error(f"Error formatting event hint prompt: {e}", exc_info=True)
        return f"[Error formatting event hint prompt: {type(e).__name__}]"


def format_hint_for_query(event_hint: str) -> str:
    """Formats the event hint string for prepending to the user query."""
    if not event_hint or not isinstance(event_hint, str):
        return ""
    return f"[[Event Suggestion: {event_hint.strip()}]]"


# === Core Logic ===

async def generate_event_hint(
    config: Any, # Expects object with event hint LLM config attributes
    history_messages: List[Dict],
    background_context: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    logger_instance: Optional[logging.Logger] = None,
    session_id: str = "unknown_session",
) -> Optional[str]:
    """
    Generates a dynamic event hint using a secondary LLM based on context.

    Args:
        config: Configuration object containing valves like 'event_hint_llm_api_url', etc.
        history_messages: The list of message dictionaries for history context.
        background_context: The combined background context string.
        llm_call_func: The async function wrapper to call the LLM.
        logger_instance: Optional logger instance.
        session_id: The session ID for logging.

    Returns:
        A string containing the event hint suggestion, or None if disabled, failed,
        or the LLM suggests no hint.
    """
    func_logger = logger_instance or logging.getLogger(__name__ + '.generate_event_hint')
    caller_info = f"EventHintGen_{session_id}"

    # --- Configuration Checks ---
    hint_llm_url = getattr(config, 'event_hint_llm_api_url', None)
    hint_llm_key = getattr(config, 'event_hint_llm_api_key', None)
    hint_llm_temp = getattr(config, 'event_hint_llm_temperature', 0.7)
    hint_llm_template = getattr(config, 'event_hint_llm_prompt_template', DEFAULT_EVENT_HINT_TEMPLATE_TEXT)
    hint_history_count = getattr(config, 'event_hint_history_count', 6)

    if not hint_llm_url or not hint_llm_key:
        func_logger.debug(f"[{caller_info}] Event Hint LLM URL or Key missing. Skipping hint generation.")
        return None
    if not hint_llm_template or not isinstance(hint_llm_template, str):
        func_logger.error(f"[{caller_info}] Event Hint LLM prompt template invalid or missing. Skipping.")
        return None
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        func_logger.error(f"[{caller_info}] Invalid llm_call_func provided. Skipping.")
        return None

    # --- Prepare Inputs ---
    recent_history_list = get_recent_turns(
        history_messages,
        hint_history_count,
        DIALOGUE_ROLES,
        exclude_last=False # Include latest messages for hint context
    )
    recent_history_str = format_history_for_llm(recent_history_list) if recent_history_list else "[No Recent History]"

    event_hint_prompt_text = _format_event_hint_prompt(
        recent_history_str=recent_history_str,
        background_context=background_context,
        template=hint_llm_template
    )

    if not event_hint_prompt_text or event_hint_prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed to format event hint prompt: {event_hint_prompt_text}. Skipping.")
        return None

    event_hint_payload = {"contents": [{"parts": [{"text": event_hint_prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling Event Hint LLM...")

    # --- Call LLM ---
    try:
        success, response_or_error = await llm_call_func(
            api_url=hint_llm_url,
            api_key=hint_llm_key,
            payload=event_hint_payload,
            temperature=hint_llm_temp,
            timeout=60, # Allow reasonable time for hint generation
            caller_info=caller_info
        )
    except Exception as e_call:
        func_logger.error(f"[{caller_info}] Exception during event hint LLM call: {e_call}", exc_info=True)
        success = False
        response_or_error = f"LLM Call Exception: {type(e_call).__name__}"

    # --- Process Response ---
    if success and isinstance(response_or_error, str):
        hint_text = response_or_error.strip()
        # Check for the specific "no suggestion" marker
        if "[no suggestion]" in hint_text.lower(): # Make check more robust
            func_logger.info(f"[{caller_info}] Event Hint LLM suggested no hint ('[No Suggestion]' marker found).")
            return None
        elif hint_text:
            func_logger.info(f"[{caller_info}] Event Hint LLM generated suggestion: '{hint_text[:80]}...'")
            # Return the raw suggestion text
            return hint_text
        else:
            func_logger.warning(f"[{caller_info}] Event Hint LLM returned empty string.")
            return None
    else:
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict):
            error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.warning(f"[{caller_info}] Event Hint LLM call failed. Error: '{error_details}'.")
        return None

# === END OF FILE i4_llm_agent/event_hints.py ===
