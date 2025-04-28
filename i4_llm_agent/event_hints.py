# === START OF FILE i4_llm_agent/event_hints.py ===
# i4_llm_agent/event_hints.py

import logging
import asyncio
from typing import (
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

# --- Placeholders for Prompt Formatting (Includes Time) ---
EVENT_HINT_HISTORY_PLACEHOLDER = "{recent_history}"
EVENT_HINT_CONTEXT_PLACEHOLDER = "{background_context}"
EVENT_HINT_SEASON_PLACEHOLDER = "{current_season}"
EVENT_HINT_WEATHER_PLACEHOLDER = "{current_weather}"
EVENT_HINT_TIME_PLACEHOLDER = "{current_time_of_day}" # <<< Check: Present


# === MODIFIED PROMPT TEMPLATE (v2 - Emphasize World State Hints) ===
DEFAULT_EVENT_HINT_TEMPLATE_TEXT = f"""
[[SYSTEM ROLE: Contextual Environmental Detail Suggestor]]

**Objective:** Analyze the dialogue, background context, and **current world state** (Season, Weather, Time) to suggest ONE brief, plausible, minor environmental detail or sensory event that **reflects and reinforces** the provided world state. Avoid major plot changes or character commands. Output ONLY the suggestion text or "[No Suggestion]".

**Established World State (Use this as factual basis):**
*   **Season:** {EVENT_HINT_SEASON_PLACEHOLDER}
*   **Weather:** {EVENT_HINT_WEATHER_PLACEHOLDER}
*   **Time of Day:** {EVENT_HINT_TIME_PLACEHOLDER}

**RECENT DIALOGUE HISTORY (Consider the flow):**
---
{EVENT_HINT_HISTORY_PLACEHOLDER}
---

**BACKGROUND CONTEXT (Character info, location, etc.):**
---
{EVENT_HINT_CONTEXT_PLACEHOLDER}
---

**Instructions:**

1.  **Ground in World State:** Your primary goal is to generate a suggestion strongly tied to the **established Season, Weather, and Time of Day**.
2.  **Focus on Environment/Senses:** Keep it brief and focused on sensory details or minor environmental occurrences.
3.  **Examples Based on World State:**
    *   If **Weather=Cloudy, Time=Night:** "The thick clouds overhead block out any moonlight."
    *   If **Season=Spring, Weather=Clear, Time=Morning:** "Dewdrops glitter on the new spring leaves in the morning sun."
    *   If **Season=Autumn, Weather=Windy, Time=Afternoon:** "A gust of wind sends dry autumn leaves skittering across the path."
    *   If **Weather=Rainy:** "The steady drumming of rain on the roof is clearly audible."
    *   If **Time=Evening:** "The shadows lengthen as evening approaches."
    *   If **Weather=Cool:** "A cool breeze stirs the nearby branches."
4.  **Avoid:** Do NOT suggest character actions, dialogue, major plot points, or significant weather *changes* (just reflect the *current* state).
5.  **Output:** If a fitting detail comes to mind based on the world state, output only the suggestion text. If no fitting detail seems appropriate, output: `[No Suggestion]`

**EVENT/DETAIL SUGGESTION:**
"""
# === END MODIFIED PROMPT TEMPLATE ===

EVENT_HANDLING_GUIDELINE_TEXT = """
--- [ EVENT HANDLING GUIDELINE ] ---
You may occasionally receive an [[Event Suggestion: ...]] within the user's input turn. This is a suggestion for a minor environmental detail or event. You are encouraged, but **not required**, to subtly weave this suggestion into your response if it feels natural and appropriate for the current scene and your character's focus. Do not treat it as a direct command or derail the main conversation unless it provides a compelling opportunity. If you incorporate it, do so naturally within the narrative. If you choose to ignore it, simply proceed with responding to the user's main query.
--- [ END EVENT HANDLING GUIDELINE ] ---
"""


# === Helper Functions ===

# --- Helper: Includes time_of_day ---
def _format_event_hint_prompt(
    recent_history_str: str,
    background_context: str,
    current_season: Optional[str],
    current_weather: Optional[str],
    current_time_of_day: Optional[str], # <<< Check: Parameter present
    template: str
) -> str:
    """
    Formats the prompt for the Event Hint LLM, including world state (season, weather, time).

    Args:
        recent_history_str: Formatted recent dialogue history.
        background_context: Background context string.
        current_season: The current season string (e.g., "Summer").
        current_weather: The current weather string (e.g., "Clear").
        current_time_of_day: The current time of day string (e.g., "Morning").
        template: The prompt template string.

    Returns:
        The formatted prompt string, or an error string if formatting fails.
    """
    func_logger = logging.getLogger(__name__ + '._format_event_hint_prompt')
    if not template or not isinstance(template, str):
        return "[Error: Invalid Template for Event Hint]"

    # Use placeholders or defaults if state is None/empty
    season_text = current_season if current_season else "Not Specified"
    weather_text = current_weather if current_weather else "Not Specified"
    time_text = current_time_of_day if current_time_of_day else "Not Specified" # <<< Uses param

    # Basic replace for safety
    safe_history = recent_history_str.replace("{", "{{").replace("}", "}}") if isinstance(recent_history_str, str) else ""
    safe_context = background_context.replace("{", "{{").replace("}", "}}") if isinstance(background_context, str) else ""
    safe_season = season_text.replace("{", "{{").replace("}", "}}")
    safe_weather = weather_text.replace("{", "{{").replace("}", "}}")
    safe_time = time_text.replace("{", "{{").replace("}", "}}") # <<< Uses param

    try:
        # Create the dictionary of placeholders to format
        format_dict = {
            EVENT_HINT_HISTORY_PLACEHOLDER.strip('{}'): safe_history,
            EVENT_HINT_CONTEXT_PLACEHOLDER.strip('{}'): safe_context,
            EVENT_HINT_SEASON_PLACEHOLDER.strip('{}'): safe_season,
            EVENT_HINT_WEATHER_PLACEHOLDER.strip('{}'): safe_weather,
            EVENT_HINT_TIME_PLACEHOLDER.strip('{}'): safe_time, # <<< Uses placeholder
        }
        # Use .format() here as the template itself contains placeholders
        formatted_prompt = template.format(**format_dict)
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
    # Remove potential leading/trailing markers or whitespace before formatting
    clean_hint = event_hint.strip()
    if clean_hint.lower() == '[no suggestion]':
        return "" # Don't format the "no suggestion" marker itself
    return f"[[Event Suggestion: {clean_hint}]]"


# === Core Logic ===

# --- Core Function: Accepts time_of_day ---
async def generate_event_hint(
    config: Any, # Expects object with event hint LLM config attributes
    history_messages: List[Dict],
    background_context: str,
    current_season: Optional[str],
    current_weather: Optional[str],
    current_time_of_day: Optional[str], # <<< Check: Parameter present
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    logger_instance: Optional[logging.Logger] = None,
    session_id: str = "unknown_session",
) -> Optional[str]:
    """
    Generates a dynamic event hint using a secondary LLM based on context and world state.

    Args:
        config: Configuration object containing valves like 'event_hint_llm_api_url', etc.
        history_messages: List of message dictionaries for history context.
        background_context: Combined background context string.
        current_season: The current season string.
        current_weather: The current weather string.
        current_time_of_day: The current time of day string.
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
    # Use the template defined in this file by default (which is now the updated one)
    hint_llm_template = getattr(config, 'event_hint_llm_prompt_template', DEFAULT_EVENT_HINT_TEMPLATE_TEXT)
    hint_history_count = getattr(config, 'event_hint_history_count', 6)

    if not hint_llm_url or not hint_llm_key:
        func_logger.debug(f"[{caller_info}] Event Hint LLM URL or Key missing. Skipping hint generation.")
        return None
    # Use the updated default template if the one from config is missing/invalid
    if not hint_llm_template or not isinstance(hint_llm_template, str):
        func_logger.warning(f"[{caller_info}] Event Hint LLM prompt template invalid or missing in config. Using updated default.")
        hint_llm_template = DEFAULT_EVENT_HINT_TEMPLATE_TEXT
        if not hint_llm_template or not isinstance(hint_llm_template, str): # Check default again
             func_logger.error(f"[{caller_info}] Default event hint template also invalid. Skipping.")
             return None

    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        func_logger.error(f"[{caller_info}] Invalid llm_call_func provided. Skipping.")
        return None

    # --- Prepare Inputs ---
    # Ensure history_messages is a list before processing
    if not isinstance(history_messages, list):
        func_logger.warning(f"[{caller_info}] history_messages is not a list ({type(history_messages)}). Using empty history for hint.")
        history_messages = []

    recent_history_list = get_recent_turns(
        history_messages,
        hint_history_count,
        DIALOGUE_ROLES,
        exclude_last=False # Include latest messages for hint context
    )
    recent_history_str = format_history_for_llm(recent_history_list) if recent_history_list else "[No Recent History]"

    # Format the prompt using the helper that now includes world state
    event_hint_prompt_text = _format_event_hint_prompt(
        recent_history_str=recent_history_str,
        background_context=background_context,
        current_season=current_season,         # Pass world state
        current_weather=current_weather,       # Pass world state
        current_time_of_day=current_time_of_day, # Pass world state <<< Uses param
        template=hint_llm_template # Use the potentially updated template
    )

    if not event_hint_prompt_text or event_hint_prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed to format event hint prompt: {event_hint_prompt_text}. Skipping.")
        return None

    event_hint_payload = {"contents": [{"parts": [{"text": event_hint_prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling Event Hint LLM (with updated prompt emphasizing world state)...")
    # Optional: Log the actual prompt being sent for debugging
    # func_logger.debug(f"[{caller_info}] Event Hint Prompt Text:\n------\n{event_hint_prompt_text}\n------")

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