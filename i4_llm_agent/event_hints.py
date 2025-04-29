# === START OF FILE i4_llm_agent/event_hints.py ===
# i4_llm_agent/event_hints.py

import logging
import asyncio
import json # Added for parsing weather proposal
import re # Added for parsing response lines
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
EVENT_HINT_TIME_PLACEHOLDER = "{current_time_of_day}"

# === MODIFIED PROMPT TEMPLATE (v3 - Dual Output: Hint + Weather Proposal) ===
DEFAULT_EVENT_HINT_TEMPLATE_TEXT = f"""
[[SYSTEM ROLE: Environmental Detail & Weather Suggestor]]

**Objective:** Perform two tasks based on the current world state and context:
1.  Suggest ONE brief, plausible, minor environmental detail/sensory event reflecting the CURRENT world state (Season, Weather, Time).
2.  Propose the MOST LIKELY weather for the *next* logical time step, considering the current state.

**Established World State (Use this as factual basis):**
*   Season: {EVENT_HINT_SEASON_PLACEHOLDER}
*   Weather: {EVENT_HINT_WEATHER_PLACEHOLDER}
*   Time of Day: {EVENT_HINT_TIME_PLACEHOLDER}

**RECENT DIALOGUE HISTORY (Consider the flow):**
---
{EVENT_HINT_HISTORY_PLACEHOLDER}
---

**BACKGROUND CONTEXT (Character info, location, etc.):**
---
{EVENT_HINT_CONTEXT_PLACEHOLDER}
---

**Instructions:**

**Part 1: Environmental Hint**
*   Generate a brief sensory detail strongly tied to the **established Season, Weather, and Time of Day**.
*   Focus on environment/senses. Avoid major plot changes or character commands.
*   Examples based on World State:
    *   If Weather=Cloudy, Time=Night: "The thick clouds overhead block out any moonlight."
    *   If Season=Spring, Weather=Clear, Time=Morning: "Dewdrops glitter on the new spring leaves."
    *   If Weather=Rainy: "The steady drumming of rain on the roof is clearly audible."
*   Output Format: Start the line with `Hint: `. If no suitable hint comes to mind, output `Hint: [No Suggestion]`.

**Part 2: Weather Proposal**
*   Consider the current Season, Weather, and Time of Day. What is the most plausible weather for the *next* few hours or the next time block (e.g., if Morning, consider Afternoon)?
*   Prioritize common, realistic transitions (e.g., Clear -> Cloudy, Cloudy -> Rainy, Rainy -> Clearing, Clear -> Foggy (night/morning)). Avoid drastic, unexplained changes unless the season strongly suggests it (e.g., sudden summer thunderstorm).
*   Output Format: Start the line with `WeatherProposal: `. Provide a JSON object containing the current weather and the proposed next weather. Use standard terms (Clear, Cloudy, Rainy, Stormy, Windy, Foggy, Snowy, Cool, Warm, Hot, Cold). If no change is likely, the proposed weather should be the same as the current weather.
    *   Example: `WeatherProposal: {{{{ "previous_weather": "Clear", "new_weather": "Cloudy" }}}}` # Escaped braces for format
    *   Example (No Change): `WeatherProposal: {{{{ "previous_weather": "Rainy", "new_weather": "Rainy" }}}}` # Escaped braces for format

**Combined Output:**
Provide BOTH the `Hint:` line AND the `WeatherProposal:` line, each on its own line. The order matters: Hint first, then WeatherProposal.

**OUTPUT:**
Hint: <Your generated hint text or [No Suggestion]>
WeatherProposal: {{{{ "previous_weather": "<current_weather>", "new_weather": "<proposed_weather>" }}}}
"""
# === END MODIFIED PROMPT TEMPLATE ===

# Existing guideline text (unchanged)
EVENT_HANDLING_GUIDELINE_TEXT = """
--- [ EVENT HANDLING GUIDELINE ] ---
You may occasionally receive an [[Event Suggestion: ...]] within the user's input turn. This is a suggestion for a minor environmental detail or event. You are encouraged, but **not required**, to subtly weave this suggestion into your response if it feels natural and appropriate for the current scene and your character's focus. Do not treat it as a direct command or derail the main conversation unless it provides a compelling opportunity. If you incorporate it, do so naturally within the narrative. If you choose to ignore it, simply proceed with responding to the user's main query.
--- [ END EVENT HANDLING GUIDELINE ] ---
"""

# === Helper Functions ===

# Helper: _format_event_hint_prompt (unchanged, uses the template passed to it)
def _format_event_hint_prompt(
    recent_history_str: str,
    background_context: str,
    current_season: Optional[str],
    current_weather: Optional[str],
    current_time_of_day: Optional[str],
    template: str
) -> str:
    """
    Formats the prompt for the Event Hint LLM, including world state (season, weather, time).
    (Remains unchanged, relies on the template string provided)
    """
    func_logger = logging.getLogger(__name__ + '._format_event_hint_prompt')
    if not template or not isinstance(template, str):
        return "[Error: Invalid Template for Event Hint]"

    season_text = current_season if current_season else "Not Specified"
    weather_text = current_weather if current_weather else "Not Specified"
    time_text = current_time_of_day if current_time_of_day else "Not Specified"

    safe_history = recent_history_str.replace("{", "{{").replace("}", "}}") if isinstance(recent_history_str, str) else ""
    safe_context = background_context.replace("{", "{{").replace("}", "}}") if isinstance(background_context, str) else ""
    safe_season = season_text.replace("{", "{{").replace("}", "}}")
    safe_weather = weather_text.replace("{", "{{").replace("}", "}}")
    safe_time = time_text.replace("{", "{{").replace("}", "}}")

    try:
        format_dict = {
            EVENT_HINT_HISTORY_PLACEHOLDER.strip('{}'): safe_history,
            EVENT_HINT_CONTEXT_PLACEHOLDER.strip('{}'): safe_context,
            EVENT_HINT_SEASON_PLACEHOLDER.strip('{}'): safe_season,
            EVENT_HINT_WEATHER_PLACEHOLDER.strip('{}'): safe_weather,
            EVENT_HINT_TIME_PLACEHOLDER.strip('{}'): safe_time,
        }
        formatted_prompt = template.format(**format_dict)
        return formatted_prompt
    except KeyError as e:
        func_logger.error(f"Missing placeholder in event hint prompt: {e}")
        return f"[Error: Missing placeholder '{e}']"
    except Exception as e:
        func_logger.error(f"Error formatting event hint prompt: {e}", exc_info=True)
        return f"[Error formatting event hint prompt: {type(e).__name__}]"

# Helper: format_hint_for_query (unchanged)
def format_hint_for_query(event_hint: str) -> str:
    """Formats the event hint string for prepending to the user query."""
    if not event_hint or not isinstance(event_hint, str):
        return ""
    clean_hint = event_hint.strip()
    if clean_hint.lower() == '[no suggestion]':
        return ""
    return f"[[Event Suggestion: {clean_hint}]]"


# === Core Logic ===

# --- MODIFIED Core Function: Returns hint text AND weather proposal dict ---
async def generate_event_hint(
    config: Any,
    history_messages: List[Dict],
    background_context: str,
    current_season: Optional[str],
    current_weather: Optional[str],
    current_time_of_day: Optional[str],
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    logger_instance: Optional[logging.Logger] = None,
    session_id: str = "unknown_session",
) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """
    Generates a dynamic event hint AND proposes a weather change using a
    secondary LLM based on context and world state.

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
        A tuple containing:
        - Optional[str]: The event hint suggestion text (or None if none generated/error).
        - Dict[str, Optional[str]]: The weather proposal dictionary
          (e.g., {"previous_weather": "...", "new_weather": "..."}).
          Returns a default dict with current weather if parsing fails.
    """
    func_logger = logger_instance or logging.getLogger(__name__ + '.generate_event_hint')
    caller_info = f"EventHintGen_{session_id}"

    # --- Default return values ---
    default_weather_proposal = {
        "previous_weather": current_weather if current_weather else "Unknown",
        "new_weather": current_weather if current_weather else "Unknown"
    }
    hint_text_result: Optional[str] = None
    weather_proposal_result: Dict[str, Optional[str]] = default_weather_proposal.copy()

    # --- Configuration Checks ---
    hint_llm_url = getattr(config, 'event_hint_llm_api_url', None)
    hint_llm_key = getattr(config, 'event_hint_llm_api_key', None)
    hint_llm_temp = getattr(config, 'event_hint_llm_temperature', 0.7)
    # Use the NEW template defined in this file by default
    hint_llm_template = getattr(config, 'event_hint_llm_prompt_template', DEFAULT_EVENT_HINT_TEMPLATE_TEXT)
    hint_history_count = getattr(config, 'event_hint_history_count', 6)

    if not hint_llm_url or not hint_llm_key:
        func_logger.debug(f"[{caller_info}] Event Hint LLM URL or Key missing. Skipping hint/weather generation.")
        return hint_text_result, weather_proposal_result # Return defaults
    # Use the updated default template if the one from config is missing/invalid
    if not hint_llm_template or not isinstance(hint_llm_template, str):
        func_logger.warning(f"[{caller_info}] Event Hint LLM prompt template invalid or missing in config. Using updated default (v3).")
        hint_llm_template = DEFAULT_EVENT_HINT_TEMPLATE_TEXT # v3 template
        if not hint_llm_template or not isinstance(hint_llm_template, str):
             func_logger.error(f"[{caller_info}] Default event hint template (v3) also invalid. Skipping.")
             return hint_text_result, weather_proposal_result # Return defaults

    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        func_logger.error(f"[{caller_info}] Invalid llm_call_func provided. Skipping.")
        return hint_text_result, weather_proposal_result # Return defaults

    # --- Prepare Inputs ---
    if not isinstance(history_messages, list):
        func_logger.warning(f"[{caller_info}] history_messages is not a list ({type(history_messages)}). Using empty history.")
        history_messages = []

    recent_history_list = get_recent_turns(
        history_messages, hint_history_count, DIALOGUE_ROLES, exclude_last=False
    )
    recent_history_str = format_history_for_llm(recent_history_list) if recent_history_list else "[No Recent History]"

    # Format the prompt using the helper (which uses the v3 template now)
    event_hint_prompt_text = _format_event_hint_prompt(
        recent_history_str=recent_history_str,
        background_context=background_context,
        current_season=current_season,
        current_weather=current_weather,
        current_time_of_day=current_time_of_day,
        template=hint_llm_template # Use the v3 template
    )

    if not event_hint_prompt_text or event_hint_prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed to format event hint prompt: {event_hint_prompt_text}. Skipping.")
        return hint_text_result, weather_proposal_result # Return defaults

    event_hint_payload = {"contents": [{"parts": [{"text": event_hint_prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling Event Hint LLM (v3 prompt for hint + weather)...")
    # Optional: Log the actual prompt being sent for debugging
    # func_logger.debug(f"[{caller_info}] Event Hint Prompt Text (v3):\n------\n{event_hint_prompt_text}\n------")

    # --- Call LLM ---
    success = False
    response_or_error = "LLM Call Not Attempted"
    try:
        success, response_or_error = await llm_call_func(
            api_url=hint_llm_url,
            api_key=hint_llm_key,
            payload=event_hint_payload,
            temperature=hint_llm_temp,
            timeout=60,
            caller_info=caller_info
        )
    except Exception as e_call:
        func_logger.error(f"[{caller_info}] Exception during event hint LLM call: {e_call}", exc_info=True)
        success = False
        response_or_error = f"LLM Call Exception: {type(e_call).__name__}"

    # --- Process Response ---
    if success and isinstance(response_or_error, str):
        llm_output_raw = response_or_error.strip()
        func_logger.debug(f"[{caller_info}] Hint LLM Raw Output:\n{llm_output_raw}")

        # Parse Hint Line
        hint_match = re.search(r"^Hint:(.*)$", llm_output_raw, re.MULTILINE | re.IGNORECASE)
        if hint_match:
            extracted_hint = hint_match.group(1).strip()
            if extracted_hint and extracted_hint.lower() != '[no suggestion]':
                hint_text_result = extracted_hint
                func_logger.info(f"[{caller_info}] Parsed Hint: '{hint_text_result[:80]}...'")
            else:
                func_logger.info(f"[{caller_info}] Parsed Hint: [No Suggestion]")
                hint_text_result = None # Explicitly None for no suggestion
        else:
            func_logger.warning(f"[{caller_info}] Could not parse 'Hint:' line from LLM output.")
            # Decide if we should still try to parse weather or return defaults
            # Let's try to parse weather anyway

        # Parse Weather Proposal Line
        weather_match = re.search(r"^WeatherProposal:(.*)$", llm_output_raw, re.MULTILINE | re.IGNORECASE)
        if weather_match:
            json_str = weather_match.group(1).strip()
            # Clean potential markdown code blocks around the JSON
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            try:
                parsed_json = json.loads(json_str)
                if isinstance(parsed_json, dict) and \
                   'previous_weather' in parsed_json and \
                   'new_weather' in parsed_json:
                    # Validate types (allow None for new_weather initially, though prompt asks for value)
                    prev_w = parsed_json['previous_weather']
                    new_w = parsed_json['new_weather']
                    if isinstance(prev_w, str) and (isinstance(new_w, str) or new_w is None):
                         weather_proposal_result = {
                             "previous_weather": prev_w,
                             "new_weather": new_w if isinstance(new_w, str) else prev_w # Default new to previous if null/invalid
                         }
                         func_logger.info(f"[{caller_info}] Parsed Weather Proposal: {weather_proposal_result}")
                    else:
                        func_logger.warning(f"[{caller_info}] Parsed Weather JSON values have incorrect types: prev={type(prev_w)}, new={type(new_w)}. Using defaults.")
                        # weather_proposal_result remains default
                else:
                    func_logger.warning(f"[{caller_info}] Parsed Weather JSON lacks required keys ('previous_weather', 'new_weather'). Using defaults. JSON: {json_str}")
                    # weather_proposal_result remains default
            except json.JSONDecodeError as e_json:
                func_logger.error(f"[{caller_info}] Failed to parse Weather Proposal JSON: {e_json}. Raw JSON string: '{json_str}'. Using defaults.")
                # weather_proposal_result remains default
            except Exception as e_parse:
                 func_logger.error(f"[{caller_info}] Unexpected error processing Weather Proposal JSON: {e_parse}. Raw JSON string: '{json_str}'. Using defaults.", exc_info=True)
                 # weather_proposal_result remains default
        else:
            func_logger.warning(f"[{caller_info}] Could not parse 'WeatherProposal:' line from LLM output. Using default weather proposal.")
            # weather_proposal_result remains default

    else: # LLM call failed or returned non-string
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict):
            error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.warning(f"[{caller_info}] Event Hint LLM call failed or returned invalid type. Error: '{error_details}'. Returning defaults.")
        # hint_text_result remains None
        # weather_proposal_result remains default

    # Return the final parsed results (or defaults if errors occurred)
    return hint_text_result, weather_proposal_result

# === END OF FILE i4_llm_agent/event_hints.py ===