# === START OF FILE i4_llm_agent/world_state_parser.py ===
# i4_llm_agent/world_state_parser.py

import logging
import json
import asyncio # Required for Coroutine type hint
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple, Union

# Attempt import for history formatting (optional context for parser LLM)
try:
    from .history import format_history_for_llm, get_recent_turns, DIALOGUE_ROLES
except ImportError:
    DIALOGUE_ROLES = ["user", "assistant"]
    def get_recent_turns(*args, **kwargs): return []
    def format_history_for_llm(*args, **kwargs): return "[History Unavailable]"
    logging.getLogger(__name__).warning("Failed to import history utils in world_state_parser.py")


logger = logging.getLogger(__name__) # i4_llm_agent.world_state_parser

# --- Constants ---

# Placeholders for the parsing prompt
WS_PARSE_RESPONSE_PLACEHOLDER = "{llm_response_text}"
WS_PARSE_HISTORY_PLACEHOLDER = "{recent_history_str}"
WS_PARSE_CURRENT_DAY_PLACEHOLDER = "{current_day}"
WS_PARSE_CURRENT_TIME_PLACEHOLDER = "{current_time_of_day}"
WS_PARSE_CURRENT_WEATHER_PLACEHOLDER = "{current_weather}"
WS_PARSE_CURRENT_SEASON_PLACEHOLDER = "{current_season}"

# Default prompt template for the World State Parsing LLM
# (Remains unchanged with escaped braces for example JSON)
DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT = f"""
[[SYSTEM ROLE: World State Change Detector]]

**Objective:** Analyze the provided 'LLM Response Text' and 'Recent History' to identify explicit or strongly implied *changes* to the world state relative to the 'Current World State'. Output ONLY a JSON object containing the detected changes.

**Current World State (Reference Point):**
*   Day: {WS_PARSE_CURRENT_DAY_PLACEHOLDER}
*   Time of Day: {WS_PARSE_CURRENT_TIME_PLACEHOLDER}
*   Weather: {WS_PARSE_CURRENT_WEATHER_PLACEHOLDER}
*   Season: {WS_PARSE_CURRENT_SEASON_PLACEHOLDER}

**LLM Response Text (Analyze This Primarily):**
---
{WS_PARSE_RESPONSE_PLACEHOLDER}
---

**Recent History (For Context):**
---
{WS_PARSE_HISTORY_PLACEHOLDER}
---

**Instructions:**

1.  **Focus on Changes:** Compare the events described in the 'LLM Response Text' against the 'Current World State'. Identify ONLY clear changes.
2.  **Day Increment:** If the text indicates one or more full days have passed (e.g., "next morning", "the following day", "two days later"), set `day_increment` to the integer number of days passed (usually 1). Otherwise, set it to `0`.
3.  **Time of Day:** If the text explicitly states or strongly implies a *new* time of day (Morning, Afternoon, Evening, Night) that is *different* from the 'Current World State', set `time_of_day` to the new value (e.g., "Morning"). Otherwise, set it to `null`. If the day increments, the time *usually* resets to "Morning" unless specified otherwise in the text.
4.  **Weather:** If the text describes a *new* weather condition (e.g., "Rainy", "Clear", "Snowy", "Cloudy", "Stormy", "Windy", "Foggy", "Cool", "Warm") *different* from the 'Current World State', set `weather` to the new value. Otherwise, set it to `null`. Combine adjectives if appropriate (e.g., "Cloudy and Cool").
5.  **Season:** If the text explicitly mentions a *new* season (Spring, Summer, Autumn, Winter) *different* from the 'Current World State', set `season` to the new value. Otherwise, set it to `null`.
6.  **Output Format:** Respond ONLY with a valid JSON object with the following structure. Use `0` or `null` for fields where no change was detected relative to the current state.

    ```json
    {{{{  # Escaped literal brace
      "day_increment": <integer, 0 if no change>,
      "time_of_day": "<string>" | null,
      "weather": "<string>" | null,
      "season": "<string>" | null
    }}}}  # Escaped literal brace
    ```

7.  **Accuracy:** Only report changes explicitly stated or very strongly implied. Do not infer subtle shifts. If unsure, report no change (`0` or `null`).
8.  **No Changes:** If the text describes events happening within the *current* world state without changing it, output: `{{"day_increment": 0, "time_of_day": null, "weather": null, "season": null}}`

**JSON Output:**
"""

# --- Helper Functions ---

# <<< MODIFIED: Use .replace() instead of .format() >>>
def _format_world_state_parse_prompt(
    template: str,
    llm_response_text: str,
    recent_history_str: str,
    current_day: int,
    current_time: str,
    current_weather: str,
    current_season: str
) -> str:
    """Formats the prompt for the World State Parsing LLM using .replace()."""
    func_logger = logging.getLogger(__name__ + '._format_world_state_parse_prompt')
    if not template or not isinstance(template, str):
        return "[Error: Invalid Template for World State Parse]"

    # Still make values safe in case they contain braces
    safe_response = str(llm_response_text) # Ensure string
    safe_history = str(recent_history_str) # Ensure string
    safe_time = str(current_time) if current_time else "Unknown"
    safe_weather = str(current_weather) if current_weather else "Unknown"
    safe_season = str(current_season) if current_season else "Unknown"
    safe_day = str(current_day) # Ensure string

    try:
        # Perform sequential replacements
        formatted_prompt = template.replace(WS_PARSE_RESPONSE_PLACEHOLDER, safe_response)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_HISTORY_PLACEHOLDER, safe_history)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_CURRENT_DAY_PLACEHOLDER, safe_day)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_CURRENT_TIME_PLACEHOLDER, safe_time)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_CURRENT_WEATHER_PLACEHOLDER, safe_weather)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_CURRENT_SEASON_PLACEHOLDER, safe_season)

        # Basic check if placeholders might still exist (e.g., if a constant was misspelled)
        if any(ph in formatted_prompt for ph in [
            WS_PARSE_RESPONSE_PLACEHOLDER, WS_PARSE_HISTORY_PLACEHOLDER,
            WS_PARSE_CURRENT_DAY_PLACEHOLDER, WS_PARSE_CURRENT_TIME_PLACEHOLDER,
            WS_PARSE_CURRENT_WEATHER_PLACEHOLDER, WS_PARSE_CURRENT_SEASON_PLACEHOLDER
        ]):
            func_logger.warning(f"Potential placeholder missed during .replace() formatting.")
            # Optionally return an error or just the potentially broken string
            # return "[Error: Placeholder replacement failed]"

        return formatted_prompt
    except Exception as e:
        # Catch any unexpected errors during replacement
        func_logger.error(f"Error formatting world state parse prompt using .replace(): {e}", exc_info=True)
        return f"[Error formatting prompt with .replace(): {type(e).__name__}]"
# <<< END MODIFICATION >>>


# --- Core Logic (Remains the same) ---

async def parse_world_state_with_llm(
    llm_response_text: str,
    history_messages: List[Dict],
    current_state: Dict[str, Any],
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    ws_parse_llm_config: Dict[str, Any],
    logger_instance: Optional[logging.Logger] = None,
    session_id: str = "unknown_session",
) -> Dict[str, Any]:
    """
    Uses an LLM to parse the main LLM response for world state changes.

    Args:
        llm_response_text: The text generated by the main LLM.
        history_messages: Recent dialogue history for context.
        current_state: Dict containing current 'day', 'time_of_day', 'weather', 'season'.
        llm_call_func: The async function wrapper to call the LLM.
        ws_parse_llm_config: Dict containing 'url', 'key', 'temp', 'prompt_template'.
        logger_instance: Optional logger instance.
        session_id: The session ID for logging.

    Returns:
        A dictionary containing detected changes (keys: 'day_increment',
        'time_of_day', 'weather', 'season'). Returns an empty dictionary {}
        if parsing fails, the LLM fails, or no changes are detected.
    """
    func_logger = logger_instance or logging.getLogger(__name__ + '.parse_world_state_with_llm')
    caller_info = f"WorldStateParseLLM_{session_id}"
    detected_changes: Dict[str, Any] = {}

    # --- Validate Config ---
    llm_url = ws_parse_llm_config.get('url')
    llm_key = ws_parse_llm_config.get('key')
    llm_temp = ws_parse_llm_config.get('temp', 0.3) # Default to low temp for parsing
    prompt_template = ws_parse_llm_config.get('prompt_template', DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT) # Use default if not provided

    if not llm_url or not llm_key:
        func_logger.error(f"[{caller_info}] LLM URL or Key missing in config. Skipping parse.")
        return detected_changes
    if not prompt_template or not isinstance(prompt_template, str):
        func_logger.error(f"[{caller_info}] Prompt template invalid or missing. Skipping.")
        # Attempt to use the default template as a fallback if the config one was bad
        prompt_template = DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT
        if not prompt_template or not isinstance(prompt_template, str):
             func_logger.error(f"[{caller_info}] Default prompt template also invalid. Cannot proceed.")
             return detected_changes
        func_logger.warning(f"[{caller_info}] Using default prompt template due to invalid config.")

    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
         func_logger.error(f"[{caller_info}] Invalid llm_call_func provided. Skipping.")
         return detected_changes
    if not llm_response_text or not isinstance(llm_response_text, str):
        func_logger.debug(f"[{caller_info}] No LLM response text provided to parse. Skipping.")
        return detected_changes

    # --- Prepare Inputs ---
    # Use a small amount of history for context (e.g., last 4 turns)
    # Ensure history_messages is a list before processing
    if not isinstance(history_messages, list):
        func_logger.warning(f"[{caller_info}] history_messages is not a list ({type(history_messages)}). Using empty history.")
        history_messages = []

    history_context_turns = get_recent_turns(history_messages, 4, DIALOGUE_ROLES, exclude_last=False)
    recent_history_str = format_history_for_llm(history_context_turns)

    # Ensure current_state is a dictionary
    if not isinstance(current_state, dict):
        func_logger.error(f"[{caller_info}] current_state is not a dict ({type(current_state)}). Using defaults.")
        current_state = {}

    current_day = current_state.get('day', 1)
    current_time = current_state.get('time_of_day', 'Unknown')
    current_weather = current_state.get('weather', 'Unknown')
    current_season = current_state.get('season', 'Unknown')

    # Format the prompt using the modified helper
    prompt_text = _format_world_state_parse_prompt(
        template=prompt_template,
        llm_response_text=llm_response_text,
        recent_history_str=recent_history_str,
        current_day=current_day,
        current_time=current_time,
        current_weather=current_weather,
        current_season=current_season
    )

    if not prompt_text or prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed to format world state parse prompt: {prompt_text}. Skipping.")
        return detected_changes

    # --- Call LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling World State Parsing LLM...")
    # Optional: Log the prompt for debugging
    # func_logger.debug(f"[{caller_info}] World State Parse Prompt:\n------\n{prompt_text}\n------")

    try:
        success, response_or_error = await llm_call_func(
            api_url=llm_url,
            api_key=llm_key,
            payload=payload,
            temperature=llm_temp,
            timeout=45, # Parsing should be relatively quick
            caller_info=caller_info
        )
    except Exception as e_call:
        func_logger.error(f"[{caller_info}] Exception during WS Parse LLM call: {e_call}", exc_info=True)
        success = False
        response_or_error = f"LLM Call Exception: {type(e_call).__name__}"

    # --- Process Response ---
    if success and isinstance(response_or_error, str):
        llm_output_text = response_or_error.strip()
        # Attempt to parse the output as JSON
        try:
            # Clean potential markdown code blocks
            if llm_output_text.startswith("```json"):
                llm_output_text = llm_output_text[7:]
            if llm_output_text.endswith("```"):
                llm_output_text = llm_output_text[:-3]
            llm_output_text = llm_output_text.strip()

            # Handle empty string case
            if not llm_output_text:
                func_logger.warning(f"[{caller_info}] WS Parse LLM returned empty string.")
                return detected_changes # Return empty if LLM gave nothing back

            parsed_json = json.loads(llm_output_text)

            if isinstance(parsed_json, dict):
                # Extract and validate values, adding to detected_changes ONLY if changed
                day_inc = parsed_json.get('day_increment')
                if isinstance(day_inc, int) and day_inc > 0:
                    detected_changes['day_increment'] = day_inc
                    func_logger.debug(f"[{caller_info}] LLM detected day_increment: {day_inc}")

                time_of_day = parsed_json.get('time_of_day')
                # Check if it's a non-empty string and not the literal 'null'
                if isinstance(time_of_day, str) and time_of_day.strip() and time_of_day.lower() != 'null':
                    # Compare against current_time before adding
                    if time_of_day != current_time:
                        detected_changes['time_of_day'] = time_of_day
                        func_logger.debug(f"[{caller_info}] LLM detected time_of_day change: {time_of_day}")
                    else:
                         func_logger.debug(f"[{caller_info}] LLM reported time_of_day '{time_of_day}', but it matches current state. Ignoring.")

                weather = parsed_json.get('weather')
                if isinstance(weather, str) and weather.strip() and weather.lower() != 'null':
                     # Compare against current_weather before adding
                     if weather != current_weather:
                        detected_changes['weather'] = weather
                        func_logger.debug(f"[{caller_info}] LLM detected weather change: {weather}")
                     else:
                         func_logger.debug(f"[{caller_info}] LLM reported weather '{weather}', but it matches current state. Ignoring.")


                season = parsed_json.get('season')
                if isinstance(season, str) and season.strip() and season.lower() != 'null':
                     # Compare against current_season before adding
                     if season != current_season:
                        detected_changes['season'] = season
                        func_logger.debug(f"[{caller_info}] LLM detected season change: {season}")
                     else:
                        func_logger.debug(f"[{caller_info}] LLM reported season '{season}', but it matches current state. Ignoring.")


                if detected_changes:
                    func_logger.info(f"[{caller_info}] World state changes parsed from LLM: {detected_changes}")
                else:
                    func_logger.info(f"[{caller_info}] LLM parsing successful, but reported no state changes relative to current.")

            else:
                func_logger.warning(f"[{caller_info}] LLM output was valid JSON but not a dictionary: {type(parsed_json)}. Output: {llm_output_text}")

        except json.JSONDecodeError as e_json:
            func_logger.error(f"[{caller_info}] Failed to parse LLM output as JSON: {e_json}. Output: {llm_output_text}")
        except Exception as e_parse:
             func_logger.error(f"[{caller_info}] Error processing parsed JSON: {e_parse}. Output: {llm_output_text}", exc_info=True)

    elif not success:
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict):
            error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.warning(f"[{caller_info}] World State Parse LLM call failed. Error: '{error_details}'.")

    else: # Success but not a string?
         func_logger.warning(f"[{caller_info}] World State Parse LLM call succeeded but returned unexpected type: {type(response_or_error)}")


    return detected_changes

# === REMOVED OLD REGEX CODE ===

# === END OF FILE i4_llm_agent/world_state_parser.py ===