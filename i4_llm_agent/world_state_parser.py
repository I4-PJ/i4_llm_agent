# === START OF FILE i4_llm_agent/world_state_parser.py ===
# i4_llm_agent/world_state_parser.py

import logging
import json
import asyncio # Required for Coroutine type hint
import os # For path manipulation in debug logger
from datetime import datetime, timezone # For timestamp in debug logger
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

# Placeholders for the Day/Time parsing prompt
WS_PARSE_RESPONSE_PLACEHOLDER = "{llm_response_text}"
WS_PARSE_HISTORY_PLACEHOLDER = "{recent_history_str}"
WS_PARSE_CURRENT_DAY_PLACEHOLDER = "{current_day}"
WS_PARSE_CURRENT_TIME_PLACEHOLDER = "{current_time_of_day}"
# Removed Weather/Season placeholders

# === MODIFIED: Simplified Default prompt template for Day/Time Parsing ===
DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT = f"""
[[SYSTEM ROLE: Day and Time Change Detector]]

**Objective:** Analyze the provided 'LLM Response Text' and 'Recent History' to identify explicit or strongly implied *changes* ONLY to the Day number and Time of Day, relative to the 'Current State'. Output ONLY a JSON object containing the detected changes.

**Current State (Reference Point):**
*   Day: {WS_PARSE_CURRENT_DAY_PLACEHOLDER}
*   Time of Day: {WS_PARSE_CURRENT_TIME_PLACEHOLDER}

**LLM Response Text (Analyze This Primarily):**
---
{WS_PARSE_RESPONSE_PLACEHOLDER}
---

**Recent History (For Context):**
---
{WS_PARSE_HISTORY_PLACEHOLDER}
---

**Instructions:**

1.  **Focus on Day/Time Changes:** Compare the events described in the 'LLM Response Text' against the 'Current State'. Identify ONLY clear changes to Day or Time of Day.
2.  **Day Increment:** If the text indicates one or more full days have passed (e.g., "next morning", "the following day", "two days later"), set `day_increment` to the integer number of days passed (usually 1). Otherwise, set it to `0`.
3.  **Time of Day:** If the text explicitly states or strongly implies a *new* time of day (Morning, Afternoon, Evening, Night) that is *different* from the 'Current State', set `time_of_day` to the new value (e.g., "Morning"). Otherwise, set it to `null`. If the day increments, the time *usually* resets to "Morning" unless specified otherwise in the text. Ignore vague terms like "later".
4.  **Output Format:** Respond ONLY with a valid JSON object with the following structure. Use `0` or `null` for fields where no change was detected relative to the current state.

    ```json
    {{{{  # Escaped literal brace
      "day_increment": <integer, 0 if no change>,
      "time_of_day": "<string>" | null
    }}}}  # Escaped literal brace
    ```

5.  **Accuracy:** Only report changes explicitly stated or very strongly implied by narrative progression (e.g., mentioning sunrise implies morning). Do not infer subtle shifts. If unsure, report no change (`0` or `null`).
6.  **No Changes:** If the text describes events happening within the *current* day/time without changing it, output: `{{"day_increment": 0, "time_of_day": null}}`

**JSON Output:**
"""

# === NEW: Prompt Template for Weather Confirmation ===
WEATHER_CONFIRM_PROMPT_TEMPLATE = """
[[SYSTEM ROLE: Narrative Weather Consistency Check]]

**Task:** Determine if the 'LLM Response Text' explicitly contradicts the 'Proposed Weather Change'.

**Proposed Weather Change:** From '{previous_weather}' to '{new_weather}'.

**LLM Response Text (Narrative):**
---
{llm_response_text}
---

**Instructions:**
1. Read the 'LLM Response Text'.
2. Does the text contain any descriptions of weather that **clearly and directly contradict** the idea that the weather is now '{new_weather}'?
3. Focus on explicit statements (e.g., if proposed is "Rainy", look for "sun shining", "clear sky", "bright sun", "no clouds", etc.; if proposed is "Clear", look for "raining", "storm clouds", "snow falling", etc.). Ignore subtle implications or lack of mention.
4. Output ONLY "YES" if a clear contradiction exists in the text.
5. Output ONLY "NO" if no clear contradiction is found OR if weather is not mentioned at all.

**Contradiction Found (YES/NO):**
"""


# --- Helper Functions ---

# === MODIFIED: Simplified formatter for Day/Time prompt ===
def _format_world_state_parse_prompt(
    template: str,
    llm_response_text: str,
    recent_history_str: str,
    current_day: int,
    current_time: str,
    # Removed weather/season parameters
) -> str:
    """Formats the prompt for the Day/Time Parsing LLM using .replace()."""
    func_logger = logging.getLogger(__name__ + '._format_world_state_parse_prompt')
    if not template or not isinstance(template, str):
        return "[Error: Invalid Template for Day/Time Parse]"

    safe_response = str(llm_response_text)
    safe_history = str(recent_history_str)
    safe_time = str(current_time) if current_time else "Unknown"
    safe_day = str(current_day)

    try:
        # Perform sequential replacements for Day/Time only
        formatted_prompt = template.replace(WS_PARSE_RESPONSE_PLACEHOLDER, safe_response)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_HISTORY_PLACEHOLDER, safe_history)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_CURRENT_DAY_PLACEHOLDER, safe_day)
        formatted_prompt = formatted_prompt.replace(WS_PARSE_CURRENT_TIME_PLACEHOLDER, safe_time)

        # Basic check if placeholders might still exist
        if any(ph in formatted_prompt for ph in [
            WS_PARSE_RESPONSE_PLACEHOLDER, WS_PARSE_HISTORY_PLACEHOLDER,
            WS_PARSE_CURRENT_DAY_PLACEHOLDER, WS_PARSE_CURRENT_TIME_PLACEHOLDER
        ]):
            func_logger.warning(f"Potential placeholder missed during .replace() formatting for Day/Time prompt.")

        return formatted_prompt
    except Exception as e:
        func_logger.error(f"Error formatting day/time parse prompt using .replace(): {e}", exc_info=True)
        return f"[Error formatting prompt with .replace(): {type(e).__name__}]"

# === NEW: Formatter for Weather Confirmation prompt ===
def _format_weather_confirm_prompt(
    template: str,
    proposed_weather_change: Dict[str, Optional[str]],
    llm_response_text: str
) -> str:
    """Formats the prompt for the Weather Confirmation LLM using .format()."""
    func_logger = logging.getLogger(__name__ + '._format_weather_confirm_prompt')
    if not template or not isinstance(template, str):
        return "[Error: Invalid Template for Weather Confirm]"
    if not isinstance(proposed_weather_change, dict):
        return "[Error: Invalid weather proposal dictionary]"

    prev_w = str(proposed_weather_change.get("previous_weather", "Unknown"))
    new_w = str(proposed_weather_change.get("new_weather", "Unknown"))
    safe_response = str(llm_response_text).replace("{", "{{").replace("}", "}}") # Escape braces in narrative

    try:
        # Use .format() as the template itself contains placeholders
        formatted_prompt = template.format(
            previous_weather=prev_w,
            new_weather=new_w,
            llm_response_text=safe_response
        )
        return formatted_prompt
    except KeyError as e:
        func_logger.error(f"Missing placeholder in weather confirm prompt: {e}")
        return f"[Error: Missing placeholder '{e}']"
    except Exception as e:
        func_logger.error(f"Error formatting weather confirm prompt: {e}", exc_info=True)
        return f"[Error formatting weather confirm prompt: {type(e).__name__}]"


# --- Core Logic ---

# === MODIFIED: Focused Day/Time Parser LLM Call ===
async def parse_world_state_with_llm(
    llm_response_text: str,
    history_messages: List[Dict],
    current_day: int, # Needs current day for prompt
    current_time_of_day: str, # Needs current time for prompt
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    ws_parse_llm_config: Dict[str, Any], # Config for THIS LLM call
    logger_instance: Optional[logging.Logger] = None,
    session_id: str = "unknown_session",
    debug_path_getter: Optional[Callable[[str], Optional[str]]] = None
) -> Dict[str, Any]:
    """
    Uses an LLM to parse the main LLM response for **Day and Time of Day** changes only.

    Args:
        llm_response_text: The text generated by the main LLM.
        history_messages: Recent dialogue history for context.
        current_day: Current day number (for prompt context).
        current_time_of_day: Current time string (for prompt context).
        llm_call_func: The async function wrapper to call the LLM.
        ws_parse_llm_config: Dict containing 'url', 'key', 'temp', 'prompt_template'
                             for the Day/Time parsing LLM.
        logger_instance: Optional logger instance.
        session_id: The session ID for logging.
        debug_path_getter: Optional function to get the debug log path.

    Returns:
        A dictionary containing detected changes (keys: 'day_increment', 'time_of_day').
        Returns an empty dictionary {} if parsing fails or no changes are detected.
    """
    func_logger = logger_instance or logging.getLogger(__name__ + '.parse_world_state_with_llm')
    caller_info = f"DayTimeParseLLM_{session_id}"
    detected_changes: Dict[str, Any] = {"day_increment": 0, "time_of_day": None} # Initialize defaults

    # --- Internal Debug Logging Helper (Same as before) ---
    ws_parse_debug_log_path: Optional[str] = None
    ws_parse_debug_log_failed: bool = False
    def _log_ws_parse_debug(content: Any, is_input: bool, log_suffix: str):
        nonlocal ws_parse_debug_log_path, ws_parse_debug_log_failed
        if ws_parse_debug_log_failed: return
        if ws_parse_debug_log_path is None and callable(debug_path_getter):
            try:
                ws_parse_debug_log_path = debug_path_getter(log_suffix) # Use provided suffix
                if ws_parse_debug_log_path is None: ws_parse_debug_log_failed = True
            except Exception as e_get_path: ws_parse_debug_log_path = None; ws_parse_debug_log_failed = True
        if not ws_parse_debug_log_path:
            if not ws_parse_debug_log_failed: func_logger.error(f"[{caller_info}] WS Parse Debug: Cannot log {log_suffix}, no valid path."); ws_parse_debug_log_failed = True
            return
        try:
            ts = datetime.now(timezone.utc).isoformat(); log_type = "INPUT_PROMPT" if is_input else "RAW_OUTPUT"; log_content_str = str(content)
            log_line = f"\n--- [{ts}] SESSION: {session_id} - {log_suffix}_{log_type} --- START ---\n{log_content_str}\n--- [{ts}] SESSION: {session_id} - {log_suffix}_{log_type} --- END ---\n\n"
            with open(ws_parse_debug_log_path, "a", encoding="utf-8") as f: f.write(log_line)
        except Exception as e_log: func_logger.error(f"[{caller_info}] WS Parse Debug: Error writing {log_suffix} to log file '{ws_parse_debug_log_path}': {e_log}", exc_info=False); ws_parse_debug_log_failed = True
    # --- End Internal Helper ---

    # --- Validate Config ---
    llm_url = ws_parse_llm_config.get('url')
    llm_key = ws_parse_llm_config.get('key')
    llm_temp = ws_parse_llm_config.get('temp', 0.3)
    prompt_template = ws_parse_llm_config.get('prompt_template', DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT)

    if not llm_url or not llm_key: func_logger.error(f"[{caller_info}] LLM URL/Key missing. Skipping."); return {}
    if not prompt_template or not isinstance(prompt_template, str): prompt_template = DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT; func_logger.warning(f"[{caller_info}] Invalid prompt template, using default.");
    if not prompt_template or not isinstance(prompt_template, str): func_logger.error(f"[{caller_info}] Default prompt template invalid. Cannot proceed."); return {}
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func): func_logger.error(f"[{caller_info}] Invalid llm_call_func. Skipping."); return {}
    if not llm_response_text or not isinstance(llm_response_text, str): func_logger.debug(f"[{caller_info}] No response text. Skipping."); return {}

    # --- Prepare Inputs ---
    if not isinstance(history_messages, list): history_messages = []; func_logger.warning(f"[{caller_info}] history_messages not list.")
    history_context_turns = get_recent_turns(history_messages, 4, DIALOGUE_ROLES, exclude_last=False)
    recent_history_str = format_history_for_llm(history_context_turns)

    # Format the simplified prompt
    prompt_text = _format_world_state_parse_prompt(
        template=prompt_template,
        llm_response_text=llm_response_text,
        recent_history_str=recent_history_str,
        current_day=current_day,
        current_time=current_time_of_day # Pass current time for context
    )

    if not prompt_text or prompt_text.startswith("[Error:"): func_logger.error(f"[{caller_info}] Failed format prompt: {prompt_text}. Skipping."); return {}

    # --- Call LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling Day/Time Parsing LLM...")
    _log_ws_parse_debug(content=prompt_text, is_input=True, log_suffix=".DEBUG_DAYTIME_PARSE") # Use specific suffix

    success = False; response_or_error = "Init Error"
    try:
        success, response_or_error = await llm_call_func(
            api_url=llm_url, api_key=llm_key, payload=payload,
            temperature=llm_temp, timeout=45, caller_info=caller_info )
    except Exception as e_call: func_logger.error(f"[{caller_info}] Exception during LLM call: {e_call}", exc_info=True); success = False; response_or_error = f"LLM Call Exception: {type(e_call).__name__}"

    _log_ws_parse_debug(content=response_or_error, is_input=False, log_suffix=".DEBUG_DAYTIME_PARSE") # Use specific suffix

    # --- Process Response ---
    if success and isinstance(response_or_error, str):
        llm_output_text = response_or_error.strip()
        try:
            if llm_output_text.startswith("```json"): llm_output_text = llm_output_text[7:]
            if llm_output_text.endswith("```"): llm_output_text = llm_output_text[:-3]
            llm_output_text = llm_output_text.strip()

            if not llm_output_text: func_logger.warning(f"[{caller_info}] LLM returned empty string."); return {}

            parsed_json = json.loads(llm_output_text)
            if isinstance(parsed_json, dict):
                # Extract Day Increment
                day_inc = parsed_json.get('day_increment')
                if isinstance(day_inc, int) and day_inc >= 0: # Allow 0
                    detected_changes['day_increment'] = day_inc
                    func_logger.debug(f"[{caller_info}] LLM detected day_increment: {day_inc}")
                else:
                    func_logger.warning(f"[{caller_info}] Invalid 'day_increment' in response: {day_inc}. Using 0.")
                    detected_changes['day_increment'] = 0

                # Extract Time of Day
                time_of_day = parsed_json.get('time_of_day')
                if isinstance(time_of_day, str) and time_of_day.strip() and time_of_day.lower() != 'null':
                    # Check if it differs from current time before storing change
                    if time_of_day != current_time_of_day:
                        detected_changes['time_of_day'] = time_of_day
                        func_logger.debug(f"[{caller_info}] LLM detected time_of_day change: {time_of_day}")
                    else:
                         func_logger.debug(f"[{caller_info}] LLM reported time '{time_of_day}', matches current. No change recorded.")
                         detected_changes['time_of_day'] = None # Explicitly no change
                elif time_of_day is None or (isinstance(time_of_day, str) and time_of_day.lower() == 'null'):
                    func_logger.debug(f"[{caller_info}] LLM reported null time_of_day. No change recorded.")
                    detected_changes['time_of_day'] = None # Explicitly no change
                else:
                    func_logger.warning(f"[{caller_info}] Invalid 'time_of_day' in response: {time_of_day}. Recording no change.")
                    detected_changes['time_of_day'] = None

                # Log final detected changes for Day/Time
                if detected_changes.get('day_increment', 0) > 0 or detected_changes.get('time_of_day') is not None:
                    func_logger.info(f"[{caller_info}] Day/Time changes parsed from LLM: {detected_changes}")
                else:
                    func_logger.info(f"[{caller_info}] Day/Time LLM parsing successful, reported no state changes.")

            else: func_logger.warning(f"[{caller_info}] LLM output not dict: {type(parsed_json)}. Output: {llm_output_text}")
        except json.JSONDecodeError as e_json: func_logger.error(f"[{caller_info}] Failed parse JSON: {e_json}. Output: {llm_output_text}")
        except Exception as e_parse: func_logger.error(f"[{caller_info}] Error processing parsed JSON: {e_parse}. Output: {llm_output_text}", exc_info=True)

    elif not success:
        error_details = str(response_or_error); func_logger.warning(f"[{caller_info}] Day/Time Parse LLM call failed. Error: '{error_details}'.")
    else: func_logger.warning(f"[{caller_info}] Day/Time Parse LLM call returned unexpected type: {type(response_or_error)}")

    # Return ONLY the detected Day/Time changes (or defaults)
    return detected_changes


# === NEW: Weather Confirmation LLM Call ===
async def confirm_weather_change_with_llm(
    proposed_weather_change: Dict[str, Optional[str]],
    llm_response_text: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    weather_confirm_llm_config: Dict[str, Any], # Expects Inventory LLM config passed here
    logger_instance: Optional[logging.Logger] = None,
    session_id: str = "unknown_session",
    debug_path_getter: Optional[Callable[[str], Optional[str]]] = None # Allow debug logging
) -> bool:
    """
    Uses an LLM to check if the narrative response contradicts a proposed weather change.

    Args:
        proposed_weather_change: Dict containing 'previous_weather' and 'new_weather'.
        llm_response_text: The text generated by the main LLM (the narrative).
        llm_call_func: The async function wrapper to call the LLM.
        weather_confirm_llm_config: Dict containing 'url', 'key', 'temp' for this call
                                    (intended to use Inventory LLM config).
        logger_instance: Optional logger instance.
        session_id: The session ID for logging.
        debug_path_getter: Optional function to get the debug log path.

    Returns:
        True if the LLM response indicates a contradiction ("YES"),
        False otherwise ("NO" or error).
    """
    func_logger = logger_instance or logging.getLogger(__name__ + '.confirm_weather_change_with_llm')
    caller_info = f"WeatherConfirmLLM_{session_id}"
    contradiction_found = False # Default to no contradiction

    # --- Internal Debug Logging Helper (Same structure as above) ---
    wc_debug_log_path: Optional[str] = None
    wc_debug_log_failed: bool = False
    def _log_wc_debug(content: Any, is_input: bool):
        nonlocal wc_debug_log_path, wc_debug_log_failed
        if wc_debug_log_failed: return
        if wc_debug_log_path is None and callable(debug_path_getter):
            try: wc_debug_log_path = debug_path_getter(".DEBUG_WEATHER_CONFIRM");
            except Exception: wc_debug_log_path = None
        if not wc_debug_log_path:
             if not wc_debug_log_failed: func_logger.error(f"[{caller_info}] WC Debug: Cannot log, no path."); wc_debug_log_failed = True;
             return
        try:
            ts=datetime.now(timezone.utc).isoformat(); log_type="INPUT" if is_input else "OUTPUT"; log_content=str(content)
            log_line=f"\n--- [{ts}] {session_id} WEATHER_CONFIRM {log_type} ---\n{log_content}\n--- END ---\n\n"
            with open(wc_debug_log_path, "a", encoding="utf-8") as f: f.write(log_line)
        except Exception as e: func_logger.error(f"[{caller_info}] WC Debug Error writing log: {e}"); wc_debug_log_failed = True
    # --- End Internal Helper ---

    # --- Validate Inputs & Config ---
    if not isinstance(proposed_weather_change, dict) or \
       "previous_weather" not in proposed_weather_change or \
       "new_weather" not in proposed_weather_change:
        func_logger.error(f"[{caller_info}] Invalid proposed_weather_change dict. Skipping check.")
        return False # Cannot check without proposal

    if not llm_response_text or not isinstance(llm_response_text, str):
        func_logger.debug(f"[{caller_info}] No narrative text provided. Assuming no contradiction.")
        return False # Nothing to check against

    llm_url = weather_confirm_llm_config.get('url')
    llm_key = weather_confirm_llm_config.get('key')
    llm_temp = weather_confirm_llm_config.get('temp', 0.3) # Use temp from config

    if not llm_url or not llm_key:
        func_logger.error(f"[{caller_info}] LLM URL/Key missing in weather_confirm_llm_config. Skipping check.")
        return False
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        func_logger.error(f"[{caller_info}] Invalid llm_call_func. Skipping check.")
        return False

    # --- Prepare and Format Prompt ---
    prompt_text = _format_weather_confirm_prompt(
        template=WEATHER_CONFIRM_PROMPT_TEMPLATE,
        proposed_weather_change=proposed_weather_change,
        llm_response_text=llm_response_text
    )

    if not prompt_text or prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed format weather confirm prompt: {prompt_text}. Skipping.")
        return False

    # --- Call LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling Weather Confirmation LLM...")
    _log_wc_debug(content=prompt_text, is_input=True)

    success = False; response_or_error = "Init Error"
    try:
        success, response_or_error = await llm_call_func(
            api_url=llm_url, api_key=llm_key, payload=payload,
            temperature=llm_temp, timeout=30, # Should be very fast
            caller_info=caller_info
        )
    except Exception as e_call:
        func_logger.error(f"[{caller_info}] Exception during LLM call: {e_call}", exc_info=True)
        success = False; response_or_error = f"LLM Call Exception: {type(e_call).__name__}"

    _log_wc_debug(content=response_or_error, is_input=False)

    # --- Process Response ---
    if success and isinstance(response_or_error, str):
        llm_output = response_or_error.strip().upper()
        if llm_output == "YES":
            contradiction_found = True
            func_logger.info(f"[{caller_info}] LLM detected narrative contradiction with proposed weather.")
        elif llm_output == "NO":
            contradiction_found = False
            func_logger.info(f"[{caller_info}] LLM found no narrative contradiction with proposed weather.")
        else:
            func_logger.warning(f"[{caller_info}] LLM returned unexpected output for YES/NO: '{response_or_error}'. Assuming NO contradiction.")
            contradiction_found = False
    else:
        error_details = str(response_or_error)
        func_logger.warning(f"[{caller_info}] Weather Confirm LLM call failed or returned invalid type. Error: '{error_details}'. Assuming NO contradiction.")
        contradiction_found = False # Default to false on error

    return contradiction_found


# === END OF FILE i4_llm_agent/world_state_parser.py ===