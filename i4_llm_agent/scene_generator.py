# === START OF FILE i4_llm_agent/scene_generator.py ===
# i4_llm_agent/scene_generator.py

import logging
import json
import asyncio
from typing import Dict, Any, Optional, Callable, Coroutine, List, Tuple, Union

# Import history utils only if needed for extra context formatting (currently not planned)
# from .history import format_history_for_llm, get_recent_turns, DIALOGUE_ROLES

logger = logging.getLogger(__name__) # i4_llm_agent.scene_generator

# --- Constants ---

# Placeholders for the Scene Assessment/Generation prompt
PREVIOUS_RESPONSE_PLACEHOLDER = "{llm_response_N_minus_1}"
CURRENT_QUERY_PLACEHOLDER = "{user_query_N}"
PREVIOUS_KEYWORDS_PLACEHOLDER = "{previous_keywords_json}"
PREVIOUS_DESCRIPTION_PLACEHOLDER = "{previous_description}"

# Default prompt template for Scene Assessment/Generation LLM
DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT = f"""
[[SYSTEM ROLE: Scene Assessor & Generator]]

**Objective:** Analyze the current interaction context (last response, user query) against the previously established scene. Determine if the scene has significantly changed. If it has, generate a new description and keywords. If not, return the previous scene data unchanged.

**Inputs:**
1.  **Last LLM Response:** The narrative text generated at the end of the previous turn.
    ```text
    {PREVIOUS_RESPONSE_PLACEHOLDER}
    ```
2.  **Current User Query:** The user's input for the current turn.
    ```text
    {CURRENT_QUERY_PLACEHOLDER}
    ```
3.  **Previous Scene Keywords (JSON Array):** Keywords describing the established scene from the last turn.
    ```json
    {PREVIOUS_KEYWORDS_PLACEHOLDER}
    ```
4.  **Previous Scene Description:** The full text description of the established scene from the last turn.
    ```text
    {PREVIOUS_DESCRIPTION_PLACEHOLDER}
    ```

**Instructions:**

1.  **Analyze Context:** Read the 'Last LLM Response' and 'Current User Query'. Understand the action or focus of the current turn.
2.  **Assess Scene Change:** Compare the action/focus of the current turn against the 'Previous Scene Keywords' and 'Previous Scene Description'. Has the effective location, ambiance, or core setting fundamentally changed?
    *   Examples of **Change:** Explicitly moving to a new named location (inn -> stables, forest -> cave), environment drastically altering (calm -> battlefield), time shifting significantly causing ambiance change (day -> night).
    *   Examples of **No Change:** Interacting within the same location (ordering a drink in the tavern), minor shifts in focus within the same scene, continuation of dialogue.
3.  **Determine Output:**
    *   **If NO significant scene change is detected:** Your output **MUST** be the *exact* JSON object provided below, containing the *unchanged* previous keywords and description.
        ```json
        {{
          "keywords": {PREVIOUS_KEYWORDS_PLACEHOLDER},
          "description": "{PREVIOUS_DESCRIPTION_PLACEHOLDER}"
        }}
        ```
        *(Ensure the description string within the JSON is properly escaped if it contains quotes)*
    *   **If a significant scene change IS detected:**
        *   Generate a concise list of 3-5 new keywords describing the **new** scene (location, mood, key elements). Format as a JSON array of strings.
        *   Generate a brief (2-3 sentences) atmospheric description of the **new** scene, focusing on sensory details (sight, sound, smell).
        *   Your output **MUST** be a *new* JSON object containing the *new* keywords and *new* description.
            ```json
            {{
              "keywords": ["new_keyword_1", "new_keyword_2", ...],
              "description": "Your newly generated 2-3 sentence description of the new scene."
            }}
            ```

4.  **Output Format:** Respond ONLY with a single, valid JSON object matching one of the two structures described above. Do not include any other text, markdown formatting, or explanations outside the JSON object.

**JSON Output:**
"""

# --- Helper Functions ---
# === START MODIFIED _format_scene_assessment_prompt ===
def _format_scene_assessment_prompt(
    template: str,
    previous_llm_response: str,
    current_user_query: str,
    previous_keywords_json: str, # Expecting stringified JSON list
    previous_description: str,
    period_setting: Optional[str] = None # <<< NEW PARAMETER
) -> str:
    """
    Formats the prompt for the Scene Assessment/Generation LLM.
    Optionally prepends a period setting instruction.
    """
    func_logger = logging.getLogger(__name__ + '._format_scene_assessment_prompt')
    if not template or not isinstance(template, str):
        return "[Error: Invalid Template for Scene Assessment]"

    # --- NEW: Prepend Period Setting Instruction ---
    if period_setting and isinstance(period_setting, str):
        clean_period = period_setting.strip()
        if clean_period:
            instruction = f"[[Setting Instruction: Generate content appropriate for a '{clean_period}' setting.]]\n\n"
            template = instruction + template # Prepend the instruction
            func_logger.debug(f"Prepended period setting instruction: '{clean_period}'")
    # --- END NEW ---

    # Basic safety for inputs
    safe_prev_resp = str(previous_llm_response)
    safe_curr_query = str(current_user_query)
    safe_prev_keys = str(previous_keywords_json)
    safe_prev_desc = str(previous_description)

    try:
        # Use replace for safety, especially with potentially complex description strings
        formatted_prompt = template.replace(PREVIOUS_RESPONSE_PLACEHOLDER, safe_prev_resp)
        formatted_prompt = formatted_prompt.replace(CURRENT_QUERY_PLACEHOLDER, safe_curr_query)
        formatted_prompt = formatted_prompt.replace(PREVIOUS_KEYWORDS_PLACEHOLDER, safe_prev_keys)

        # Special handling for description placeholder within the "NO CHANGE" JSON example
        escaped_prev_desc_for_json = json.dumps(safe_prev_desc) # Get JSON escaped string
        if escaped_prev_desc_for_json.startswith('"') and escaped_prev_desc_for_json.endswith('"'):
             escaped_prev_desc_for_json = escaped_prev_desc_for_json[1:-1]

        formatted_prompt = formatted_prompt.replace(
            f'"{PREVIOUS_DESCRIPTION_PLACEHOLDER}"', # Target placeholder within example JSON
            f'"{escaped_prev_desc_for_json}"'        # Replace with escaped string in quotes
         )

        # Replace the placeholder for the description in the main input section
        formatted_prompt = formatted_prompt.replace(PREVIOUS_DESCRIPTION_PLACEHOLDER, safe_prev_desc)


        # Check if any placeholders might still exist (simple check)
        if any(ph in formatted_prompt for ph in [
            PREVIOUS_RESPONSE_PLACEHOLDER, CURRENT_QUERY_PLACEHOLDER,
            PREVIOUS_KEYWORDS_PLACEHOLDER, PREVIOUS_DESCRIPTION_PLACEHOLDER
        ]):
            func_logger.warning(f"Potential placeholder missed during .replace() formatting for scene prompt.")

        return formatted_prompt
    except Exception as e:
        func_logger.error(f"Error formatting scene assessment prompt: {e}", exc_info=True)
        return f"[Error formatting scene assessment prompt: {type(e).__name__}]"
# === END MODIFIED _format_scene_assessment_prompt ===


# --- Core Logic ---

# === START MODIFIED assess_and_generate_scene ===
async def assess_and_generate_scene(
    previous_llm_response: str,
    current_user_query: str,
    previous_scene_data: Optional[Dict[str, Any]], # Expects {"keywords": [...], "description": "..."} or None
    scene_llm_config: Dict[str, Any],
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    logger_instance: Optional[logging.Logger] = None,
    session_id: str = "unknown_session",
    period_setting: Optional[str] = None # <<< NEW PARAMETER
) -> Dict[str, Any]:
    """
    Assesses if the scene has changed based on the latest interaction and previous state.
    If changed, generates new keywords and description using an LLM.
    If not changed, returns the previous scene data.
    Optionally includes a period setting instruction in the LLM prompt.

    Args:
        previous_llm_response: The narrative response from the previous turn.
        current_user_query: The user's query for the current turn.
        previous_scene_data: Dict containing 'keywords' (list) and 'description' (str) from the last assessment, or None.
        scene_llm_config: Dict containing 'url', 'key', 'temp', 'prompt_template' for the Scene LLM.
        llm_call_func: The async function wrapper to call the LLM.
        logger_instance: Optional logger instance.
        session_id: The session ID for logging.
        period_setting: Optional string describing the historical period/setting (e.g., 'Late Medieval').

    Returns:
        A dictionary containing the effective 'keywords' (list) and 'description' (str) for the current turn.
        Returns previous data on LLM/parsing errors to maintain state.
        Returns default empty state if first turn and error occurs.
    """
    func_logger = logger_instance or logging.getLogger(__name__ + '.assess_and_generate_scene')
    caller_info = f"SceneAssessGen_{session_id}"

    # Define default empty/fallback state
    default_empty_scene = {"keywords": [], "description": ""}

    # Prepare previous state inputs for formatting, handling None case
    prev_keywords_list = []
    prev_description_str = ""
    if isinstance(previous_scene_data, dict):
        prev_keywords_list = previous_scene_data.get("keywords", [])
        prev_description_str = previous_scene_data.get("description", "")
        if not isinstance(prev_keywords_list, list): prev_keywords_list = []
        if not isinstance(prev_description_str, str): prev_description_str = ""
    previous_keywords_json_str = "[]" # Default to empty JSON array string
    try:
        previous_keywords_json_str = json.dumps(prev_keywords_list)
    except Exception:
        func_logger.warning(f"[{caller_info}] Failed to dump previous keywords to JSON string. Using empty list '[]'.")

    # --- Validate Config ---
    llm_url = scene_llm_config.get('url')
    llm_key = scene_llm_config.get('key')
    llm_temp = scene_llm_config.get('temp', 0.4) # Slightly higher temp might be okay for generation
    prompt_template = scene_llm_config.get('prompt_template', DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT)

    if not llm_url or not llm_key: func_logger.error(f"[{caller_info}] Scene LLM URL/Key missing. Returning previous state."); return previous_scene_data or default_empty_scene
    if not prompt_template or not isinstance(prompt_template, str): prompt_template = DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT; func_logger.warning(f"[{caller_info}] Invalid prompt template, using default.");
    if not prompt_template or not isinstance(prompt_template, str): func_logger.error(f"[{caller_info}] Default scene prompt template invalid. Cannot proceed."); return previous_scene_data or default_empty_scene
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func): func_logger.error(f"[{caller_info}] Invalid llm_call_func. Returning previous state."); return previous_scene_data or default_empty_scene

    # --- Format Prompt (MODIFIED: Pass period_setting) ---
    prompt_text = _format_scene_assessment_prompt(
        template=prompt_template,
        previous_llm_response=previous_llm_response,
        current_user_query=current_user_query,
        previous_keywords_json=previous_keywords_json_str,
        previous_description=prev_description_str,
        period_setting=period_setting # <-- PASSING NEW ARG
    )

    if not prompt_text or prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed format prompt: {prompt_text}. Returning previous state."); return previous_scene_data or default_empty_scene

    # --- Call LLM ---
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling Scene Assessment/Generation LLM...")

    success = False; response_or_error = "Init Error"
    try:
        success, response_or_error = await llm_call_func(
            api_url=llm_url, api_key=llm_key, payload=payload,
            temperature=llm_temp, timeout=60, caller_info=caller_info )
    except Exception as e_call:
        func_logger.error(f"[{caller_info}] Exception during LLM call: {e_call}", exc_info=True)
        success = False; response_or_error = f"LLM Call Exception: {type(e_call).__name__}"

    # --- Process Response ---
    if success and isinstance(response_or_error, str):
        llm_output_text = response_or_error.strip()
        try:
            # Basic cleanup - remove potential markdown fences
            if llm_output_text.startswith("```json"): llm_output_text = llm_output_text[7:]
            if llm_output_text.endswith("```"): llm_output_text = llm_output_text[:-3]
            llm_output_text = llm_output_text.strip()

            if not llm_output_text:
                func_logger.warning(f"[{caller_info}] Scene LLM returned empty string. Returning previous state.")
                return previous_scene_data or default_empty_scene

            parsed_json = json.loads(llm_output_text)

            # Validate structure
            if isinstance(parsed_json, dict) and \
               'keywords' in parsed_json and isinstance(parsed_json['keywords'], list) and \
               'description' in parsed_json and isinstance(parsed_json['description'], str):

                # Check if keywords are all strings (basic check)
                if all(isinstance(kw, str) for kw in parsed_json['keywords']):
                    func_logger.info(f"[{caller_info}] Scene LLM returned valid JSON structure. Keywords: {parsed_json['keywords']}, Desc len: {len(parsed_json['description'])}")
                    # Return the validated data from the LLM
                    return {
                        "keywords": parsed_json['keywords'],
                        "description": parsed_json['description']
                    }
                else:
                    func_logger.warning(f"[{caller_info}] Invalid keyword type in list: {parsed_json['keywords']}. Returning previous state.")
                    return previous_scene_data or default_empty_scene
            else:
                func_logger.warning(f"[{caller_info}] Scene LLM output failed structure validation. Type: {type(parsed_json)}, Keys: {parsed_json.keys() if isinstance(parsed_json, dict) else 'N/A'}. Output: {llm_output_text[:200]}... Returning previous state.")
                return previous_scene_data or default_empty_scene

        except json.JSONDecodeError as e_json:
            func_logger.error(f"[{caller_info}] Failed parse JSON response: {e_json}. Output: {llm_output_text[:200]}... Returning previous state.")
            return previous_scene_data or default_empty_scene
        except Exception as e_parse:
            func_logger.error(f"[{caller_info}] Error processing parsed JSON: {e_parse}. Output: {llm_output_text[:200]}... Returning previous state.", exc_info=True)
            return previous_scene_data or default_empty_scene
    else:
        # LLM call failed or returned non-string
        error_details = str(response_or_error)
        func_logger.warning(f"[{caller_info}] Scene LLM call failed or returned invalid type. Error: '{error_details}'. Returning previous state.")
        return previous_scene_data or default_empty_scene
# === END MODIFIED assess_and_generate_scene ===
