# i4_llm_agent/api_client.py

import requests
import logging
import urllib.parse
import json
# <<< MODIFIED Import Order: Tuple first >>>
from typing import Tuple, Union, Optional, Dict, List, Any

# Gets a logger named 'i4_llm_agent.api_client'
logger = logging.getLogger(__name__)

# <<< Return Type Hint uses Tuple >>>
def call_google_llm_api(
    api_url: str,
    api_key: str,
    payload: Dict[str, Any],
    temperature: float,
    timeout: int = 90,
    caller_info: str = "LLM"
) -> Tuple[bool, Union[str, Dict]]:
    """
    Calls a Google Generative Language API endpoint (standard format).

    Returns:
        Tuple[bool, Union[str, Dict]]: (success, result_or_error_dict)
    """
    logger.info(f"Attempting to call {caller_info} (Google Standard Format)...")

    # --- Configuration Checks ---
    if not api_url or not api_key:
        error_msg = "Missing API Key" if not api_key else "Missing LLM configuration (URL)"
        logger.error(f"[{caller_info}] {error_msg}.")
        return False, {"error_type": "ConfigurationError", "message": error_msg}

    # --- Prepare Payload and Headers ---
    if "generationConfig" not in payload:
         payload["generationConfig"] = {"temperature": temperature}
    elif "temperature" not in payload["generationConfig"]:
         payload["generationConfig"]["temperature"] = temperature

    final_api_url_with_key = f"{api_url}?key={urllib.parse.quote(api_key)}"
    headers = {"Content-Type": "application/json"}

    logger.debug(f"Calling {caller_info} API: URL={final_api_url_with_key}")

    # --- Make API Call ---
    try:
        response = requests.post(
            final_api_url_with_key, json=payload, headers=headers, timeout=timeout
        )
        logger.debug(f"{caller_info} API Response Status Code: {response.status_code}")
        response_text_snippet = response.text[:500]
        logger.debug(f"{caller_info} API Response Body Snippet: {response_text_snippet}...")

        response.raise_for_status() # Raises HTTPError for 4xx/5xx

        # --- Process Successful Response ---
        result = response.json()

        candidates = result.get("candidates")
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            first_candidate = candidates[0]
            content = first_candidate.get("content")
            if content and content.get("parts") and isinstance(content["parts"], list) and len(content["parts"]) > 0:
                text_part = content["parts"][0].get("text")
                if text_part is not None:
                    response_text = text_part.strip()
                    logger.info(f"Successfully parsed response from {caller_info}.")
                    return True, response_text # Success Return
                else:
                    logger.error(f"{caller_info} response 'text' field is null.")
                    return False, {"error_type": "ParsingError", "message": f"{caller_info}: Response text field is null", "response_body": result}
            else:
                finish_reason = first_candidate.get("finishReason")
                safety_ratings = first_candidate.get("safetyRatings")
                error_message = f"{caller_info}: Response missing 'content' or 'parts'."
                if finish_reason == "SAFETY": error_message = f"{caller_info}: Content blocked by safety filters: {safety_ratings}"
                elif finish_reason == "RECITATION": error_message = f"{caller_info}: Content blocked by recitation filters."
                elif finish_reason == "OTHER": error_message = f"{caller_info}: Content blocked for other reasons."
                logger.warning(error_message + f" Finish Reason: {finish_reason}")
                return False, {"error_type": "BlockedContent", "message": error_message, "finish_reason": finish_reason, "response_body": result}
        else:
            error_details = result.get("error")
            if error_details and isinstance(error_details, dict):
                api_error_message = error_details.get('message', 'Unknown API Error')
                logger.error(f"{caller_info} Google API error object: {error_details}")
                return False, {"error_type": "APIError", "message": f"{caller_info}: {api_error_message}", "status_code": error_details.get('code'), "response_body": result}
            elif "promptFeedback" in result and result["promptFeedback"].get("blockReason") == "SAFETY":
                 block_reason = f"{caller_info}: Prompt blocked by safety filters."
                 logger.warning(f"{block_reason} Details: {result.get('promptFeedback')}")
                 return False, {"error_type": "BlockedPrompt", "message": block_reason, "response_body": result}
            else:
                logger.error(f"Unexpected response from {caller_info} (no candidates/error field).")
                return False, {"error_type": "UnexpectedResponse", "message": f"{caller_info}: Unexpected API response structure (no candidates/error)", "response_body": result}

    # --- Handle Exceptions ---
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during {caller_info} API call: {e}", exc_info=False)
        return False, {"error_type": "TimeoutError", "message": f"{caller_info}: API request timed out"}
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        response_body_text = e.response.text
        logger.error(f"HTTPError {status_code} during {caller_info} API call: {e}", exc_info=False)
        logger.error(f"Response body on HTTPError: {response_body_text[:1000]}...")
        try:
            parsed_body = json.loads(response_body_text)
        except json.JSONDecodeError:
            parsed_body = response_body_text
        return False, {"error_type": "HTTPError", "message": f"{caller_info}: API request failed - HTTP {status_code}", "status_code": status_code, "response_body": parsed_body}
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException during {caller_info} API call: {e}", exc_info=False)
        return False, {"error_type": "RequestError", "message": f"{caller_info}: API request failed - {type(e).__name__}"}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from {caller_info}: {e}", exc_info=True)
        response_body_snippet = ""
        if "response" in locals() and hasattr(response, "text"):
             response_body_snippet = response.text[:500]
        return False, {"error_type": "JSONDecodeError", "message": f"{caller_info}: Failed to decode API response.", "details": response_body_snippet}
    except Exception as e:
        logger.error(f"Unexpected error processing {caller_info} API call: {e}", exc_info=True)
        return False, {"error_type": "UnexpectedError", "message": f"{caller_info}: Unexpected error - {type(e).__name__}"}