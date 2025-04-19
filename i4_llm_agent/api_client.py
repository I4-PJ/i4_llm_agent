# [[START MODIFIED api_client.py]]
# i4_llm_agent/api_client.py

import requests
import logging
import urllib.parse # Needed for URL parsing
import json
import os
from typing import Tuple, Union, Optional, Dict, List, Any

# Gets a logger named 'i4_llm_agent.api_client'
logger = logging.getLogger(__name__)
# Basic logging config if running standalone for testing
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')


# --- Helper: Payload Conversion ---
def _convert_google_to_openai_payload(
    google_payload: Dict[str, Any], model_name: str, temperature: float
) -> Dict[str, Any]:
    """
    Converts Google 'contents' format to OpenAI 'messages' format.
    Handles system prompt extraction and ACK skipping.
    """
    openai_messages = []
    system_prompt = None
    google_contents = google_payload.get("contents", [])

    skip_next_model_turn = False

    for i, turn in enumerate(google_contents):
        if skip_next_model_turn:
            skip_next_model_turn = False # Reset flag
            if turn.get("role") == "model":
                 logger.debug("Skipping model ACK turn based on previous system instruction.")
                 continue
            else:
                 # This case shouldn't normally happen if history is well-formed
                 logger.warning("Expected model ACK turn to skip, but found different role. Processing normally.")


        role = turn.get("role")
        parts = turn.get("parts", [])
        text = parts[0].get("text", "") if parts else ""

        if not role or not isinstance(text, str):
            logger.warning(f"Skipping turn with invalid role or text content: {turn}")
            continue

        if role == "user":
            # Check if this is the first user message AND looks like a system prompt
            # Using startswith for flexibility (e.g., "System Prompt:", "SYSTEM INSTRUCTIONS:")
            lower_text_strip = text.strip().lower()
            if i == 0 and (
                lower_text_strip.startswith("system instructions:") or
                lower_text_strip.startswith("system prompt:") or
                lower_text_strip.startswith("system directive:")
            ):
                 lines = text.split('\n', 1)
                 # Extract the content after the indicator line
                 system_prompt_content = lines[1].strip() if len(lines) > 1 else ""
                 if system_prompt_content:
                     system_prompt = system_prompt_content
                     logger.debug("Extracted system prompt from first user message.")

                     # Check if the *very next* turn is a simple model ACK to skip
                     if i + 1 < len(google_contents):
                         next_turn = google_contents[i+1]
                         next_role = next_turn.get("role")
                         next_parts = next_turn.get("parts", [])
                         next_text = next_parts[0].get("text", "") if next_parts else ""
                         # More robust ACK check, case-insensitive, stripping punctuation
                         ack_texts = {"understood", "ok", "okay", "i understand"}
                         processed_next_text = next_text.strip().lower().rstrip('.!')
                         if next_role == "model" and processed_next_text in ack_texts:
                             skip_next_model_turn = True # Set flag to skip the ACK on the next iteration
                             logger.debug(f"Flagging model ACK ('{next_text}') for skipping after system prompt extraction.")
                     # Regardless of ACK, continue to next turn; don't add system prompt user message itself
                     continue
                 else:
                     # If the system prompt content is empty after the indicator, treat it as a normal message
                     logger.warning("Found system prompt indicator in first user message, but content was empty. Treating as normal message.")

            # Add regular user message
            openai_messages.append({"role": "user", "content": text})

        elif role == "model":
            # Add assistant message (already checked for ACK skip above)
            openai_messages.append({"role": "assistant", "content": text})
        else:
            logger.warning(f"Unknown role '{role}' encountered during conversion. Skipping turn.")

    # Construct final OpenAI payload
    openai_payload = {
        "model": model_name,
        "messages": openai_messages,
        "temperature": temperature,
        # Add other common OpenAI params if needed, like max_tokens, top_p etc.
        # Note: Need mechanism to pass these if required, maybe via generationConfig?
        # "max_tokens": google_payload.get("generationConfig", {}).get("maxOutputTokens"), # Example
    }

    # Prepend the extracted system prompt if it exists
    if system_prompt:
        # Ensure system prompt isn't added if messages list is empty (edge case)
        if openai_payload["messages"]:
            openai_payload["messages"].insert(0, {"role": "system", "content": system_prompt})
            logger.debug("Prepended system prompt to OpenAI messages.")
        else:
            # If only a system prompt was found, add it as the first message
             openai_payload["messages"].append({"role": "system", "content": system_prompt})
             logger.debug("Added system prompt as the only message.")


    logger.debug(f"Payload conversion complete. OpenAI messages count: {len(openai_payload['messages'])}")
    return openai_payload


# --- Internal Google API Client Implementation ---
def _execute_google_llm_api( # Renamed with underscore
    api_url: str, # Base URL without fragment or key
    api_key: str,
    payload: Dict[str, Any], # Google 'contents' format
    temperature: float,
    timeout: int = 90,
    caller_info: str = "LLM_Google_Exec" # Updated caller info
) -> Tuple[bool, Union[str, Dict]]:
    """Internal function to execute call against Google Generative Language API."""
    logger.info(f"Executing Google API call via _execute_google_llm_api for URL: {api_url[:80]}...")
    # URL and Key validation happens in the dispatcher now, but double-check is fine
    if not api_url or not api_key:
        error_msg = "Missing API Key" if not api_key else "Missing Google URL"
        logger.error(f"[{caller_info}] {error_msg}.")
        # Return standardized error dict
        return False, {"error_type": "ConfigurationError", "message": error_msg, "source": "google_exec"}

    # Ensure generationConfig exists and set temperature
    if "generationConfig" not in payload:
        payload["generationConfig"] = {}
    # Ensure temperature is float (might come from valve as int/str)
    payload["generationConfig"]["temperature"] = float(temperature)

    try:
        # Add API key as query parameter
        query_separator = '&' if '?' in api_url else '?'
        final_api_url_with_key = f"{api_url}{query_separator}key={urllib.parse.quote(api_key)}"
    except Exception as e:
        logger.error(f"[{caller_info}] Error constructing Google API URL with key: {e}", exc_info=True)
        return False, {"error_type": "ConfigurationError", "message": "Invalid Google API URL format for key insertion", "source": "google_exec"}

    headers = {"Content-Type": "application/json"}
    logger.debug(f"[{caller_info}] Calling Google API: URL starts '{final_api_url_with_key[:80]}...'")
    # logger.debug(f"[{caller_info}] Google Payload: {json.dumps(payload)}") # Uncomment for deep debug

    try:
        response = requests.post(final_api_url_with_key, json=payload, headers=headers, timeout=timeout)
        logger.debug(f"[{caller_info}] Response Status Code: {response.status_code}")
        response_text_snippet = response.text[:500] if response.text else "[EMPTY RESPONSE BODY]"
        logger.debug(f"[{caller_info}] Response Body Snippet: {response_text_snippet}...")

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Attempt to parse JSON only on success status codes
        try:
            result = response.json()
        except json.JSONDecodeError as e_json:
             logger.error(f"[{caller_info}] Failed JSON decode for successful response (Status {response.status_code}): {e_json}", exc_info=False)
             logger.error(f"[{caller_info}] RAW Body causing JSONDecodeError: >>>{response.text}<<<")
             return False, {"error_type": "JSONDecodeError", "message": f"{caller_info}: Failed JSON decode of successful response.", "response_body": response.text, "status_code": response.status_code, "source": "google_exec"}

        # Process the parsed JSON response
        candidates = result.get("candidates")

        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            first_candidate = candidates[0]
            content = first_candidate.get("content")

            if content and content.get("parts") and isinstance(content["parts"], list) and len(content["parts"]) > 0:
                text_part = content["parts"][0].get("text")
                if text_part is not None: # Allows empty string "" as valid response
                    logger.info(f"[{caller_info}] Successfully parsed response text.")
                    # SUCCESS: Return True and the extracted text string
                    return True, text_part # .strip() removed - preserve leading/trailing spaces if LLM returns them
                else:
                    # This means "text": null was explicitly returned
                    logger.error(f"[{caller_info}] Response 'text' field is null inside parts.")
                    return False, {"error_type": "ParsingError", "message": f"{caller_info}: Response text field is null", "response_body": result, "source": "google_exec"}
            else:
                # Handle cases like blocked content based on finishReason, even if parts are missing
                finish_reason = first_candidate.get("finishReason")
                safety_ratings = first_candidate.get("safetyRatings")
                error_message = f"{caller_info}: Response missing 'content' or 'parts'."
                error_type = "ParsingError" # Default if no specific block reason

                if finish_reason == "SAFETY":
                    error_message = f"{caller_info}: Content blocked by safety filters: {safety_ratings}"
                    error_type = "BlockedContent"
                elif finish_reason == "RECITATION":
                    error_message = f"{caller_info}: Content blocked by recitation filters."
                    error_type = "BlockedContent"
                elif finish_reason == "OTHER":
                     error_message = f"{caller_info}: Content blocked for other reasons: {safety_ratings}" # Include safety ratings if available
                     error_type = "BlockedContent"
                elif finish_reason == "MAX_TOKENS":
                     error_message = f"{caller_info}: Response stopped due to max tokens limit, but content/parts missing."
                     error_type = "ParsingError" # It finished, but we can't parse expected output
                elif finish_reason: # Any other reason (e.g., STOP) but content missing
                    error_message = f"{caller_info}: Finished with reason '{finish_reason}' but missing content/parts."
                    error_type = "ParsingError" # Treat as parsing issue if content absent

                logger.warning(f"{error_message} Finish Reason: {finish_reason}")
                return False, {"error_type": error_type, "message": error_message, "finish_reason": finish_reason, "response_body": result, "source": "google_exec"}
        else:
            # Handle API-level errors reported in the JSON body (e.g., invalid API key)
            error_details = result.get("error")
            if error_details and isinstance(error_details, dict):
                api_error_message = error_details.get('message', 'Unknown Google API Error')
                api_error_code = error_details.get('code') # This is Google's specific code (e.g., 400)
                api_error_status = error_details.get('status') # e.g., 'INVALID_ARGUMENT'
                logger.error(f"[{caller_info}] Google API error: Code={api_error_code}, Status={api_error_status}, Msg='{api_error_message}'")
                logger.debug(f"Full Google API error: {error_details}")
                # Pass Google's error code if available, otherwise HTTP status
                return False, {"error_type": "APIError", "message": f"{caller_info}: {api_error_message}", "status_code": api_error_code or response.status_code, "api_status": api_error_status, "response_body": result, "source": "google_exec"}
            # Handle prompt feedback block reason (if top-level error doesn't exist)
            elif "promptFeedback" in result and result["promptFeedback"].get("blockReason"):
                 block_reason = result["promptFeedback"]["blockReason"]
                 block_details = result.get('promptFeedback') # Get full feedback details
                 error_message = f"{caller_info}: Prompt blocked. Reason: {block_reason}"
                 logger.warning(f"{error_message} Details: {block_details}")
                 return False, {"error_type": "BlockedPrompt", "message": error_message, "block_reason": block_reason, "response_body": result, "source": "google_exec"}
            else:
                # Response structure is unexpected (e.g., empty JSON, missing 'candidates' and 'error')
                logger.error(f"[{caller_info}] Unexpected successful API response structure (missing candidates/error).")
                return False, {"error_type": "UnexpectedResponse", "message": f"{caller_info}: Unexpected API response structure", "response_body": result, "source": "google_exec"}

    # --- Exception Handling ---
    except requests.exceptions.Timeout as e:
        logger.error(f"[{caller_info}] Timeout after {timeout}s: {e}", exc_info=False)
        return False, {"error_type": "TimeoutError", "message": f"{caller_info}: API request timed out after {timeout}s", "source": "google_exec"}
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        response_body_text = e.response.text
        logger.error(f"[{caller_info}] HTTPError {status_code}: {e}", exc_info=False)
        logger.error(f"[{caller_info}] RAW Response body on HTTPError: {response_body_text[:1000]}...")
        # Try to parse error details from Google's JSON response even for HTTP errors
        parsed_body = response_body_text # Default to raw text
        error_message = f"{caller_info}: API request failed - HTTP {status_code}"
        error_type = "HTTPError"
        api_status = None
        try:
            parsed_error_json = json.loads(response_body_text)
            parsed_body = parsed_error_json
            error_details = parsed_error_json.get("error", {})
            if isinstance(error_details, dict):
                error_message = error_details.get("message", error_message)
                # Use API status if available (e.g., PERMISSION_DENIED)
                api_status = error_details.get('status')
                # Use specific error type if available from google error structure
                # error_type = error_details.get("status", error_type) # Example if status maps to type
            logger.debug(f"[{caller_info}] Parsed JSON from HTTPError body.")
        except json.JSONDecodeError:
            logger.warning(f"[{caller_info}] Failed decode JSON from HTTPError body (Status {status_code}). Using raw text.")
        return False, {"error_type": error_type, "message": error_message, "status_code": status_code, "api_status": api_status, "response_body": parsed_body, "source": "google_exec"}
    except requests.exceptions.RequestException as e:
        # Catch other request errors (DNS, ConnectionError, etc.)
        logger.error(f"[{caller_info}] RequestException: {e}", exc_info=False)
        return False, {"error_type": "RequestError", "message": f"{caller_info}: API request failed - {type(e).__name__}", "source": "google_exec"}
    except json.JSONDecodeError as e:
        # This case should be less likely now JSON parsing is inside try block, but keep as safeguard
        logger.error(f"[{caller_info}] Unexpected JSONDecodeError outside response handling: {e}", exc_info=True)
        response_body_snippet = ""
        # Need to ensure 'response' exists if we hit this edge case
        if 'response' in locals() and hasattr(response, 'text'):
             response_body_snippet = response.text[:500]
        return False, {"error_type": "JSONDecodeError", "message": f"{caller_info}: Failed to decode API response.", "details": response_body_snippet, "source": "google_exec"}
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(f"[{caller_info}] Unexpected error during Google API execution: {e}", exc_info=True)
        return False, {"error_type": "UnexpectedError", "message": f"{caller_info}: Unexpected error - {type(e).__name__}", "source": "google_exec"}


# --- OpenAI-Compatible API Client ---
def call_openai_compatible_api(
    api_url: str, # Base URL without fragment
    api_key: str,
    payload: Dict[str, Any], # OpenAI 'messages' format
    temperature: float, # Already set in payload by converter/dispatcher
    timeout: int = 90,
    caller_info: str = "LLM_OpenAI_Compat"
) -> Tuple[bool, Dict]: # Returns raw JSON dict on success OR error dict on failure
    """Calls an OpenAI-compatible API endpoint (like OpenRouter). Returns raw success dict."""
    logger.info(f"Attempting to call {caller_info} (OpenAI-Compatible Format) URL: {api_url[:80]}...")
    if not api_url or not api_key:
        error_msg = "Missing API Key" if not api_key else "Missing OpenAI-compatible URL"
        logger.error(f"[{caller_info}] {error_msg}.")
        # Note: Returning the raw dict, dispatcher handles normalization
        return False, {"error_type": "ConfigurationError", "message": error_msg, "source": "openai_client"}

    # Validate core payload elements expected by OpenAI spec
    if "model" not in payload or not payload["model"]:
        logger.error(f"[{caller_info}] OpenAI payload 'model' is missing or empty.")
        return False, {"error_type": "PayloadError", "message": "'model' name missing or empty in OpenAI payload", "source": "openai_client"}
    if "messages" not in payload or not isinstance(payload["messages"], list) or not payload["messages"]:
         logger.error(f"[{caller_info}] OpenAI payload 'messages' list missing/empty/invalid.")
         return False, {"error_type": "PayloadError", "message": "'messages' list missing/empty/invalid in OpenAI payload", "source": "openai_client"}

    # Set headers for OpenAI-compatible APIs (Bearer token)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        # Add specific headers required by providers like OpenRouter if necessary
        # "HTTP-Referer": $YOUR_SITE_URL, # Optional, for OpenRouter
        # "X-Title": $YOUR_APP_NAME      # Optional, for OpenRouter
    }

    # Ensure temperature is set (should be done by dispatcher, but double-check)
    payload["temperature"] = float(temperature)

    logger.debug(f"[{caller_info}] Calling OpenAI-compatible API: URL={api_url}, Model={payload.get('model')}")
    # Keep this uncommented for debugging potential payload issues
    logger.debug(f"[{caller_info}] OpenAI Payload: {json.dumps(payload)}")

    response = None # Initialize response variable
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        logger.debug(f"[{caller_info}] Response Status Code: {response.status_code}")
        response_text_snippet = response.text[:500] if response.text else "[EMPTY RESPONSE BODY]"
        logger.debug(f"[{caller_info}] Response Body Snippet: {response_text_snippet}...")

        # --- Check for HTTP Errors FIRST ---
        try:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        except requests.exceptions.HTTPError as e:
            # Re-raise to be caught by the outer HTTPError handler block below
            # This keeps HTTP error handling consistent
            raise e

        # --- Process 2xx Response (Success Status Code) ---
        # logger.debug(f"[{caller_info}] Raw OK response text: >>>{response.text}<<<") # Keep commented out
        try:
            result = response.json()
            logger.debug(f"[{caller_info}] Parsed JSON response successfully.")

            # [[[ MODIFICATION START ]]]
            # Check if the successful response body actually contains an error structure
            if "error" in result and isinstance(result.get("error"), dict):
                error_obj = result["error"]
                error_message = error_obj.get("message", "API returned error structure in 2xx response")
                error_type = error_obj.get("type", "APIErrorInSuccess")
                error_code = error_obj.get("code")
                logger.error(f"[{caller_info}] API returned 2xx status ({response.status_code}) but body contains error: Type='{error_type}', Code='{error_code}', Msg='{error_message}'")
                # Return False and the structured error dictionary
                # Add status_code from the response as it was technically successful HTTP-wise
                return False, {"error_type": error_type, "message": error_message, "status_code": response.status_code, "error_code": error_code, "response_body": result, "source": "openai_client"}
            # [[[ MODIFICATION END ]]]

            # If no "error" key found, proceed assuming it's a valid success response
            logger.info(f"[{caller_info}] Valid success response structure received.")
            # SUCCESS: Return True and the raw JSON dictionary
            return True, result

        except json.JSONDecodeError as e_json:
            logger.error(f"[{caller_info}] Failed JSON decode for successful response (Status {response.status_code}): {e_json}", exc_info=False)
            logger.error(f"[{caller_info}] RAW Body causing JSONDecodeError: >>>{response.text}<<<")
            # Return error dict, dispatcher will see success=False
            return False, {"error_type": "JSONDecodeError", "message": f"{caller_info}: Failed decode successful response.", "response_body": response.text, "status_code": response.status_code, "source": "openai_client"}
        except Exception as e_parse:
             # Catch other potential errors during JSON parsing/processing
             logger.error(f"[{caller_info}] Unexpected error processing successful response: {e_parse}", exc_info=True)
             return False, {"error_type": "ResponseProcessingError", "message": f"{caller_info}: Error processing successful response - {type(e_parse).__name__}", "response_body": response.text if response else "[No Response]", "status_code": response.status_code if response else None, "source": "openai_client"}

    # --- Exception Handling ---
    except requests.exceptions.Timeout as e:
        logger.error(f"[{caller_info}] Timeout after {timeout}s: {e}", exc_info=False)
        return False, {"error_type": "TimeoutError", "message": f"{caller_info}: API request timed out after {timeout}s", "source": "openai_client"}
    except requests.exceptions.HTTPError as e:
        # This block now handles only genuine 4xx/5xx errors raised by raise_for_status()
        status_code = e.response.status_code
        response_body_text = e.response.text
        logger.error(f"[{caller_info}] HTTPError {status_code}: {e}", exc_info=False)
        logger.error(f"[{caller_info}] RAW Response body on HTTPError: {response_body_text[:1000]}...")

        # Try to parse the error response body (often JSON for OpenAI-like APIs)
        parsed_body = response_body_text # Default to raw text
        error_message = f"{caller_info}: API request failed - HTTP {status_code}"
        error_type = "HTTPError" # Default type
        error_code = None # Specific code within the error object, if any
        try:
            parsed_error_json = json.loads(response_body_text)
            parsed_body = parsed_error_json
            # Standard OpenAI error structure: { "error": { "message": "...", "type": "...", "param": "...", "code": "..." } }
            error_obj = parsed_error_json.get("error", {})
            if isinstance(error_obj, dict):
                error_message = error_obj.get("message", error_message)
                error_type = error_obj.get("type", error_type) # e.g., "invalid_request_error"
                error_code = error_obj.get("code") # e.g., "invalid_api_key"
            logger.debug(f"[{caller_info}] Parsed JSON from HTTPError body.")
        except json.JSONDecodeError:
            logger.warning(f"[{caller_info}] Failed decode JSON from HTTPError body (Status {status_code}). Using raw text.")
        except Exception as e_parse_err:
            logger.warning(f"[{caller_info}] Error parsing JSON from HTTPError body: {e_parse_err}")

        return False, {"error_type": error_type, "message": error_message, "status_code": status_code, "error_code": error_code, "response_body": parsed_body, "source": "openai_client"}
    except requests.exceptions.RequestException as e:
        # Catch other request errors (DNS, ConnectionError, etc.)
        logger.error(f"[{caller_info}] RequestException: {e}", exc_info=False)
        return False, {"error_type": "RequestError", "message": f"{caller_info}: API request failed - {type(e).__name__}", "source": "openai_client"}
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(f"[{caller_info}] Unexpected error during OpenAI-compatible API call: {e}", exc_info=True)
        return False, {"error_type": "UnexpectedError", "message": f"{caller_info}: Unexpected error - {type(e).__name__}", "source": "openai_client"}


# --- API Dispatcher Function (Public Interface for Pipe) ---
# NOTE: This function is NAMED call_google_llm_api to match the pipe's import
def call_google_llm_api(
    api_url: str, # Expect URL potentially containing #model_name fragment
    api_key: str,
    payload: Dict[str, Any], # Expects Google 'contents' format from pipe
    temperature: float,
    timeout: int = 90,
    caller_info: str = "LLM_Dispatcher", # Info about the origin of the call within the pipe
) -> Tuple[bool, Union[str, Dict]]:
    """
    Dispatches LLM call based on API URL format and fragment.
    Acts as the main entry point called by the Session Memory Pipe.
    Normalizes responses to (bool, str_content | dict_error).

    Args:
        api_url (str): The API endpoint URL, potentially including a '#model/name' fragment
                       for OpenAI-compatible APIs.
        api_key (str): The API key.
        payload (Dict[str, Any]): The request payload, MUST be in Google's 'contents' format.
        temperature (float): The generation temperature.
        timeout (int): Request timeout in seconds.
        caller_info (str): Identifier for the calling process (e.g., 'Summarizer', 'RAGQuery').

    Returns:
        Tuple[bool, Union[str, Dict]]:
            (True, str): On success, returns the LLM response text.
            (False, Dict): On failure, returns a dictionary containing error details
                           ('error_type', 'message', 'status_code'?, 'response_body'?, 'source').
    """
    logger.info(f"[{caller_info}] Dispatching LLM call for raw URL: {api_url}")

    model_name: Optional[str] = None
    base_api_url: str = api_url
    is_google_api = False
    is_openai_api = False

    # --- Step 1: Parse URL and Determine API Type ---
    try:
        parsed_url = urllib.parse.urlparse(api_url)
        # The actual URL used for the request should NOT include the fragment
        base_api_url = urllib.parse.urlunparse(parsed_url._replace(fragment=""))
        # Use the base URL (lowercase, no trailing slash) for type checking
        url_for_check = base_api_url.lower().rstrip('/')

        # Check for keywords indicating API type
        # Prioritize specific known OpenAI endpoints first
        if "openrouter.ai/api/v1/chat/completions" in url_for_check or \
           url_for_check.endswith("/v1/chat/completions"): # Common pattern
            is_openai_api = True
            # If it's OpenAI type, model MUST be in fragment
            if parsed_url.fragment:
                model_name = parsed_url.fragment
                logger.info(f"[{caller_info}] OpenAI-compatible API detected. Extracted model_name='{model_name}' from fragment.")
            else:
                logger.error(f"[{caller_info}] OpenAI-compatible URL ('{base_api_url}') REQUIRES a model name fragment (e.g., '#model/name').")
                return False, {"error_type": "ConfigurationError", "message": "Model name fragment missing from OpenAI-compatible URL", "source": "dispatcher"}
        # Check for Google endpoint structure AFTER OpenAI checks
        elif "googleapis.com" in url_for_check and "models/" in url_for_check and ":generatecontent" in url_for_check:
            is_google_api = True
            logger.info(f"[{caller_info}] Google API detected.")
            # Warn if fragment exists on Google URL, as it's ignored
            if parsed_url.fragment:
                 logger.warning(f"[{caller_info}] Model fragment ('{parsed_url.fragment}') found on Google API URL ('{base_api_url}') - fragment will be ignored.")
        else:
            # Could not determine type - treat as error
            logger.error(f"[{caller_info}] Cannot determine API type from URL: '{base_api_url}'. Supported patterns: googleapis.com/...:generateContent, .../v1/chat/completions#model/name")
            return False, {"error_type": "ConfigurationError", "message": "Cannot determine API type from URL structure", "source": "dispatcher"}

    except Exception as e_url:
        logger.error(f"[{caller_info}] Error parsing API URL '{api_url}': {e_url}", exc_info=True)
        return False, {"error_type": "ConfigurationError", "message": f"Invalid API URL format: {e_url}", "source": "dispatcher"}

    # --- Step 2: Dispatch to Correct Client ---
    if is_google_api:
        logger.debug(f"[{caller_info}] Calling internal Google client for URL: {base_api_url}")
        # Pass the original Google payload and base URL to the *renamed* internal function
        # Payload is already in the correct format.
        success, result_or_error = _execute_google_llm_api(
            api_url=base_api_url, # URL without fragment or key
            api_key=api_key,
            payload=payload, # Google format
            temperature=temperature,
            timeout=timeout,
            caller_info=f"{caller_info}_Google" # More specific caller info
        )
        # _execute_google_llm_api already returns the desired (bool, str|dict) format
        return success, result_or_error

    elif is_openai_api:
        # We already checked and extracted model_name if we got here
        logger.debug(f"[{caller_info}] Calling OpenAI-compatible client for URL: {base_api_url} with Model: {model_name}")

        # Convert Google payload to OpenAI payload
        try:
            openai_payload = _convert_google_to_openai_payload(payload, model_name, temperature)
        except Exception as e_convert:
            logger.error(f"[{caller_info}] Failed payload conversion from Google to OpenAI: {e_convert}", exc_info=True)
            return False, {"error_type": "PayloadConversionError", "message": "Failed payload conversion for OpenAI", "details": str(e_convert), "source": "dispatcher"}

        # Call the OpenAI client with the CONVERTED payload and BASE URL
        raw_success, raw_response_or_error_dict = call_openai_compatible_api(
             api_url=base_api_url, # URL without fragment
             api_key=api_key,
             payload=openai_payload, # OpenAI format
             temperature=temperature, # Temp is inside openai_payload now
             timeout=timeout,
             caller_info=f"{caller_info}_OpenAICompat" # More specific caller info
        )

        # --- Step 3: Normalize OpenAI Response ---
        if raw_success:
            # raw_response_or_error_dict is the successful JSON dict from the API
            try:
                choices = raw_response_or_error_dict.get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    # Handle potential streaming chunk structure vs completion structure
                    first_choice = choices[0]
                    message = first_choice.get("message") # Standard completion
                    delta = first_choice.get("delta") # Streaming chunk

                    content = None
                    finish_reason = first_choice.get("finish_reason")

                    if message and isinstance(message, dict):
                        content = message.get("content")
                    elif delta and isinstance(delta, dict): # Handle streaming case if needed (though pipe likely expects full response)
                        content = delta.get("content")
                        logger.warning(f"[{caller_info}] Received potential streaming chunk (delta) from OpenAI API. Processing content.")

                    # Check if content was successfully extracted
                    if content is not None and isinstance(content, str):
                        logger.info(f"[{caller_info}] Successfully parsed response text from OpenAI-compatible API. Finish Reason: {finish_reason}")
                        # SUCCESS: Return True and the extracted text string
                        # Do not strip() here - let the pipe handle final formatting if needed
                        return True, content
                    else:
                        logger.error(f"[{caller_info}] OpenAI response missing/invalid 'content' in message/delta. Finish Reason: {finish_reason}")
                        return False, {"error_type": "ParsingError", "message": f"{caller_info}: OpenAI response missing/invalid 'content'", "finish_reason": finish_reason, "response_body": raw_response_or_error_dict, "source": "dispatcher"}
                else:
                    # Response has no 'choices' array or it's empty
                    # Keep the enhanced logging here from previous step
                    logger.error(f"[{caller_info}] OpenAI response missing/empty 'choices' array. Full Response: {json.dumps(raw_response_or_error_dict)}")
                    return False, {"error_type": "ParsingError", "message": f"{caller_info}: OpenAI response missing/empty 'choices'", "response_body": raw_response_or_error_dict, "source": "dispatcher"}
            except Exception as e_parse:
                 # Error during the parsing of the successful JSON response
                 logger.error(f"[{caller_info}] Error parsing successful OpenAI response structure: {e_parse}", exc_info=True)
                 return False, {"error_type": "ParsingError", "message": "Error parsing successful OpenAI response structure", "response_body": raw_response_or_error_dict, "source": "dispatcher"}
        else:
            # The call_openai_compatible_api function failed and returned an error dictionary
            # This now correctly includes the case where the API returned 2xx + error body
            logger.error(f"[{caller_info}] OpenAI-compatible API call failed. Returning error dict from client.")
            # raw_response_or_error_dict is already the error dictionary
            return False, raw_response_or_error_dict # FAILURE: Return the error dictionary

    else:
        # Should be unreachable if URL check logic is correct
        logger.critical(f"[{caller_info}] Dispatcher logic failed to identify API type after checks. This indicates an internal bug.")
        return False, {"error_type": "InternalError", "message": "Dispatcher logic failed to select API path", "source": "dispatcher"}

# [[END MODIFIED api_client.py]]