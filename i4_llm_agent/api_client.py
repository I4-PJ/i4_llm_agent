# i4_llm_agent/api_client.py

import requests
import logging
import urllib.parse
from typing import Optional, Dict, List, Any

# Gets a logger named 'i4_llm_agent.api_client'
logger = logging.getLogger(__name__)

def call_google_llm_api(
    api_url: str,
    api_key: str,
    payload: Dict[str, Any],
    temperature: float, # Include temperature in payload generation
    timeout: int = 90,
    caller_info: str = "LLM" # e.g., "Refiner LLM" or "Final LLM" for logs
) -> Optional[str]:
    """
    Calls a Google Generative Language API endpoint (standard format).

    Args:
        api_url: The base API endpoint URL (e.g., .../gemini-1.5-flash-latest:generateContent).
        api_key: The API key.
        payload: The request payload dictionary (expects 'contents' key).
        temperature: The temperature setting for the generationConfig.
        timeout: Request timeout in seconds.
        caller_info: String identifier for logging purposes.

    Returns:
        The extracted text content from the API response, or an error string starting with '[Error:'.
        Returns None only if essential config (URL/Key) is missing before call attempt.
    """
    logger.info(f"Attempting to call {caller_info} (Google Standard Format)...")

    if not api_url or not api_key:
        logger.error(f"Missing {caller_info} configuration (URL or Key).")
        # Return None or a specific error if config is missing *before* the call
        # Based on original code, key missing returns specific error:
        if not api_key:
             return "[Error: API Key is missing]"
        return "[Error: Missing LLM configuration]"

    # Ensure generationConfig is part of the payload passed in or add it
    if "generationConfig" not in payload:
         payload["generationConfig"] = {"temperature": temperature}
    elif "temperature" not in payload["generationConfig"]:
         payload["generationConfig"]["temperature"] = temperature

    # Add API key to URL
    final_api_url_with_key = f"{api_url}?key={urllib.parse.quote(api_key)}"
    headers = {"Content-Type": "application/json"}

    logger.debug(f"Calling {caller_info} API: URL={final_api_url_with_key}")
    # Avoid logging potentially sensitive payloads in production unless necessary
    # logger.debug(f"Payload for {caller_info} (First 500 chars): {str(payload)[:500]}...")

    response_text = None
    try:
        response = requests.post(
            final_api_url_with_key, json=payload, headers=headers, timeout=timeout
        )
        logger.debug(f"{caller_info} API Response Status Code: {response.status_code}")
        response_text_snippet = response.text[:500] # Log only a snippet
        logger.debug(f"{caller_info} API Response Body Snippet: {response_text_snippet}...")

        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        result = response.json()
        # Avoid logging the full raw response unless debugging
        # logger.debug(f"Raw {caller_info} API JSON Response (First 500 chars): {str(result)[:500]}...")

        # Parse Response (Standard Google Gemini format)
        candidates = result.get("candidates")
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            first_candidate = candidates[0]
            content = first_candidate.get("content")
            if content and content.get("parts") and isinstance(content["parts"], list) and len(content["parts"]) > 0:
                text_part = content["parts"][0].get("text")
                if text_part is not None:
                    response_text = text_part.strip()
                    logger.info(f"Successfully parsed response from {caller_info} (Gemini format).")
                    # Avoid logging full parsed text unless debugging
                    # logger.debug(f"{caller_info} Parsed Text (Full): {response_text}")
                else:
                    logger.error(f"{caller_info} response 'text' field is null. Response Snippet: {response_text_snippet}")
                    response_text = f"[Error: {caller_info} response text is null]"
            else:
                # Check for safety blocks or other reasons
                finish_reason = first_candidate.get("finishReason")
                safety_ratings = first_candidate.get("safetyRatings")
                if finish_reason == "SAFETY":
                    logger.warning(f"{caller_info} content blocked by safety filters: {safety_ratings}")
                    response_text = f"[Error: {caller_info} content blocked by safety filters]"
                elif finish_reason == "RECITATION":
                     logger.warning(f"{caller_info} content blocked by recitation filters.")
                     response_text = f"[Error: {caller_info} content blocked by recitation filters]"
                elif finish_reason == "OTHER":
                     logger.warning(f"{caller_info} content blocked for other reasons.")
                     response_text = f"[Error: {caller_info} content blocked for other reasons]"
                else:
                    logger.error(f"{caller_info} response missing 'content' or 'parts'. Reason: {finish_reason}. Response Snippet: {response_text_snippet}")
                    response_text = f"[Error: Unexpected {caller_info} API response structure (content/parts)]"
        else:
            # Check for explicit API errors
            error_details = result.get("error")
            if error_details:
                logger.error(f"{caller_info} Google API error object: {error_details}")
                response_text = f"[Error: {caller_info} API Error - {error_details.get('message', 'Unknown')}]"
            # Handle cases like empty candidate list without explicit error
            elif "promptFeedback" in result and result["promptFeedback"].get("blockReason") == "SAFETY":
                 logger.warning(f"{caller_info} prompt blocked by safety filters: {result.get('promptFeedback')}")
                 response_text = f"[Error: {caller_info} prompt blocked by safety filters]"
            else:
                logger.error(f"Unexpected response from {caller_info} (no candidates/error field). Response Snippet: {response_text_snippet}")
                response_text = f"[Error: Unexpected {caller_info} API response structure (candidates)]"

    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during {caller_info} API call: {e}", exc_info=False) # Keep logs cleaner
        response_text = f"[Error: {caller_info} API request timed out]"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during {caller_info} API call: {e}", exc_info=False) # Keep logs cleaner
        # Log response body if available, even on error
        if "response" in locals() and hasattr(response, "text"):
             logger.error(f"Response body on error: {response.text[:1000]}...") # Log more on actual error
        response_text = f"[Error: {caller_info} API request failed - {type(e).__name__}]"
    except Exception as e:
        # Catch JSONDecodeError, unexpected errors during parsing
        logger.error(f"Error processing {caller_info} API response: {e}", exc_info=True) # Log full traceback here
        response_text = f"[Error: Failed to process {caller_info} API response - {type(e).__name__}]"

    # Ensure we always return a string or None (as per original signature, though error strings preferred)
    return response_text if response_text is not None else f"[Error: Unknown issue processing {caller_info} response]"
