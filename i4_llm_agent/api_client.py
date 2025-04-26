# === START OF FILE i4_llm_agent/api_client.py ===
# i4_llm_agent/api_client.py

import logging
import urllib.parse # Needed for URL parsing
import json
import asyncio # Added for potential sleep/retry logic if needed later
import re # Included import for regular expressions
from typing import Tuple, Union, Optional, Dict, List, Any

# --- NEW: Import litellm ---
try:
    import litellm
    # Optional: Configure litellm settings, e.g., logging
    # litellm.set_verbose = True # Uncomment for detailed litellm logs
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False
    # Logging handled by Pipe startup


# Gets a logger named 'i4_llm_agent.api_client'
logger = logging.getLogger(__name__)

# --- Helper: Payload Conversion (Handles Missing Role) ---
def _convert_google_to_openai_payload(
    google_payload: Dict[str, Any], model_name: str, temperature: float
) -> Dict[str, Any]:
    """
    Converts Google 'contents' format to OpenAI 'messages' format.
    Handles system prompt extraction and ACK skipping.
    Handles single messages looking like system prompts.
    MODIFIED: Assumes 'user' role if 'role' key is missing but text is present.
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
                 logger.warning("Expected model ACK turn to skip, but found different role. Processing normally.")

        role = turn.get("role")
        parts = turn.get("parts", [])
        text = parts[0].get("text", "") if parts else ""

        # --- MODIFICATION START: Handle missing role ---
        if not role and isinstance(text, str) and text.strip():
            role = "user" # Assume user role if missing but text exists
            logger.debug(f"Turn {i} missing 'role', defaulting to '{role}' as text is present.")
        # --- MODIFICATION END ---

        # Basic validation after potential defaulting
        if not role or not isinstance(text, str):
            logger.warning(f"Skipping turn with invalid role ('{role}') or text content after potential defaulting: {turn}")
            continue

        # --- Process based on role (now guaranteed to exist if text is valid) ---
        if role == "user":
            lower_text_strip = text.strip().lower()
            is_system_indicator = (
                lower_text_strip.startswith("system instructions:") or
                lower_text_strip.startswith("system prompt:") or
                lower_text_strip.startswith("system directive:") or
                lower_text_strip.startswith("**system prompt") or
                lower_text_strip.startswith("**role:**")
            )

            # Check for system prompt extraction (only if multiple messages exist)
            if i == 0 and is_system_indicator and len(google_contents) > 1:
                 lines = text.split('\n', 1)
                 system_prompt_content = ""
                 if len(lines) > 1:
                     potential_content = lines[1].strip()
                     if potential_content:
                         system_prompt_content = potential_content
                     elif len(lines) > 2:
                         for line in lines[2:]:
                              potential_content = line.strip()
                              if potential_content:
                                   system_prompt_content = potential_content
                                   break

                 if system_prompt_content:
                     system_prompt = system_prompt_content
                     logger.debug("Extracted system prompt from first user message (multiple messages present).")

                     # Check for ACK skip
                     if i + 1 < len(google_contents):
                         next_turn = google_contents[i+1]
                         next_role = next_turn.get("role")
                         next_parts = next_turn.get("parts", [])
                         next_text = next_parts[0].get("text", "") if next_parts else ""
                         ack_texts = {"understood", "ok", "okay", "i understand", "acknowledged", "received", "noted"}
                         processed_next_text = next_text.strip().lower().rstrip('.!')
                         if next_role == "model" and processed_next_text in ack_texts:
                             skip_next_model_turn = True
                             logger.debug(f"Flagging model ACK ('{next_text}') for skipping after system prompt extraction.")
                     continue # Skip adding this turn
                 else:
                     logger.warning("Found system prompt indicator in first message, but content was empty. Treating as normal user message.")

            # Add regular user message
            openai_messages.append({"role": "user", "content": text})
            if i == 0 and is_system_indicator and len(google_contents) == 1:
                 logger.debug("Treating single input message that looked like system prompt as a regular user message.")

        elif role == "model":
            openai_messages.append({"role": "assistant", "content": text})
        else:
            # This should be less likely now, but good to keep
            logger.warning(f"Unknown role '{role}' encountered during conversion after defaulting. Skipping turn.")

    # Construct final payload
    openai_payload = {"messages": openai_messages}

    # Prepend system prompt if extracted
    if system_prompt and openai_payload["messages"]:
         if openai_payload["messages"][0].get("role") != "system":
             openai_payload["messages"].insert(0, {"role": "system", "content": system_prompt})
             logger.debug("Prepended extracted system prompt to OpenAI messages.")
         else:
              logger.warning("Attempted to prepend system prompt, but first message was already system role.")
    elif system_prompt and not openai_payload["messages"]:
         logger.warning("System prompt extracted, but message list ended up empty. Adding system prompt as only message.")
         openai_payload["messages"].append({"role": "system", "content": system_prompt})

    # Final check for empty messages (should not happen if input had content)
    if not openai_payload["messages"] and any(t.get("parts", [{}])[0].get("text", "").strip() for t in google_contents):
         logger.error("Conversion resulted in empty messages despite input having text content! Check logic.")
         # Optionally, could add the raw text back as a user message here as a last resort

    logger.debug(f"Payload conversion complete. OpenAI messages count: {len(openai_payload['messages'])}")
    return openai_payload


# --- Main API Dispatcher using LiteLLM (Public Interface for Pipe) ---
async def call_google_llm_api( # Keep original name for compatibility
    api_url: str, # Expect URL potentially containing #model_name fragment
    api_key: str,
    payload: Dict[str, Any], # Expects Google 'contents' format from pipe
    temperature: float,
    timeout: int = 90,
    caller_info: str = "LLM_Dispatcher", # Info about the origin of the call within the pipe
) -> Tuple[bool, Union[str, Dict]]:
    """
    Dispatches LLM call using LiteLLM based on API URL format and fragment.
    Acts as the main entry point called by the Session Memory Pipe.
    Accepts Google 'contents' format payload and converts it for LiteLLM.
    Normalizes responses to (bool, str_content | dict_error).
    REMOVED STREAMING SUPPORT.

    Args:
        api_url (str): The API endpoint URL, potentially including a '#model/name' fragment.
                       Examples:
                       - Google: "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
                       - OpenAI Native: "https://api.openai.com/v1/chat/completions#gpt-4o" (api_base derived)
                       - OpenRouter: "https://openrouter.ai/api/v1/chat/completions#mistralai/mistral-7b-instruct" (api_base derived)
                       - Other OAI-Compat: "http://localhost:1234/v1/chat/completions#local-model" (api_base derived)
        api_key (str): The API key.
        payload (Dict[str, Any]): The request payload, MUST be in Google's 'contents' format.
        temperature (float): The generation temperature.
        timeout (int): Request timeout in seconds for litellm.
        caller_info (str): Identifier for the calling process (e.g., 'Summarizer', 'RAGQuery').

    Returns:
        Tuple[bool, Union[str, Dict]]:
            (True, str): On success, returns the LLM response text.
            (False, Dict): On failure, returns a dictionary containing error details
                           ('error_type', 'message', 'status_code'?, 'response_body'?, 'source').
    """
    if not LITELLM_AVAILABLE:
        logger.critical(f"[{caller_info}] LiteLLM library is not available. Cannot make API calls.")
        return False, {"error_type": "SetupError", "message": "LiteLLM library not found or failed to import.", "source": "litellm_adapter"}

    logger.info(f"[{caller_info}] Dispatching LLM call via LiteLLM for raw URL: {api_url}")

    litellm_model: Optional[str] = None
    api_base: Optional[str] = None

    # --- Step 1: Parse URL and Determine LiteLLM Parameters ---
    try:
        parsed_url = urllib.parse.urlparse(api_url)
        base_api_url_str = urllib.parse.urlunparse(parsed_url._replace(fragment="", query=""))
        url_for_check = base_api_url_str.lower().rstrip('/')
        fragment = parsed_url.fragment

        if "googleapis.com" in url_for_check and "models/" in url_for_check and ":generatecontent" in url_for_check:
            logger.debug(f"[{caller_info}] Google URL check: Matched googleapis.com, /models/, :generatecontent")
            logger.debug(f"[{caller_info}] Google URL check: url_for_check = '{url_for_check}'")
            google_regex = r"/models/([a-zA-Z0-9.-]+):generatecontent"
            match = re.search(google_regex, url_for_check)
            logger.debug(f"[{caller_info}] Google URL check: Regex '{google_regex}' search result: {match}")

            if match:
                model_id = match.group(1)
                litellm_model = f"gemini/{model_id}"
                api_base = None
                logger.info(f"[{caller_info}] Google API detected. LiteLLM Model: '{litellm_model}'")
            else:
                 raise ValueError("Could not extract model name from Google API URL using regex.")
        elif "openrouter.ai/api/v1/chat/completions" in url_for_check:
             if fragment:
                 litellm_model = f"openrouter/{fragment}"
                 api_base = "https://openrouter.ai/api/v1"
                 logger.info(f"[{caller_info}] OpenRouter API detected. LiteLLM Model: '{litellm_model}', API Base: '{api_base}'")
             else:
                 raise ValueError("OpenRouter URL requires a model name fragment (e.g., '#mistralai/mistral-7b-instruct').")
        elif url_for_check.endswith("/v1/chat/completions"):
             if fragment:
                 litellm_model = fragment
                 api_base = base_api_url_str.removesuffix("/v1/chat/completions").rstrip('/')
                 if "api.openai.com" in url_for_check:
                      logger.info(f"[{caller_info}] OpenAI Native API detected. LiteLLM Model: '{litellm_model}', API Base: '{api_base}'")
                 else:
                      logger.info(f"[{caller_info}] Generic OpenAI-Compatible API detected. LiteLLM Model: '{litellm_model}', API Base: '{api_base}'")
             else:
                 raise ValueError("OpenAI-compatible URL requires a model name fragment (e.g., '#gpt-4o').")
        else:
            raise ValueError(f"Cannot determine API type or model from URL structure: '{api_url}'. See examples in docstring.")

    except ValueError as ve:
         logger.error(f"[{caller_info}] Error parsing API URL '{api_url}': {ve}")
         return False, {"error_type": "ConfigurationError", "message": str(ve), "source": "litellm_adapter"}
    except Exception as e_url:
        logger.error(f"[{caller_info}] Unexpected error parsing API URL '{api_url}': {e_url}", exc_info=True)
        return False, {"error_type": "ConfigurationError", "message": f"Unexpected error parsing API URL: {e_url}", "source": "litellm_adapter"}

    if not litellm_model:
         logger.error(f"[{caller_info}] Failed to determine LiteLLM model string from URL '{api_url}'.")
         return False, {"error_type": "ConfigurationError", "message": "Failed to determine LiteLLM model string.", "source": "litellm_adapter"}


    # --- Step 2: Convert Google Payload to OpenAI Messages ---
    openai_messages: List[Dict] = []
    try:
        converted_payload_dict = _convert_google_to_openai_payload(payload, litellm_model, temperature)
        openai_messages = converted_payload_dict.get("messages", [])
        if not openai_messages:
             logger.error(f"[{caller_info}] Payload conversion resulted in empty messages list. Original Google payload: {json.dumps(payload)}") # Log original payload
             raise ValueError("Payload conversion resulted in empty messages list.")
        logger.debug(f"[{caller_info}] Successfully converted Google payload to {len(openai_messages)} OpenAI messages.")
    except Exception as e_convert:
        logger.error(f"[{caller_info}] Failed payload conversion from Google to OpenAI format: {e_convert}", exc_info=True)
        return False, {"error_type": "PayloadConversionError", "message": f"Failed payload conversion: {e_convert}", "source": "litellm_adapter"}

    # --- Step 3: Call LiteLLM ---
    logger.info(f"[{caller_info}] Calling litellm.acompletion with Model='{litellm_model}', Temp={temperature}, Timeout={timeout}, API Base='{api_base or 'Default'}'")
    try:
        call_temp = float(temperature)
        litellm_kwargs = {
            "model": litellm_model,
            "messages": openai_messages,
            "api_key": api_key,
            "temperature": call_temp,
            "request_timeout": timeout,
        }
        if api_base:
            litellm_kwargs["api_base"] = api_base

        response = await litellm.acompletion(**litellm_kwargs)

        # --- Step 4: Process LiteLLM Response ---
        if not response:
            raise ValueError("LiteLLM returned an empty response.")

        content_text = None
        finish_reason = None
        try:
            if hasattr(response, 'choices') and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, 'message') and first_choice.message:
                     content_text = getattr(first_choice.message, 'content', None)
                finish_reason = getattr(first_choice, 'finish_reason', None)
            elif isinstance(response, dict):
                 choices = response.get("choices", [])
                 if choices and isinstance(choices, list) and len(choices) > 0:
                      first_choice = choices[0]
                      message = first_choice.get("message", {})
                      content_text = message.get("content")
                      finish_reason = first_choice.get("finish_reason")

        except (AttributeError, KeyError, IndexError, TypeError) as e_parse:
            logger.error(f"[{caller_info}] Error parsing LiteLLM response structure: {e_parse}", exc_info=True)
            logger.debug(f"[{caller_info}] Raw LiteLLM Response: {response}")
            return False, {"error_type": "ParsingError", "message": "Error parsing successful LiteLLM response structure", "response_body": str(response), "source": "litellm_adapter"}

        if content_text is not None and isinstance(content_text, str):
            logger.info(f"[{caller_info}] LiteLLM call successful. Finish Reason: {finish_reason}. Returning content.")
            return True, content_text
        else:
            logger.error(f"[{caller_info}] LiteLLM response missing/invalid 'content'. Finish Reason: {finish_reason}")
            return False, {"error_type": "ParsingError", "message": f"{caller_info}: LiteLLM response missing/invalid 'content'", "finish_reason": finish_reason, "response_body": str(response), "source": "litellm_adapter"}

    # --- Step 5: Handle LiteLLM Exceptions ---
    except litellm.exceptions.AuthenticationError as e:
        logger.error(f"[{caller_info}] LiteLLM AuthenticationError: {e}", exc_info=False)
        return False, {"error_type": "AuthenticationError", "message": str(e), "status_code": getattr(e, 'status_code', 401), "response_body": getattr(e, 'response', None), "source": "litellm_adapter"}
    except litellm.exceptions.RateLimitError as e:
        logger.error(f"[{caller_info}] LiteLLM RateLimitError: {e}", exc_info=False)
        return False, {"error_type": "RateLimitError", "message": str(e), "status_code": getattr(e, 'status_code', 429), "response_body": getattr(e, 'response', None), "source": "litellm_adapter"}
    except litellm.exceptions.APIConnectionError as e:
        logger.error(f"[{caller_info}] LiteLLM APIConnectionError: {e}", exc_info=False)
        return False, {"error_type": "APIConnectionError", "message": str(e), "status_code": getattr(e, 'status_code', 500), "response_body": getattr(e, 'response', None), "source": "litellm_adapter"}
    except litellm.exceptions.Timeout as e:
        logger.error(f"[{caller_info}] LiteLLM Timeout after {timeout}s: {e}", exc_info=False)
        return False, {"error_type": "TimeoutError", "message": f"LiteLLM request timed out after {timeout}s: {e}", "source": "litellm_adapter"}
    except litellm.exceptions.BadRequestError as e:
        logger.error(f"[{caller_info}] LiteLLM BadRequestError: {e}", exc_info=False)
        return False, {"error_type": "BadRequestError", "message": str(e), "status_code": getattr(e, 'status_code', 400), "response_body": getattr(e, 'response', None), "source": "litellm_adapter"}
    except litellm.exceptions.APIError as e:
        logger.error(f"[{caller_info}] LiteLLM APIError: {e} (Status: {getattr(e, 'status_code', 'N/A')})", exc_info=False)
        return False, {"error_type": "APIError", "message": str(e), "status_code": getattr(e, 'status_code', None), "response_body": getattr(e, 'response', None), "source": "litellm_adapter"}
    except Exception as e:
        logger.error(f"[{caller_info}] Unexpected error during LiteLLM execution: {e}", exc_info=True)
        return False, {"error_type": "UnexpectedError", "message": f"{caller_info}: Unexpected error - {type(e).__name__}: {e}", "source": "litellm_adapter"}

# === END OF FILE i4_llm_agent/api_client.py ===