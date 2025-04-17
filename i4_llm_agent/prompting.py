# i4_llm_agent/prompting.py

import logging
import re
import asyncio
# <<< Adjusted Import Order >>>
from typing import Tuple, Union, Optional, Dict, List, Any, Callable, Coroutine

# --- Existing Imports from i4_llm_agent ---
from .history import get_recent_turns, format_history_for_llm, DIALOGUE_ROLES
# <<< Assuming llm_call_func signature/import from pipe's perspective >>>
# This function expects the async wrapper from the pipe context
# Example signature:
# async def llm_call_wrapper(api_url, api_key, payload, temperature, timeout, caller_info) -> Tuple[bool, Union[str, Dict]]: pass

logger = logging.getLogger(__name__) # 'i4_llm_agent.prompting'

# --- Constants for Context Tags (Existing) ---
KNOWN_CONTEXT_TAGS = {
    "owi": ("<context>", "</context>"),
    "t1": ("<mempipe_recent_summary>", "</mempipe_recent_summary>"),
    "t2_rag": ("<mempipe_rag_result>", "</mempipe_rag_result>"),
}
TAG_LABELS = {
    "owi": "OWI Context",
    "t1": "Recent Summaries (T1)",
    "t2_rag": "Related Summaries (T2 RAG)",
}
EMPTY_CONTEXT_PLACEHOLDER = "[No Background Information Available]"

# --- Constants for Refiner ---
# <<< Define Refiner constants here >>>
REFINER_QUERY_PLACEHOLDER = "[Insert Latest User Query Here]"
REFINER_CONTEXT_PLACEHOLDER = "[Insert Retrieved Documents Here]"
REFINER_HISTORY_PLACEHOLDER = "[Insert Recent Chat History Here]"

# <<< Default Refiner Template (Adapted from i4_rag) >>>
DEFAULT_REFINER_PROMPT_TEMPLATE = f"""
**Role:** Roleplay Context Extractor
**Task:** Analyze the provided CONTEXT DOCUMENTS (character backstories, relationship histories, past events, lore) and the RECENT CHAT HISTORY (dialogue, actions, emotional expressions).
**Objective:** Based ONLY on this information, extract and describe the specific details, memories, relationship dynamics, stated feelings, significant past events, or relevant character traits that are **essential for understanding the full context** of and accurately answering the LATEST USER QUERY from a roleplaying perspective.
**Instructions:**
1.  **Identify the core subject** of the LATEST USER QUERY and any immediately related contextual elements.
2.  **Extract Key Information:** Prioritize extracting verbatim sentences or short passages that **directly address** the core subject and related elements.
3.  **Describe Key Dynamics:** For relationship queries, don't just state the status (e.g., \"complex\"); **extract specific details or events from the context that illustrate *why* it's complex** (e.g., foundational events, major conflicts, stated feelings, defining interactions).
4.  **Include Foundational Context:** Extract specific details about significant past events or character history mentioned in the context that **directly led to or fundamentally define** the current situation or relationship relevant to the query.
5.  **Incorporate Recent Developments:** Include details from the RECENT CHAT HISTORY that show the *current* state or recent evolution of the situation or relationship.
6.  **Be Descriptive but Focused:** Capture the nuance and specific details present in the source material relevant to the query. Avoid overly generic summaries, especially regarding character relationships and motivations.
7.  **Prioritize Relevance over Extreme Brevity:** While redundancy should be removed, ensure that key descriptive details illustrating relationships, motivations, or foundational events are included, even if it makes the summary slightly longer. Focus conciseness on removing truly irrelevant information.
8.  **Ensure Accuracy:** Do not infer, assume, or add information not explicitly present in the provided CONTEXT DOCUMENTS or RECENT CHAT HISTORY.
9.  **Output:** Present the extracted points clearly. If no relevant information is found, state clearly: \"No specific details relevant to the query were found in the provided context.\"

**LATEST USER QUERY:** {REFINER_QUERY_PLACEHOLDER}
**CONTEXT DOCUMENTS:**
---
{REFINER_CONTEXT_PLACEHOLDER}
---
**RECENT CHAT HISTORY:**
---
{REFINER_HISTORY_PLACEHOLDER}
---

Concise Relevant Information (for final answer generation):
"""


# --- Function: Clean Context Tags (Existing) ---
def clean_context_tags(system_content: str) -> str:
    if not system_content or not isinstance(system_content, str): return ""
    cleaned = system_content
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = r"\s*" + re.escape(start_tag) + r".*?" + re.escape(end_tag) + r"\s*"
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# --- Function: Process System Prompt (Existing) ---
# <<< Return Type Hint uses Tuple >>>
def process_system_prompt(messages: List[Dict]) -> Tuple[str, Optional[str]]:
    """
    Finds the first system message, extracts OWI context tag content,
    and returns the cleaned base prompt text and the extracted context.
    """
    func_logger = logging.getLogger(__name__ + '.process_system_prompt')
    original_system_prompt_content = ""
    extracted_owi_context = None
    base_system_prompt_text = "You are a helpful assistant."

    if not isinstance(messages, list):
         func_logger.warning("Input 'messages' is not a list. Cannot process system prompt.")
         return base_system_prompt_text, None

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            original_system_prompt_content = msg.get("content", "")
            func_logger.debug(f"Found system prompt content (length {len(original_system_prompt_content)}).")
            break

    if not original_system_prompt_content:
        func_logger.debug("No system message found in input.")
        return base_system_prompt_text, None

    owi_match = re.search(r"<context>(.*?)</context>", original_system_prompt_content, re.DOTALL | re.IGNORECASE)
    if owi_match:
        extracted_owi_context = owi_match.group(1).strip()
        func_logger.debug(f"Extracted OWI context tag content (length {len(extracted_owi_context)}).")

    cleaned_base_prompt = clean_context_tags(original_system_prompt_content)
    if cleaned_base_prompt:
        base_system_prompt_text = cleaned_base_prompt
    else:
        func_logger.warning("System prompt content was empty after cleaning known tags. Using default.")

    return base_system_prompt_text, extracted_owi_context

# --- Function: Format Refiner Prompt ---
# <<< NEW FUNCTION (Adapted from i4_rag) >>>
def format_refiner_prompt(
    context: str,
    recent_history_str: str,
    query: str,
    template: Optional[str] = None # Template is now an argument
) -> str:
    """
    Constructs the prompt for the external refiner LLM using the provided template.

    Args:
        context: The RAG context string.
        recent_history_str: The formatted recent chat history string.
        query: The latest user query string.
        template: The prompt template string containing placeholders.
                  Uses a default if None is provided.

    Returns:
        The fully formatted prompt string ready for the Refiner LLM.
        Returns a basic fallback prompt if template formatting fails.
    """
    func_logger = logging.getLogger(__name__ + '.format_refiner_prompt')
    prompt_template = template if template is not None else DEFAULT_REFINER_PROMPT_TEMPLATE
    func_logger.debug(f"Using refiner prompt template (length {len(prompt_template)}).")

    # Basic sanitization (prevent breaking the prompt structure)
    safe_context = context.replace("---", "===") if isinstance(context, str) else ""
    safe_history = recent_history_str.replace("---", "===") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("---", "===") if isinstance(query, str) else ""

    # Replace placeholders
    try:
        # Using .replace() as the default template uses [VAR] style placeholders indirectly
        formatted_prompt = prompt_template
        if REFINER_CONTEXT_PLACEHOLDER in formatted_prompt:
             formatted_prompt = formatted_prompt.replace(REFINER_CONTEXT_PLACEHOLDER, safe_context)
        if REFINER_HISTORY_PLACEHOLDER in formatted_prompt:
             formatted_prompt = formatted_prompt.replace(REFINER_HISTORY_PLACEHOLDER, safe_history)
        if REFINER_QUERY_PLACEHOLDER in formatted_prompt:
             formatted_prompt = formatted_prompt.replace(REFINER_QUERY_PLACEHOLDER, safe_query)

        func_logger.debug(f"Formatted Refiner Prompt (Snippet): {formatted_prompt[:500]}...")
        return formatted_prompt
    except KeyError as e:
         func_logger.error(f"Missing placeholder in refiner prompt template: {e}. Template snippet: {prompt_template[:300]}...", exc_info=True)
    except Exception as e:
        func_logger.error(f"Error formatting refiner prompt template: {e}", exc_info=True)

    # Fallback to a very basic prompt if formatting fails
    func_logger.warning("Falling back to basic refiner prompt due to formatting error.")
    fallback_prompt = f"Context:\n{safe_context}\n\nHistory:\n{safe_history}\n\nQuery:\n{safe_query}\n\nSummarize relevant info:"
    return fallback_prompt


# --- Function: Refine External Context ---
# <<< NEW ASYNC FUNCTION >>>
async def refine_external_context(
    external_context: str,
    history_messages: List[Dict],
    latest_user_query: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # Expects async wrapper
    refiner_llm_config: Dict[str, Any], # url, key, temp, prompt_template
    skip_threshold: int,
    history_count: int,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES,
    caller_info: str = "i4_llm_agent_Refiner"
) -> str:
    """
    Optionally refines the provided external RAG context using an LLM.

    Args:
        external_context: The context string extracted (e.g., from OWI's <context> tag).
        history_messages: The full list of message history (for extracting recent turns).
        latest_user_query: The user's last message content.
        llm_call_func: The asynchronous wrapper function provided by the pipe to call LLMs.
                       Expected signature: async def func(api_url, api_key, payload, temperature, timeout, caller_info) -> Tuple[bool, Union[str, Dict]]
        refiner_llm_config: Dictionary with refiner LLM parameters:
                            'url', 'key', 'temp', 'prompt_template'.
        skip_threshold: Skip refinement if context length is below this value (chars).
        history_count: Number of recent dialogue turns to include in the refiner prompt.
        dialogue_only_roles: Roles to consider for recent history extraction.
        caller_info: Identifier string for logging.

    Returns:
        The refined context string, or the original external_context if skipped or failed.
    """
    func_logger = logging.getLogger(__name__ + '.refine_external_context')
    func_logger.debug(f"[{caller_info}] Entered refine_external_context.")

    # --- Input Validation ---
    if not external_context or not external_context.strip():
        func_logger.debug(f"[{caller_info}] Skipping refinement: Input context is empty.")
        return external_context # Return original (empty) context

    if not latest_user_query or not latest_user_query.strip():
        func_logger.warning(f"[{caller_info}] Skipping refinement: Latest user query is empty.")
        return external_context # Cannot refine without query

    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        func_logger.error(f"[{caller_info}] Skipping refinement: Invalid async llm_call_func provided.")
        return external_context # Cannot call LLM

    required_keys = ['url', 'key', 'temp', 'prompt_template']
    if not refiner_llm_config or not all(k in refiner_llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in refiner_llm_config]
        func_logger.error(f"[{caller_info}] Skipping refinement: Missing refiner LLM config keys: {missing}")
        return external_context # Cannot call LLM without config

    if not refiner_llm_config['url'] or not refiner_llm_config['key']:
        func_logger.error(f"[{caller_info}] Skipping refinement: Missing refiner LLM URL or Key.")
        return external_context # Cannot call LLM

    # --- Skip Threshold Check ---
    context_length = len(external_context)
    if skip_threshold > 0 and context_length < skip_threshold:
        func_logger.info(f"[{caller_info}] Skipping refinement: Context length ({context_length}) < Threshold ({skip_threshold}).")
        return external_context # Return original context if below threshold

    func_logger.info(f"[{caller_info}] Proceeding with refinement (Context length {context_length}).")

    # --- Prepare Refiner Inputs ---
    # 1. Get Recent History String
    recent_history_list = get_recent_turns(
        messages=history_messages,
        count=history_count,
        roles=dialogue_only_roles,
        exclude_last=True # Exclude the query itself from history context
    )
    recent_chat_history_str = format_history_for_llm(recent_history_list) if recent_history_list else "[No Recent History]"

    # 2. Format Refiner Prompt
    refiner_prompt_text = format_refiner_prompt(
        context=external_context,
        recent_history_str=recent_chat_history_str,
        query=latest_user_query,
        template=refiner_llm_config.get('prompt_template') # Pass template from config
    )

    if not refiner_prompt_text or refiner_prompt_text.startswith("[Error:") or "[Fallback:" in refiner_prompt_text:
         func_logger.error(f"[{caller_info}] Failed to format refiner prompt: {refiner_prompt_text}. Aborting refinement.")
         return external_context # Return original if prompt formatting failed

    # 3. Prepare Payload
    refiner_payload = {"contents": [{"parts": [{"text": refiner_prompt_text}]}]}

    # --- Call Refiner LLM via Wrapper ---
    func_logger.info(f"[{caller_info}] Calling Refiner LLM...")
    try:
        success, response_or_error = await llm_call_func(
            api_url=refiner_llm_config['url'],
            api_key=refiner_llm_config['key'],
            payload=refiner_payload,
            temperature=refiner_llm_config['temp'],
            timeout=90, # Standard timeout for refinement
            caller_info=caller_info,
        )
    except Exception as e_call:
         func_logger.error(f"[{caller_info}] Exception during llm_call_func: {e_call}", exc_info=True)
         success = False
         response_or_error = {"error_type": "CallWrapperException", "message": f"Exception calling LLM wrapper: {type(e_call).__name__}"}

    # --- Process Result ---
    if success and isinstance(response_or_error, str) and response_or_error.strip():
        refined_context = response_or_error.strip()
        func_logger.info(f"[{caller_info}] Refinement successful (Refined length: {len(refined_context)}).")
        # DEBUG log comparison
        func_logger.debug(f"[{caller_info}] Original Context Snippet: {external_context[:200]}...")
        func_logger.debug(f"[{caller_info}] Refined Context Snippet: {refined_context[:200]}...")
        return refined_context
    else:
        # Log the failure details
        error_details = str(response_or_error)
        if isinstance(response_or_error, dict):
            error_details = f"Type: {response_or_error.get('error_type', 'Unknown')}, Msg: {response_or_error.get('message', 'N/A')}"

        func_logger.warning(f"[{caller_info}] Refinement failed or returned empty. Error: '{error_details}'. Returning original context.")
        return external_context # Return original context on failure


# --- Function: Generate RAG Query (Existing) ---
async def generate_rag_query(
    latest_message_str: str,
    dialogue_context_str: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # Wrapper returns tuple
    llm_config: Dict[str, Any],
    caller_info: str = "i4_llm_agent_RAGQueryGen",
) -> Optional[str]: # Returns string or None/Error string
    logger.debug(f"[{caller_info}] Generating RAG query...")
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        logger.error(f"[{caller_info}] Invalid llm_call_func."); return "[Error: Invalid LLM call func]"
    required_keys = ['url', 'key', 'temp', 'prompt']
    if not llm_config or not all(k in llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in llm_config]; logger.error(f"[{caller_info}] Missing LLM config keys: {missing}"); return f"[Error: Missing RAGQ config: {missing}]"
    if not llm_config['url'] or not llm_config['key']: logger.error(f"[{caller_info}] Missing RAGQ URL/Key."); return "[Error: Missing RAGQ URL/Key]"
    if not llm_config['prompt'] or not isinstance(llm_config['prompt'], str): logger.error(f"[{caller_info}] Missing/Invalid RAGQ prompt."); return "[Error: Missing/Invalid RAGQ prompt]"
    ragq_prompt_text = "[Error: Prompt Formatting Failed]"
    try:
        template = llm_config['prompt']
        if "{latest_message}" not in template or "{dialogue_context}" not in template: logger.warning(f"[{caller_info}] RAGQ prompt missing placeholders.")
        ragq_prompt_text = template.format(latest_message=latest_message_str or "[No user message]", dialogue_context=dialogue_context_str or "[No dialogue context]")
    except KeyError as e: logger.error(f"[{caller_info}] RAGQ prompt key error: {e}."); return f"[Error: RAGQ prompt key error: {e}]"
    except Exception as e_fmt: logger.error(f"[{caller_info}] Failed format RAGQ prompt: {e_fmt}", exc_info=True); return f"[Error: RAGQ prompt format failed - {type(e_fmt).__name__}]"
    ragq_payload = {"contents": [{"parts": [{"text": ragq_prompt_text}]}]}
    logger.info(f"[{caller_info}] Calling LLM for RAG query generation...")
    # Call wrapper which returns Tuple[bool, Union[str, Dict]]
    success, response_or_error = await llm_call_func(
        api_url=llm_config['url'], api_key=llm_config['key'], payload=ragq_payload,
        temperature=llm_config['temp'], timeout=45, caller_info=caller_info,
    )
    # Process the result based on the tuple
    if success and isinstance(response_or_error, str):
        final_query = response_or_error.strip()
        if final_query: logger.info(f"[{caller_info}] Generated RAG query: '{final_query}'"); return final_query
        else: logger.warning(f"[{caller_info}] RAGQ LLM returned empty string."); return "[Error: RAGQ generation returned empty]"
    else: # Failure or unexpected type
        error_msg = str(response_or_error) if not success else f"Unexpected success type: {type(response_or_error)}"
        logger.error(f"[{caller_info}] RAGQ failed: {error_msg}");
        # Return a consistent error string format if possible
        if isinstance(response_or_error, dict):
            return f"[Error: {response_or_error.get('error_type', 'RAGQ Error')} - {response_or_error.get('message', 'Unknown')}]"
        else:
            return f"[Error: RAGQ Failed - {error_msg[:100]}]"


# --- Function: Combine Background Context (Existing) ---
def combine_background_context(
    owi_context: Optional[str] = None,
    t1_summaries: Optional[List[str]] = None,
    t2_rag_results: Optional[List[str]] = None,
    separator: str = "\n---\n",
    block_separator: str = "\n\n",
) -> str:
    context_parts = []
    if owi_context and isinstance(owi_context, str) and owi_context.strip():
        context_parts.append(f"--- Original OWI Context ---\n{owi_context.strip()}") # Label might change if refined?
    if t1_summaries and isinstance(t1_summaries, list):
        valid_t1 = [s for s in t1_summaries if isinstance(s, str) and s.strip()]
        if valid_t1: context_parts.append(f"--- Recent Summaries (T1) ---\n{separator.join(valid_t1)}")
    if t2_rag_results and isinstance(t2_rag_results, list):
        valid_t2 = [s for s in t2_rag_results if isinstance(s, str) and s.strip()]
        if valid_t2: context_parts.append(f"--- Related Summaries (T2 RAG) ---\n{separator.join(valid_t2)}")
    if not context_parts: return EMPTY_CONTEXT_PLACEHOLDER
    combined_context_string = block_separator.join(context_parts).strip()
    return combined_context_string

# --- Function: Construct Final LLM Payload (Existing) ---
def construct_final_llm_payload(
    system_prompt: str,
    history: List[Dict],
    context: Optional[str],
    query: str,
    strategy: str = 'standard',
    include_ack_turns: bool = True
) -> Dict[str, Any]:
    logger.debug(f"Constructing final LLM payload. Strategy: {strategy}, ACKs: {include_ack_turns}")
    gemini_contents = []
    system_prompt_text = system_prompt or "You are a helpful assistant."
    gemini_contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt_text}"}]})
    if include_ack_turns: gemini_contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})
    history_turns = []
    for msg in history:
        role = msg.get("role"); content = msg.get("content", "").strip()
        if role == "user" and content: history_turns.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant" and content: history_turns.append({"role": "model", "parts": [{"text": content}]})
    context_turn = None; ack_turn = None
    has_real_context = bool(context and context.strip() and context.strip() != EMPTY_CONTEXT_PLACEHOLDER)
    if has_real_context:
        context_injection_text = f"Background Information (Use this to inform your response):\n{context}"
        context_turn = {"role": "user", "parts": [{"text": context_injection_text}]}
        if include_ack_turns: ack_turn = {"role": "model", "parts": [{"text": "Understood. I have reviewed the background information."}]}
    safe_query = query.strip().replace("---", "===") if query and query.strip() else "[User query not provided]"
    final_query_turn = {"role": "user", "parts": [{"text": safe_query}]}
    if strategy == 'advanced': # Sys -> [Ctx] -> Hist -> Query
        if context_turn: gemini_contents.append(context_turn)
        if ack_turn: gemini_contents.append(ack_turn)
        gemini_contents.extend(history_turns); gemini_contents.append(final_query_turn)
    elif strategy == 'standard': # Sys -> Hist -> [Ctx] -> Query
        gemini_contents.extend(history_turns)
        if context_turn: gemini_contents.append(context_turn)
        if ack_turn: gemini_contents.append(ack_turn)
        gemini_contents.append(final_query_turn)
    else: return {"error": f"Unknown construction strategy: {strategy}"}
    final_payload = {"contents": gemini_contents}
    logger.debug(f"Final payload constructed with {len(gemini_contents)} turns.")
    return final_payload

# --- Function: Assemble Tagged Context (Existing) ---
# <<< Note: This is primarily for putting context *into* the system prompt if needed,
#     which Session Memory doesn't do. It constructs tags. Keeping for completeness. >>>
def assemble_tagged_context(base_prompt: str, contexts: Dict[str, Union[str, List[str]]]) -> str:
    cleaned_prompt = clean_context_tags(base_prompt) if base_prompt else ""
    parts = [cleaned_prompt]
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        context_data = contexts.get(key)
        if context_data:
            content_str = ""
            if isinstance(context_data, list):
                filtered_list = [item for item in context_data if isinstance(item, str) and item.strip()]
                if filtered_list: content_str = "\n---\n".join(filtered_list)
            elif isinstance(context_data, str): content_str = context_data
            else: logger.warning(f"Context data '{key}' unexpected type: {type(context_data)}."); continue
            if content_str.strip(): parts.append(f"\n\n{start_tag}\n{content_str.strip()}\n{end_tag}")
    final_prompt = "\n".join(parts).strip()
    return final_prompt

# --- Function: Extract Tagged Context (Existing) ---
# <<< Note: This extracts ALL tags. process_system_prompt is more specific for initial OWI context.
#     Keeping for completeness. >>>
def extract_tagged_context(system_content: str) -> str:
    if not system_content or not isinstance(system_content, str): return ""
    context_parts = []; found_any = False
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
         pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
         matches = re.findall(pattern, system_content, re.DOTALL | re.IGNORECASE)
         for content in matches:
             content = content.strip()
             if content: label = TAG_LABELS.get(key, key.upper()); context_parts.append(f"--- Context: {label} ---\n{content}"); found_any = True
    if not found_any: return ""
    combined = "\n\n".join(context_parts)
    return combined