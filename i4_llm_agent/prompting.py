# i4_llm_agent/prompting.py

import logging
import re
import asyncio
# <<< MODIFIED Import Order: Tuple first >>>
from typing import Tuple, Union, Optional, Dict, List, Any, Callable, Coroutine

logger = logging.getLogger(__name__) # 'i4_llm_agent.prompting'

# --- Constants for Context Tags ---
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
REFINER_QUERY_PLACEHOLDER = "[Insert Latest User Query Here]"
REFINER_CONTEXT_PLACEHOLDER = "[Insert Retrieved Documents Here]"
REFINER_HISTORY_PLACEHOLDER = "[Insert Recent Chat History Here]"
DEFAULT_REFINER_PROMPT_TEMPLATE = """
**Role:** Roleplay Context Extractor
(rest of template omitted for brevity) ...
"""

# --- Function: Clean Context Tags ---
def clean_context_tags(system_content: str) -> str:
    if not system_content or not isinstance(system_content, str): return ""
    cleaned = system_content
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = r"\s*" + re.escape(start_tag) + r".*?" + re.escape(end_tag) + r"\s*"
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# --- Function: Process System Prompt ---
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
# (No changes needed)
def format_refiner_prompt(context: str, recent_history_str: str, query: str, template: Optional[str] = None) -> str:
    prompt_template = template if template is not None else DEFAULT_REFINER_PROMPT_TEMPLATE
    safe_context = context.replace("---", "===") if isinstance(context, str) else ""
    safe_history = recent_history_str.replace("---", "===") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("---", "===") if isinstance(query, str) else ""
    try:
        formatted_prompt = prompt_template
        if REFINER_CONTEXT_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_CONTEXT_PLACEHOLDER, safe_context)
        if REFINER_HISTORY_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_HISTORY_PLACEHOLDER, safe_history)
        if REFINER_QUERY_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_QUERY_PLACEHOLDER, safe_query)
        return formatted_prompt
    except Exception as e:
        logger.error(f"Error formatting refiner prompt template: {e}", exc_info=True)
        logger.warning("Falling back to basic refiner prompt due to formatting error.")
        fallback_prompt = f"Context:\n{safe_context}\n\nHistory:\n{safe_history}\n\nQuery:\n{safe_query}\n\nSummarize relevant info:"
        return fallback_prompt

# --- Function: Generate RAG Query ---
# (No changes needed, uses the pipe's async wrapper which returns Tuple now)
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


# --- Function: Combine Background Context ---
# (No changes needed)
def combine_background_context(
    owi_context: Optional[str] = None,
    t1_summaries: Optional[List[str]] = None,
    t2_rag_results: Optional[List[str]] = None,
    separator: str = "\n---\n",
    block_separator: str = "\n\n",
) -> str:
    context_parts = []
    if owi_context and isinstance(owi_context, str) and owi_context.strip():
        context_parts.append(f"--- Original OWI Context ---\n{owi_context.strip()}")
    if t1_summaries and isinstance(t1_summaries, list):
        valid_t1 = [s for s in t1_summaries if isinstance(s, str) and s.strip()]
        if valid_t1: context_parts.append(f"--- Recent Summaries (T1) ---\n{separator.join(valid_t1)}")
    if t2_rag_results and isinstance(t2_rag_results, list):
        valid_t2 = [s for s in t2_rag_results if isinstance(s, str) and s.strip()]
        if valid_t2: context_parts.append(f"--- Related Summaries (T2 RAG) ---\n{separator.join(valid_t2)}")
    if not context_parts: return EMPTY_CONTEXT_PLACEHOLDER
    combined_context_string = block_separator.join(context_parts).strip()
    return combined_context_string

# --- Function: Construct Final LLM Payload ---
# (No changes needed)
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

# --- Function: Assemble Tagged Context ---
# (No changes needed)
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

# --- Function: Extract Tagged Context ---
# (No changes needed)
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