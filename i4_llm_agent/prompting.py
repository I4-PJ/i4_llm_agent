# i4_llm_agent/prompting.py

import logging
import re
from typing import Optional, Dict, Union, List, Any

logger = logging.getLogger(__name__) # 'i4_llm_agent.prompting'

# --- Constants for Refiner ---
# (Keep REFINER placeholders and DEFAULT_REFINER_PROMPT_TEMPLATE as they were)
REFINER_QUERY_PLACEHOLDER = "[Insert Latest User Query Here]"
REFINER_CONTEXT_PLACEHOLDER = "[Insert Retrieved Documents Here]"
REFINER_HISTORY_PLACEHOLDER = "[Insert Recent Chat History Here]"
DEFAULT_REFINER_PROMPT_TEMPLATE = """
**Role:** Roleplay Context Extractor
# ... (rest of template omitted for brevity - keep as is) ...
"""

# (Keep format_refiner_prompt function as it was)
def format_refiner_prompt(context: str, recent_history_str: str, query: str, template: Optional[str] = None) -> str:
    prompt_template = template if template is not None else DEFAULT_REFINER_PROMPT_TEMPLATE
    # logger.debug(f"Using refiner prompt template (length {len(prompt_template)}).")
    safe_context = context.replace("---", "===") if isinstance(context, str) else ""
    safe_history = recent_history_str.replace("---", "===") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("---", "===") if isinstance(query, str) else ""
    try:
        formatted_prompt = prompt_template
        if REFINER_CONTEXT_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_CONTEXT_PLACEHOLDER, safe_context)
        # else: logger.warning(f"Refiner template missing: {REFINER_CONTEXT_PLACEHOLDER}") # Optional warnings
        if REFINER_HISTORY_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_HISTORY_PLACEHOLDER, safe_history)
        # else: logger.warning(f"Refiner template missing: {REFINER_HISTORY_PLACEHOLDER}")
        if REFINER_QUERY_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_QUERY_PLACEHOLDER, safe_query)
        # else: logger.warning(f"Refiner template missing: {REFINER_QUERY_PLACEHOLDER}")
        # logger.debug(f"Formatted Refiner Prompt (Snippet): {formatted_prompt[:500]}...")
        return formatted_prompt
    except Exception as e:
        logger.error(f"Error formatting refiner prompt template: {e}", exc_info=True)
    logger.warning("Falling back to basic refiner prompt due to formatting error.")
    fallback_prompt = f"Context:\n{safe_context}\n\nHistory:\n{safe_history}\n\nQuery:\n{safe_query}\n\nSummarize relevant info:"
    return fallback_prompt

# --- Constants for Final Payload Constructor ---
EMPTY_CONTEXT_PLACEHOLDER = "[No Background Information Available]"

# <<< Updated Signature: Added include_ack_turns >>>
def construct_final_llm_payload(
    system_prompt: str,
    history: List[Dict],
    context: Optional[str],
    query: str,
    strategy: str = 'standard', # 'standard' or 'advanced'
    include_ack_turns: bool = True # <<< New parameter
) -> Dict[str, Any]:
    """
    Constructs the payload ('contents' list for Gemini-like APIs).
    Optionally includes ACK turns. Includes context block only if context exists.

    Args:
        system_prompt: The base system prompt (potentially enhanced).
        history: List of user/assistant message dictionaries (T0 slice).
        context: The combined background context string (OWI/T1/T2).
        query: The latest user query string.
        strategy: 'standard' (Sys->Hist->Ctx->Q) or 'advanced' (Sys->Ctx->Hist->Q).
        include_ack_turns: If True, adds 'model' ACK turns after system and context.

    Returns:
        A dictionary containing the 'contents' key, or {'error': 'message'}.
    """
    logger.debug(f"Constructing final LLM payload. Strategy: {strategy}, ACKs: {include_ack_turns}")
    gemini_contents = []

    # --- 1. System Instructions ---
    system_prompt_text = system_prompt or "You are a helpful assistant."
    gemini_contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt_text}"}]})
    # <<< Conditional ACK >>>
    if include_ack_turns:
        gemini_contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})

    # --- 2. Prepare History Turns (T0) ---
    history_turns = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role == "user" and content:
            history_turns.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant" and content:
            history_turns.append({"role": "model", "parts": [{"text": content}]})
        # else: logger.debug(f"Skipping history msg role '{role}' or empty content.")

    # --- 3. Prepare Context Turn (Background Info - T1/T2/OWI) ---
    context_turn = None
    ack_turn = None
    has_real_context = bool(context and context.strip() and context.strip() != EMPTY_CONTEXT_PLACEHOLDER)
    if has_real_context:
        # logger.debug("Meaningful context provided, creating context injection turn.")
        safe_context = context.replace("---", "===") if isinstance(context, str) else context
        context_injection_text = f"Background Information (Use this to inform your response):\n---\n{safe_context}\n---"
        context_turn = {"role": "user", "parts": [{"text": context_injection_text}]}
        # <<< Conditional ACK >>>
        if include_ack_turns:
            ack_turn = {"role": "model", "parts": [{"text": "Understood. I have reviewed the background information."}]}
    # else: logger.debug("No meaningful context provided, skipping context injection turn.")

    # --- 4. Prepare Final User Query ---
    safe_query = query.strip().replace("---", "===") if query and query.strip() else "[User query not provided]"
    final_query_turn = {"role": "user", "parts": [{"text": safe_query}]}

    # --- 5. Assemble based on strategy ---
    if strategy == 'advanced':
        # Order: Sys -> [ACK] -> [Context -> [ACK]] -> History -> Query
        logger.debug("Assembling payload (Advanced: Sys -> [Ctx] -> Hist -> Query)")
        if context_turn: gemini_contents.append(context_turn)
        if ack_turn: gemini_contents.append(ack_turn) # ACK follows context
        gemini_contents.extend(history_turns)
        gemini_contents.append(final_query_turn)

    elif strategy == 'standard':
        # Order: Sys -> [ACK] -> History -> [Context -> [ACK]] -> Query
        logger.debug("Assembling payload (Standard: Sys -> Hist -> [Ctx] -> Query)")
        gemini_contents.extend(history_turns)
        if context_turn: gemini_contents.append(context_turn)
        if ack_turn: gemini_contents.append(ack_turn) # ACK follows context
        gemini_contents.append(final_query_turn)

    else:
        logger.error(f"Unknown final payload construction strategy: {strategy}")
        return {"error": f"Unknown construction strategy: {strategy}"}

    # --- Final Payload ---
    final_payload = {"contents": gemini_contents}
    logger.debug(f"Final payload constructed with {len(gemini_contents)} turns.")
    return final_payload

# --- Context Tag Management ---
# (Keep KNOWN_CONTEXT_TAGS, TAG_LABELS, assemble_tagged_context,
#  extract_tagged_context, clean_context_tags functions as they were)
KNOWN_CONTEXT_TAGS = { "owi": ("<context>", "</context>"), "t1": ("<mempipe_recent_summary>", "</mempipe_recent_summary>"), "t2_rag": ("<mempipe_rag_result>", "</mempipe_rag_result>"),}
TAG_LABELS = { "owi": "OWI Context", "t1": "Recent Summaries (T1)", "t2_rag": "Related Summaries (T2 RAG)",}
def assemble_tagged_context(base_prompt: str, contexts: Dict[str, Union[str, List[str]]]) -> str:
    # logger.debug("Assembling system prompt with tagged context...")
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
            else: logger.warning(f"Context data for '{key}' unexpected type: {type(context_data)}. Skipping."); continue
            if content_str.strip():
                block = f"\n\n{start_tag}\n{content_str.strip()}\n{end_tag}"
                parts.append(block)
                # logger.debug(f"Injecting context block for '{key}'.")
    final_prompt = "\n".join(parts).strip()
    return final_prompt
def extract_tagged_context(system_content: str) -> str:
    if not system_content or not isinstance(system_content, str): return ""
    # logger.debug("Extracting combined context from known tags...")
    context_parts = []
    found_any = False
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
         pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
         matches = re.findall(pattern, system_content, re.DOTALL | re.IGNORECASE)
         for content in matches:
             content = content.strip()
             if content:
                 label = TAG_LABELS.get(key, key.upper())
                 context_parts.append(f"--- Context: {label} ---\n{content}")
                 # logger.debug(f"Extracted content for tag '{key}'.")
                 found_any = True
    if not found_any: return ""
    combined = "\n\n".join(context_parts)
    return combined
def clean_context_tags(system_content: str) -> str:
    if not system_content or not isinstance(system_content, str): return ""
    # logger.debug("Cleaning all known context tags from string...")
    cleaned = system_content
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = r"\s*" + re.escape(start_tag) + r".*?" + re.escape(end_tag) + r"\s*"
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()