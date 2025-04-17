# i4_llm_agent/prompting.py

import logging
import re
import asyncio # Needed for async function generate_rag_query
from typing import Optional, Dict, Union, List, Any, Callable, Coroutine

logger = logging.getLogger(__name__) # 'i4_llm_agent.prompting'

# --- Constants for Refiner ---
# (Keep REFINER placeholders and DEFAULT_REFINER_PROMPT_TEMPLATE as they were)
REFINER_QUERY_PLACEHOLDER = "[Insert Latest User Query Here]"
REFINER_CONTEXT_PLACEHOLDER = "[Insert Retrieved Documents Here]"
REFINER_HISTORY_PLACEHOLDER = "[Insert Recent Chat History Here]"
DEFAULT_REFINER_PROMPT_TEMPLATE = """
**Role:** Roleplay Context Extractor
You are an AI assistant specializing in analyzing dialogue history and external documents to generate a concise, relevant search query for retrieving information from a long-term memory store containing past conversation summaries.

**Objective:** Based *only* on the LATEST USER QUERY and the RECENT CHAT HISTORY provided, generate a search query that captures the core information need expressed or implied. The query should be suitable for a vector database search over conversation summaries.

**Instructions:**
1.  **Focus:** Prioritize the specific topic, question, or keywords in the LATEST USER QUERY.
2.  **Context:** Use the RECENT CHAT HISTORY *only* to understand the immediate context and potentially refine the query (e.g., resolve pronouns, understand implicit references). **DO NOT** use the `[Insert Retrieved Documents Here]` section for generating the query.
3.  **Conciseness:** Keep the query relatively short and keyword-focused. Avoid full sentences unless necessary to capture meaning.
4.  **Format:** Output *only* the generated search query, with no preamble, labels, or explanations.

**Input Data:**
*   **Latest User Query:** The most recent message from the user.
*   **Recent Chat History:** A few turns of the conversation preceding the latest user query.
*   **Retrieved Documents (IGNORE FOR QUERY GENERATION):** Summaries retrieved in a previous step. Ignore these when creating the *new* search query.

---
**LATEST USER QUERY:**
{query_placeholder}
---
**RECENT CHAT HISTORY:**
{history_placeholder}
---
**RETRIEVED DOCUMENTS (IGNORE FOR QUERY GENERATION):**
{context_placeholder}
---

**Generated Search Query:**
""" # Use specific placeholder names from constants above

# (Keep format_refiner_prompt function as it was)
def format_refiner_prompt(context: str, recent_history_str: str, query: str, template: Optional[str] = None) -> str:
    """Formats the prompt for the Refiner LLM."""
    prompt_template = template if template is not None else DEFAULT_REFINER_PROMPT_TEMPLATE
    # logger.debug(f"Using refiner prompt template (length {len(prompt_template)}).")
    safe_context = context.replace("---", "===") if isinstance(context, str) else ""
    safe_history = recent_history_str.replace("---", "===") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("---", "===") if isinstance(query, str) else ""
    try:
        formatted_prompt = prompt_template
        # Replace using defined constants for clarity
        if REFINER_CONTEXT_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_CONTEXT_PLACEHOLDER, safe_context)
        if REFINER_HISTORY_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_HISTORY_PLACEHOLDER, safe_history)
        if REFINER_QUERY_PLACEHOLDER in formatted_prompt: formatted_prompt = formatted_prompt.replace(REFINER_QUERY_PLACEHOLDER, safe_query)
        return formatted_prompt
    except Exception as e:
        logger.error(f"Error formatting refiner prompt template: {e}", exc_info=True)
    logger.warning("Falling back to basic refiner prompt due to formatting error.")
    fallback_prompt = f"Context:\n{safe_context}\n\nHistory:\n{safe_history}\n\nQuery:\n{safe_query}\n\nSummarize relevant info:"
    return fallback_prompt


# --- Constants for Final Payload Constructor ---
EMPTY_CONTEXT_PLACEHOLDER = "[No Background Information Available]"

# --- RAG Query Generation Function ---
async def generate_rag_query(
    latest_message_str: str,
    dialogue_context_str: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Optional[str]]],
    llm_config: Dict[str, Any],
    caller_info: str = "i4_llm_agent_RAGQueryGen",
) -> Optional[str]:
    """
    Generates a search query for RAG using an LLM based on the latest message
    and recent dialogue.

    Args:
        latest_message_str: The content of the latest user message.
        dialogue_context_str: A string representing the recent dialogue history
                              (e.g., formatted by format_history_for_llm).
        llm_call_func: An async function (like the pipe's wrapper) to call the LLM.
                       Expected signature: async def func(api_url, api_key, payload, temperature, ...)
        llm_config: A dictionary containing:
            - 'url': The LLM API endpoint URL.
            - 'key': The LLM API key.
            - 'temp': The temperature for the query generation LLM call.
            - 'prompt': The prompt template string (must contain {latest_message}
                        and {dialogue_context} placeholders).
        caller_info: Identifier string for logging purposes within the LLM call.

    Returns:
        The generated query string if successful, otherwise None or an error string.
        Returns None if required configuration is missing.
    """
    logger.debug(f"[{caller_info}] Generating RAG query...")

    # --- Validate Inputs ---
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        logger.error(f"[{caller_info}] Provided llm_call_func is not a valid async function.")
        return "[Error: Invalid LLM call function provided]"
    required_keys = ['url', 'key', 'temp', 'prompt']
    if not llm_config or not all(k in llm_config for k in required_keys):
        missing = [k for k in required_keys if k not in llm_config]
        logger.error(f"[{caller_info}] LLM config is missing required keys: {missing}")
        return f"[Error: Missing LLM config keys for RAG query generation: {missing}]"
    if not llm_config['url'] or not llm_config['key']:
        logger.error(f"[{caller_info}] LLM URL or Key is missing in config.")
        return "[Error: Missing LLM URL/Key for RAG query generation]"
    if not llm_config['prompt'] or not isinstance(llm_config['prompt'], str):
         logger.error(f"[{caller_info}] RAG query prompt template is missing or invalid in config.")
         return "[Error: Missing/Invalid RAG query prompt template]"

    # --- Format Prompt ---
    ragq_prompt_text = "[Error: Prompt Formatting Failed]"
    try:
        # Basic check for placeholders to avoid crashing .format if they are missing
        template = llm_config['prompt']
        if "{latest_message}" not in template or "{dialogue_context}" not in template:
            logger.warning(f"[{caller_info}] RAG query prompt template might be missing placeholders '{{latest_message}}' or '{{dialogue_context}}'.")
            # Attempt format anyway, it might use different names or logic
        ragq_prompt_text = template.format(
            latest_message=latest_message_str or "[No user message provided]",
            dialogue_context=dialogue_context_str or "[No dialogue context provided]",
        )
        # logger.debug(f"[{caller_info}] Formatted RAG query prompt: {ragq_prompt_text[:500]}...") # Log snippet
    except KeyError as e:
         logger.error(f"[{caller_info}] Prompt template missing expected key: {e}. Template: {llm_config['prompt'][:200]}...")
         return f"[Error: RAG query prompt template key error: {e}]"
    except Exception as e_fmt:
        logger.error(f"[{caller_info}] Failed to format RAG query prompt: {e_fmt}", exc_info=True)
        return f"[Error: RAG query prompt formatting failed - {type(e_fmt).__name__}]"

    # --- Construct Payload ---
    ragq_payload = {"contents": [{"parts": [{"text": ragq_prompt_text}]}]}

    # --- Call LLM ---
    logger.info(f"[{caller_info}] Calling LLM for RAG query generation...")
    generated_query = await llm_call_func(
        api_url=llm_config['url'],
        api_key=llm_config['key'],
        payload=ragq_payload,
        temperature=llm_config['temp'],
        timeout=45, # Set a reasonable timeout for query generation
        caller_info=caller_info,
    )

    # --- Process Result ---
    if isinstance(generated_query, str) and not generated_query.startswith("[Error:"):
        final_query = generated_query.strip()
        if final_query:
            logger.info(f"[{caller_info}] Successfully generated RAG query: '{final_query}'")
            return final_query
        else:
            logger.warning(f"[{caller_info}] RAG query generation LLM returned an empty string.")
            return "[Error: RAG query generation returned empty string]"
    elif isinstance(generated_query, str): # It's an error string
        logger.error(f"[{caller_info}] RAG query generation failed: {generated_query}")
        return generated_query # Propagate the error string
    else: # Should not happen if llm_call_func adheres to contract, but handle defensively
        logger.error(f"[{caller_info}] RAG query generation returned unexpected type: {type(generated_query)}. Result: {generated_query}")
        return "[Error: Unexpected result from RAG query generation LLM call]"


# --- Background Context Combination Function ---
def combine_background_context(
    owi_context: Optional[str] = None,
    t1_summaries: Optional[List[str]] = None,
    t2_rag_results: Optional[List[str]] = None,
    separator: str = "\n---\n", # Separator for items within lists (T1/T2)
    block_separator: str = "\n\n", # Separator between context types (OWI/T1/T2)
) -> str:
    """
    Combines different sources of background context into a single formatted string.

    Args:
        owi_context: String containing the original OWI context.
        t1_summaries: List of strings, each a recent T1 summary.
        t2_rag_results: List of strings, each a retrieved T2 RAG result.
        separator: String used to join multiple summaries/results within a block.
        block_separator: String used to separate the different context type blocks.

    Returns:
        A single string containing the combined context, formatted with labels,
        or a placeholder string if no context is provided.
    """
    # logger.debug("Combining background context sources...")
    context_parts = []

    # 1. OWI Context
    if owi_context and isinstance(owi_context, str) and owi_context.strip():
        context_parts.append(
            f"--- Original OWI Context ---\n{owi_context.strip()}"
        )
        # logger.debug("Added OWI context.")

    # 2. Tier 1 Summaries
    if t1_summaries and isinstance(t1_summaries, list):
        valid_t1 = [s for s in t1_summaries if isinstance(s, str) and s.strip()]
        if valid_t1:
            joined_t1 = separator.join(valid_t1)
            context_parts.append(
                f"--- Recent Summaries (T1) ---\n{joined_t1}"
            )
            # logger.debug(f"Added {len(valid_t1)} T1 summaries.")

    # 3. Tier 2 RAG Results
    if t2_rag_results and isinstance(t2_rag_results, list):
        valid_t2 = [s for s in t2_rag_results if isinstance(s, str) and s.strip()]
        if valid_t2:
            joined_t2 = separator.join(valid_t2)
            context_parts.append(
                f"--- Related Summaries (T2 RAG) ---\n{joined_t2}"
            )
            # logger.debug(f"Added {len(valid_t2)} T2 RAG results.")

    # --- Combine Blocks ---
    if not context_parts:
        # logger.debug("No background context provided.")
        return EMPTY_CONTEXT_PLACEHOLDER # Use the constant defined earlier

    combined_context_string = block_separator.join(context_parts).strip()
    # logger.debug(f"Combined context string created (length: {len(combined_context_string)}).")
    return combined_context_string


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
        context: The combined background context string (OWI/T1/T2). Should come
                 from `combine_background_context` or be None/placeholder.
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
    # Check if the context is meaningful (not None, not empty, and not the placeholder)
    has_real_context = bool(context and context.strip() and context.strip() != EMPTY_CONTEXT_PLACEHOLDER)

    if has_real_context:
        # logger.debug("Meaningful context provided, creating context injection turn.")
        # Context should already be formatted by combine_background_context
        context_injection_text = f"Background Information (Use this to inform your response):\n{context}" # Simplified injection
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
#  extract_tagged_context, clean_context_tags functions as they were - no changes needed here)
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
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned) # Consolidate excessive newlines
    return cleaned.strip()