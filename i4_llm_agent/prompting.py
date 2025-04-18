# i4_llm_agent/prompting.py

import logging
import re
import asyncio
# <<< Adjusted Import Order >>>
from typing import Tuple, Union, Optional, Dict, List, Any, Callable, Coroutine

# --- Existing Imports from i4_llm_agent ---
# Assume history utils are correctly imported
try:
    from .history import get_recent_turns, format_history_for_llm, DIALOGUE_ROLES
except ImportError:
     # Add basic fallbacks or re-raise if critical
     DIALOGUE_ROLES = ["user", "assistant"]
     def get_recent_turns(*args, **kwargs): return []
     def format_history_for_llm(*args, **kwargs): return ""
     logging.getLogger(__name__).critical("Failed to import history utils in prompting.py")


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
REFINER_QUERY_PLACEHOLDER = "{query}" # Use standard format braces now
REFINER_CONTEXT_PLACEHOLDER = "{external_context}"
REFINER_HISTORY_PLACEHOLDER = "{recent_history_str}"

# <<< Default Refiner Template (for stateless OWI refinement - kept for compatibility) >>>
DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE = f"""
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

# --- NEW: Default RAG Cache Refiner Prompt Template ---
# Define specific placeholders for the curated/cached refiner
RAG_CACHE_REFINER_QUERY_PLACEHOLDER = "{query}"
RAG_CACHE_REFINER_CURRENT_OWI_PLACEHOLDER = "{current_owi_rag}"
RAG_CACHE_REFINER_CACHED_CONTEXT_PLACEHOLDER = "{cached_pipe_rag}"
RAG_CACHE_REFINER_HISTORY_PLACEHOLDER = "{recent_history_str}"

# <<< Default Template Text for the Stateful RAG Cache Refiner >>>
DEFAULT_RAG_CACHE_REFINER_TEMPLATE_TEXT = f"""
**Role:** Session Context Curator & Refiner
**Task:** Analyze the inputs: LATEST USER QUERY, RECENT CHAT HISTORY, CURRENT OWI RETRIEVAL (fresh background info), and PREVIOUSLY REFINED CACHE (the curated background from the last turn). Synthesize these into an **updated, concise, and relevant context block** suitable for answering the query and serving as the cache for the *next* turn.
**Objective:** Create the most accurate and focused background context possible by:
    1.  Extracting information from CURRENT OWI RETRIEVAL and PREVIOUSLY REFINED CACHE that is **directly relevant** to the LATEST USER QUERY and RECENT CHAT HISTORY.
    2.  **Actively filtering out and discarding** information from *both* sources that is **no longer relevant** given the dialogue's current focus and direction.
    3.  Merging and prioritizing the relevant information, ensuring accuracy and preserving key roleplaying details (relationships, motivations, events).
    4.  Producing a single, coherent text block that represents the best possible background context *for this moment in the conversation*.

**Instructions:**
1.  **Identify Core Need:** Determine the specific background information (lore, character details, past events from documents) required to understand and answer the LATEST USER QUERY in the context of the RECENT CHAT HISTORY.
2.  **Evaluate Relevance:** Assess *both* the CURRENT OWI RETRIEVAL and the PREVIOUSLY REFINED CACHE against the core need. Extract sentences or passages **containing only background information** that directly address it.
3.  **Prioritize & Merge Background Info:**
    *   If background information overlaps between sources, prefer the version that is more specific or detailed.
    *   If CURRENT OWI RETRIEVAL contains *new relevant background info* missing from the cache, incorporate it.
    *   If CURRENT OWI RETRIEVAL contradicts cached background info, use the RECENT CHAT HISTORY *only* to help decide which background version seems more applicable *now*, but do not include the history details themselves in the output.
4.  **Filter Aggressively (Pruning Background Info):** Examine *all background information* initially deemed relevant (from both cache and new retrieval). **Discard any background details that are no longer pertinent** to the conversation's current focus. Aim to keep the output tightly focused on the relevant *background*.
5.  **Preserve Relevant RP Nuance (Background Only):** Retain key descriptive background details illustrating character relationships, motivations, or foundational events *from the source documents/cache* if they remain relevant.
6.  **Ensure Accuracy:** Do not infer or add information not present in the provided inputs. Ground the output strictly in the source background texts.
7.  **Handle Empty Inputs:** (Keep as before)
    *   If PREVIOUSLY REFINED CACHE is empty/error, focus solely on refining the CURRENT OWI RETRIEVAL.
    *   If CURRENT OWI RETRIEVAL is empty, focus on refining/filtering the PREVIOUSLY REFINED CACHE.
8.  **CRITICAL - Output Content:** **Your output MUST contain ONLY the relevant background information (lore, character details, relationship history, past events drawn from the CURRENT OWI RETRIEVAL and PREVIOUSLY REFINED CACHE). DO NOT summarize the RECENT CHAT HISTORY itself in your output.** The history is provided *only* to help you judge the relevance of the background information. If no relevant *background* information is found, state clearly: "[No relevant background context found for the current query]".

**INPUTS:**

**LATEST USER QUERY:**
{RAG_CACHE_REFINER_QUERY_PLACEHOLDER}

**CURRENT OWI RETRIEVAL:**
---
{RAG_CACHE_REFINER_CURRENT_OWI_PLACEHOLDER}
---

**PREVIOUSLY REFINED CACHE:**
---
{RAG_CACHE_REFINER_CACHED_CONTEXT_PLACEHOLDER}
---

**RECENT CHAT HISTORY:**
---
{RAG_CACHE_REFINER_HISTORY_PLACEHOLDER}
---

**OUTPUT (Updated & Filtered Relevant BACKGROUND Context ONLY):**
"""

# --- Function: Clean Context Tags (Existing) ---
def clean_context_tags(system_content: str) -> str:
    # (Keep implementation as before)
    if not system_content or not isinstance(system_content, str): return ""
    cleaned = system_content
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = r"\\s*" + re.escape(start_tag) + r".*?" + re.escape(end_tag) + r"\\s*"
        cleaned = re.sub(pattern, "\\n", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\\n{3,}', '\\n\\n', cleaned)
    return cleaned.strip()

# --- Function: Process System Prompt (Existing) ---
def process_system_prompt(messages: List[Dict]) -> Tuple[str, Optional[str]]:
    # (Keep implementation as before)
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

# --- Function: Format Refiner Prompt (for stateless refinement) ---
def format_refiner_prompt(
    external_context: str, # Renamed arg for clarity vs curated refiner
    recent_history_str: str,
    query: str,
    template: Optional[str] = None # Template is now an argument
) -> str:
    """
    Constructs the prompt for the STATELESS external refiner LLM using the provided template.
    Uses placeholders {external_context}, {recent_history_str}, {query}.
    """
    func_logger = logging.getLogger(__name__ + '.format_refiner_prompt')
    # Use provided template or the stateless default
    prompt_template = template if template is not None else DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE
    func_logger.debug(f"Using stateless refiner prompt template (length {len(prompt_template)}).")

    # Basic sanitization (prevent breaking the prompt structure)
    safe_context = external_context.replace("---", "===") if isinstance(external_context, str) else ""
    safe_history = recent_history_str.replace("---", "===") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("---", "===") if isinstance(query, str) else ""

    # Replace placeholders using .format()
    try:
        formatted_prompt = prompt_template.format(
            external_context=safe_context,
            recent_history_str=safe_history,
            query=safe_query
        )
        func_logger.debug(f"Formatted Stateless Refiner Prompt (Snippet): {formatted_prompt[:500]}...")
        return formatted_prompt
    except KeyError as e:
         func_logger.error(f"Missing placeholder in stateless refiner prompt template: {e}. Template snippet: {prompt_template[:300]}...", exc_info=True)
         return f"[Error: Missing placeholder '{e}' in template]"
    except Exception as e:
        func_logger.error(f"Error formatting stateless refiner prompt template: {e}", exc_info=True)
        return f"[Error formatting template: {type(e).__name__}]"


# --- Function: Refine External Context (Stateless OWI refinement) ---
async def refine_external_context(
    external_context: str,
    history_messages: List[Dict],
    latest_user_query: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # Expects async wrapper
    refiner_llm_config: Dict[str, Any], # url, key, temp, prompt_template
    skip_threshold: int,
    history_count: int,
    dialogue_only_roles: List[str] = DIALOGUE_ROLES,
    caller_info: str = "i4_llm_agent_StatelessRefiner"
) -> str:
    """
    Optionally refines the provided external RAG context using an LLM (STATELESS).
    Uses the 'format_refiner_prompt' function and its associated template/placeholders.
    """
    func_logger = logging.getLogger(__name__ + '.refine_external_context')
    func_logger.debug(f"[{caller_info}] Entered refine_external_context (stateless).")

    # --- Input Validation (simplified, more checks in calling code) ---
    if not external_context or not external_context.strip():
        return external_context
    if not latest_user_query or not latest_user_query.strip():
        return external_context
    if not llm_call_func: return external_context
    required_keys = ['url', 'key', 'temp'] # Template is optional here
    if not refiner_llm_config or not all(k in refiner_llm_config for k in required_keys):
        return external_context
    if not refiner_llm_config['url'] or not refiner_llm_config['key']:
         return external_context

    # --- Skip Threshold Check ---
    context_length = len(external_context)
    if skip_threshold > 0 and context_length < skip_threshold:
        func_logger.info(f"[{caller_info}] Skipping stateless refinement: Context length ({context_length}) < Threshold ({skip_threshold}).")
        return external_context

    func_logger.info(f"[{caller_info}] Proceeding with stateless refinement (Context length {context_length}).")

    # --- Prepare Refiner Inputs ---
    recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
    recent_chat_history_str = format_history_for_llm(recent_history_list) if recent_history_list else "[No Recent History]"

    # Use the specific formatter for stateless refinement
    refiner_prompt_text = format_refiner_prompt(
        external_context=external_context,
        recent_history_str=recent_chat_history_str,
        query=latest_user_query,
        template=refiner_llm_config.get('prompt_template') # Pass template from config
    )

    if not refiner_prompt_text or refiner_prompt_text.startswith("[Error:"):
         func_logger.error(f"[{caller_info}] Failed to format stateless refiner prompt: {refiner_prompt_text}. Aborting refinement.")
         return external_context

    refiner_payload = {"contents": [{"parts": [{"text": refiner_prompt_text}]}]}

    # --- Call Refiner LLM ---
    func_logger.info(f"[{caller_info}] Calling Stateless Refiner LLM...")
    try:
        success, response_or_error = await llm_call_func(
            api_url=refiner_llm_config['url'], api_key=refiner_llm_config['key'],
            payload=refiner_payload, temperature=refiner_llm_config['temp'],
            timeout=90, caller_info=caller_info,
        )
    except Exception as e_call:
         func_logger.error(f"[{caller_info}] Exception during llm_call_func: {e_call}", exc_info=True)
         success = False; response_or_error = "LLM Call Exception"

    # --- Process Result ---
    if success and isinstance(response_or_error, str) and response_or_error.strip():
        refined_context = response_or_error.strip()
        func_logger.info(f"[{caller_info}] Stateless refinement successful (Refined length: {len(refined_context)}).")
        return refined_context
    else:
        error_details = str(response_or_error)
        func_logger.warning(f"[{caller_info}] Stateless refinement failed or returned empty. Error: '{error_details}'. Returning original context.")
        return external_context


# --- NEW: Format Curated Refiner Prompt (for RAG Cache) ---
def format_curated_refiner_prompt(
    current_owi_rag: str,
    cached_pipe_rag: str,
    recent_history_str: str,
    query: str,
    template: str # Expecting the template string directly
) -> str:
    """
    Formats the prompt for the STATEFUL RAG Cache refiner LLM using its specific template.
    Uses placeholders {current_owi_rag}, {cached_pipe_rag}, {recent_history_str}, {query}.
    """
    func_logger = logging.getLogger(__name__ + '.format_curated_refiner_prompt')

    if not template or not isinstance(template, str):
        func_logger.error("Missing or invalid template string for curated refiner prompt.")
        return "[Error: Invalid Template Provided]"

    # Basic sanitization (prevent breaking the prompt structure)
    safe_current_owi = current_owi_rag.replace("---", "===") if isinstance(current_owi_rag, str) else ""
    safe_cached_rag = cached_pipe_rag.replace("---", "===") if isinstance(cached_pipe_rag, str) else ""
    safe_history = recent_history_str.replace("---", "===") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("---", "===") if isinstance(query, str) else ""

    # Replace placeholders using .format()
    try:
        # Ensure the template uses the expected placeholder names
        formatted_prompt = template.format(
            current_owi_rag=safe_current_owi,
            cached_pipe_rag=safe_cached_rag,
            recent_history_str=safe_history,
            query=safe_query
        )
        # func_logger.debug(f"Formatted Curated Refiner Prompt (Snippet): {formatted_prompt[:500]}...")
        return formatted_prompt
    except KeyError as e:
         func_logger.error(f"Missing placeholder in curated refiner prompt template: {e}. Template snippet: {template[:300]}...", exc_info=True)
         return f"[Error: Missing placeholder '{e}' in curated template]"
    except Exception as e:
        func_logger.error(f"Error formatting curated refiner prompt template: {e}", exc_info=True)
        return f"[Error formatting curated template: {type(e).__name__}]"


# --- Function: Generate RAG Query (Existing - Ensure it uses async wrapper correctly) ---
async def generate_rag_query(
    latest_message_str: str,
    dialogue_context_str: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]], # Wrapper returns tuple
    llm_config: Dict[str, Any],
    caller_info: str = "i4_llm_agent_RAGQueryGen",
) -> Optional[str]: # Returns string or None/Error string
    # (Keep implementation as before, it correctly uses the wrapper)
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
        # Using .format() for consistency, expecting {latest_message} and {dialogue_context}
        ragq_prompt_text = template.format(latest_message=latest_message_str or "[No user message]", dialogue_context=dialogue_context_str or "[No dialogue context]")
    except KeyError as e: logger.error(f"[{caller_info}] RAGQ prompt key error: {e}."); return f"[Error: RAGQ prompt key error: {e}]"
    except Exception as e_fmt: logger.error(f"[{caller_info}] Failed format RAGQ prompt: {e_fmt}", exc_info=True); return f"[Error: RAGQ prompt format failed - {type(e_fmt).__name__}]"
    ragq_payload = {"contents": [{"parts": [{"text": ragq_prompt_text}]}]}
    logger.info(f"[{caller_info}] Calling LLM for RAG query generation...")
    # Call wrapper which returns Tuple[bool, Union[str, Dict]]
    try:
        success, response_or_error = await llm_call_func(
            api_url=llm_config['url'], api_key=llm_config['key'], payload=ragq_payload,
            temperature=llm_config['temp'], timeout=45, caller_info=caller_info,
        )
    except Exception as e_call:
        logger.error(f"[{caller_info}] Exception calling LLM wrapper for RAGQ: {e_call}", exc_info=True)
        success = False
        response_or_error = f"[Error: LLM Call Exception {type(e_call).__name__}]"

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
    owi_context: Optional[str] = None, # This will now receive the *refined* context
    t1_summaries: Optional[List[str]] = None,
    t2_rag_results: Optional[List[str]] = None,
    separator: str = "\n---\n",
    block_separator: str = "\n\n",
) -> str:
    # (Keep implementation as before - it just combines whatever strings it receives)
    context_parts = []
    # Use a more generic label now, as it might be refined OWI or cached context
    if owi_context and isinstance(owi_context, str) and owi_context.strip() and owi_context != EMPTY_CONTEXT_PLACEHOLDER:
        context_parts.append(f"--- Relevant Background Context ---\n{owi_context.strip()}")
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
    context: Optional[str], # Receives output from combine_background_context
    query: str,
    strategy: str = 'standard',
    include_ack_turns: bool = True
) -> Dict[str, Any]:
    # (Keep implementation as before)
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

# --- Function: Assemble Tagged Context (Existing - Less relevant now) ---
def assemble_tagged_context(base_prompt: str, contexts: Dict[str, Union[str, List[str]]]) -> str:
    # (Keep implementation as before)
    cleaned_prompt = clean_context_tags(base_prompt) if base_prompt else ""
    parts = [cleaned_prompt]
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        context_data = contexts.get(key)
        if context_data:
            content_str = ""
            if isinstance(context_data, list):
                filtered_list = [item for item in context_data if isinstance(item, str) and item.strip()]
                if filtered_list: content_str = "\\n---\\n".join(filtered_list)
            elif isinstance(context_data, str): content_str = context_data
            else: logger.warning(f"Context data '{key}' unexpected type: {type(context_data)}."); continue
            if content_str.strip(): parts.append(f"\\n\\n{start_tag}\\n{content_str.strip()}\\n{end_tag}")
    final_prompt = "\\n".join(parts).strip()
    return final_prompt

# --- Function: Extract Tagged Context (Existing - Less relevant now) ---
def extract_tagged_context(system_content: str) -> str:
    # (Keep implementation as before)
    if not system_content or not isinstance(system_content, str): return ""
    context_parts = []; found_any = False
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
         pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
         matches = re.findall(pattern, system_content, re.DOTALL | re.IGNORECASE)
         for content in matches:
             content = content.strip()
             if content: label = TAG_LABELS.get(key, key.upper()); context_parts.append(f"--- Context: {label} ---\\n{content}"); found_any = True
    if not found_any: return ""
    combined = "\\n\\n".join(context_parts)
    return combined