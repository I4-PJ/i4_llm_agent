# i4_llm_agent/prompting.py

import logging
import re
from typing import Optional, Dict, Union, List, Any

logger = logging.getLogger(__name__) # Gets logger named 'i4_llm_agent.prompting'

# --- Constants for Refiner ---
REFINER_QUERY_PLACEHOLDER = "[Insert Latest User Query Here]"
REFINER_CONTEXT_PLACEHOLDER = "[Insert Retrieved Documents Here]"
REFINER_HISTORY_PLACEHOLDER = "[Insert Recent Chat History Here]"

# Default template copied from main pipe [1] for fallback if needed,
# but the template should ideally always be passed in from the pipe's Valves.
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

**LATEST USER QUERY:** {{REFINER_QUERY_PLACEHOLDER}}
**CONTEXT DOCUMENTS:**
---
{{REFINER_CONTEXT_PLACEHOLDER}}
---
**RECENT CHAT HISTORY:**
---
{{REFINER_HISTORY_PLACEHOLDER}}
---

Concise Relevant Information (for final answer generation):
"""

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
    prompt_template = template if template is not None else DEFAULT_REFINER_PROMPT_TEMPLATE
    logger.debug(f"Using refiner prompt template (length {len(prompt_template)}).")

    # Basic sanitization (prevent breaking the prompt structure)
    safe_context = context.replace("---", "===") if isinstance(context, str) else ""
    safe_history = recent_history_str.replace("---", "===") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("---", "===") if isinstance(query, str) else ""

    # Replace placeholders using .replace() which is safer if template is user-provided
    try:
        formatted_prompt = prompt_template
        # Check if placeholders exist before replacing
        if REFINER_CONTEXT_PLACEHOLDER in formatted_prompt:
            formatted_prompt = formatted_prompt.replace(REFINER_CONTEXT_PLACEHOLDER, safe_context)
        else:
            logger.warning(f"Refiner template missing placeholder: {REFINER_CONTEXT_PLACEHOLDER}")

        if REFINER_HISTORY_PLACEHOLDER in formatted_prompt:
            formatted_prompt = formatted_prompt.replace(REFINER_HISTORY_PLACEHOLDER, safe_history)
        else:
            logger.warning(f"Refiner template missing placeholder: {REFINER_HISTORY_PLACEHOLDER}")

        if REFINER_QUERY_PLACEHOLDER in formatted_prompt:
            formatted_prompt = formatted_prompt.replace(REFINER_QUERY_PLACEHOLDER, safe_query)
        else:
            logger.warning(f"Refiner template missing placeholder: {REFINER_QUERY_PLACEHOLDER}")

        logger.debug(f"Formatted Refiner Prompt (Snippet): {formatted_prompt[:500]}...")
        return formatted_prompt
    except Exception as e:
        logger.error(f"Error formatting refiner prompt template: {e}", exc_info=True)

    # Fallback to a very basic prompt if formatting fails
    logger.warning("Falling back to basic refiner prompt due to formatting error.")
    fallback_prompt = f"Context:\n{safe_context}\n\nHistory:\n{safe_history}\n\nQuery:\n{safe_query}\n\nSummarize relevant info:"
    return fallback_prompt


# --- Constants for Final Payload Constructor ---
EMPTY_CONTEXT_PLACEHOLDER = "[No Background Information Available]"

def construct_final_llm_payload(
    system_prompt: str,
    history: List[Dict],
    context: Optional[str],
    query: str,
    strategy: str = 'standard' # e.g., 'standard' or 'advanced'
) -> Dict[str, Any]:
    """
    Constructs the payload (specifically the 'contents' list for Gemini-like APIs)
    for the final LLM call based on the provided components and strategy.
    Removes artificial ACK turns and only includes context block if context exists.

    Args:
        system_prompt: The cleaned base system prompt.
        history: List of user/assistant message dictionaries (excluding last query).
        context: The (potentially refined) context string.
        query: The latest user query string.
        strategy: Controls the order of injection ('standard' or 'advanced').
                  'standard': Sys -> History -> [Context] -> Query
                  'advanced': Sys -> [Context] -> History -> Query

    Returns:
        A dictionary containing the 'contents' key suitable for the target LLM API.
        Returns {'error': 'message'} on failure (e.g., unknown strategy).
    """
    logger.debug(f"Constructing final LLM payload using strategy: {strategy} (No ACKs, Conditional Context)")
    gemini_contents = []

    # --- 1. System Instructions (Single User Turn) ---
    system_prompt_text = system_prompt or "You are a helpful assistant." # Fallback
    # Ensure system instructions are always present as the first turn
    gemini_contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt_text}"}]})
    # No Model ACK turn follows system instructions
    gemini_contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]}) # Explicit ACK needed

    # --- 2. Prepare History Turns ---
    history_turns = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "").strip() # Ensure content is stripped
        # Map roles to Gemini roles ('user' -> 'user', 'assistant' -> 'model')
        # Skip if role is invalid or content is empty after stripping
        if role == "user" and content:
            history_turns.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant" and content:
            history_turns.append({"role": "model", "parts": [{"text": content}]})
        else:
             logger.debug(f"Skipping history message with role '{role}' or empty content.")

    # --- 3. Prepare Context Turn (ONLY IF CONTEXT EXISTS) ---
    context_turn = None
    ack_turn = None
    # Check if context is not None, not empty after stripping, and not the placeholder text
    has_real_context = bool(context and context.strip() and context.strip() != EMPTY_CONTEXT_PLACEHOLDER)

    if has_real_context:
        logger.debug("Meaningful context provided, creating context injection turn.")
        # Sanitize context slightly (replace potential structure breakers)
        safe_context = context.replace("---", "===") if isinstance(context, str) else context
        # Use a clear header for the context block
        context_injection_text = f"Background Information (Use this to inform your response):\n---\n{safe_context}\n---"
        context_turn = {"role": "user", "parts": [{"text": context_injection_text}]}
        ack_turn = {"role": "model", "parts": [{"text": "Understood. I have reviewed the background information."}]} # Explicit ACK
    else:
        logger.debug("No meaningful context provided, skipping context injection turn.")


    # --- 4. Prepare Final User Query ---
    # Ensure query exists and sanitize
    safe_query = query.strip().replace("---", "===") if query and query.strip() else "[User query not provided or extracted]"
    final_query_turn = {"role": "user", "parts": [{"text": safe_query}]}


    # --- 5. Assemble based on strategy AND whether context exists ---
    if strategy == 'advanced':
        # Order: Sys -> ACK -> [Context -> ACK] -> History -> Query
        logger.debug("Assembling final payload (Advanced Injection: Sys -> [Context] -> History -> Query)")
        if context_turn and ack_turn:
            gemini_contents.append(context_turn) # Add context right after system if it exists
            gemini_contents.append(ack_turn)     # Add ACK for context
        gemini_contents.extend(history_turns)
        gemini_contents.append(final_query_turn)

    elif strategy == 'standard':
        # Order: Sys -> ACK -> History -> [Context -> ACK] -> Query
        logger.debug("Assembling final payload (Standard Injection: Sys -> History -> [Context] -> Query)")
        gemini_contents.extend(history_turns)
        if context_turn and ack_turn:
            gemini_contents.append(context_turn) # Add context after history if it exists
            gemini_contents.append(ack_turn)     # Add ACK for context
        gemini_contents.append(final_query_turn)

    else:
        # Handle unknown strategy
        logger.error(f"Unknown final payload construction strategy: {strategy}")
        return {"error": f"Unknown construction strategy: {strategy}"} # Return error dict


    # --- Final Payload ---
    final_payload = {"contents": gemini_contents}
    # Log the number of turns in the final payload
    logger.debug(f"Final payload constructed with {len(gemini_contents)} turns.")
    # Avoid logging the full potentially large payload unless absolutely necessary for debugging

    return final_payload

# --- Context Tag Management ---

# Define standard tags (could be made configurable via library settings)
KNOWN_CONTEXT_TAGS = {
    "owi": ("<context>", "</context>"),
    "t1": ("<mempipe_recent_summary>", "</mempipe_recent_summary>"),
    "t2_rag": ("<mempipe_rag_result>", "</mempipe_rag_result>"),
}
TAG_LABELS = { # For nicer combined context headers
    "owi": "OWI Context",
    "t1": "Recent Summaries (T1)",
    "t2_rag": "Related Summaries (T2 RAG)",
}


def assemble_tagged_context(
    base_prompt: str,
    contexts: Dict[str, Union[str, List[str]]]
) -> str:
    """
    Cleans known tags from base_prompt and injects new contexts using standard tags.

    Args:
        base_prompt: The original system prompt.
        contexts: A dictionary where keys are context types (e.g., 't1', 't2_rag', 'owi')
                  matching keys in KNOWN_CONTEXT_TAGS. Values are the context string
                  or list of strings.

    Returns:
        The modified system prompt string with injected context blocks.
    """
    logger.debug("Assembling system prompt with tagged context...")
    # Start by cleaning all known tags from the base prompt
    cleaned_prompt = clean_context_tags(base_prompt) if base_prompt else ""
    parts = [cleaned_prompt]

    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        context_data = contexts.get(key)
        if context_data:
            content_str = ""
            if isinstance(context_data, list):
                # Filter out empty strings from list before joining
                filtered_list = [item for item in context_data if isinstance(item, str) and item.strip()]
                if filtered_list:
                    content_str = "\n---\n".join(filtered_list) # Join lists with separator
            elif isinstance(context_data, str):
                content_str = context_data
            else:
                 logger.warning(f"Context data for key '{key}' has unexpected type: {type(context_data)}. Skipping.")
                 continue

            if content_str.strip():
                # Ensure proper spacing around tags
                block = f"\n\n{start_tag}\n{content_str.strip()}\n{end_tag}"
                parts.append(block)
                logger.debug(f"Injecting context block for '{key}'.")
            # else:
                 # logger.debug(f"Skipping injection for '{key}': Content is empty.") # Can be noisy

    # Join parts, ensuring no leading/trailing whitespace on the final result
    final_prompt = "\n".join(parts).strip()
    # logger.debug(f"Assembled tagged context prompt length: {len(final_prompt)}") # Potentially noisy
    return final_prompt

def extract_tagged_context(system_content: str) -> str:
    """
    Extracts content from all known context tags and combines them with labels.
    Returns an empty string if no known tags with content are found.

    Args:
        system_content: The string potentially containing tagged context (e.g., system prompt).

    Returns:
        A single string containing all extracted context blocks, separated and labeled,
        or an empty string if no relevant content is found.
    """
    if not system_content or not isinstance(system_content, str):
        return ""

    logger.debug("Extracting combined context from known tags...")
    context_parts = []
    found_any = False
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
         # Regex to find content between tags, ignoring case and spanning newlines
         # Use non-greedy matching .*?
         pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
         # Find all occurrences
         matches = re.findall(pattern, system_content, re.DOTALL | re.IGNORECASE)
         for content in matches:
             content = content.strip()
             if content:
                 label = TAG_LABELS.get(key, key.upper()) # Get nice label or use key
                 context_parts.append(f"--- Context: {label} ---\n{content}")
                 logger.debug(f"Extracted content for tag '{key}'.")
                 found_any = True
             # else:
                  # logger.debug(f"Found empty tag for '{key}'.") # Noisy

    if not found_any:
         logger.debug("No content found within known context tags.")
         return ""

    combined = "\n\n".join(context_parts)
    # logger.debug(f"Combined extracted context length: {len(combined)}.") # Noisy
    return combined


def clean_context_tags(system_content: str) -> str:
    """
    Removes all known context tags and their content from a string.

    Args:
        system_content: The string to clean.

    Returns:
        The string with all known context tag blocks removed.
    """
    if not system_content or not isinstance(system_content, str):
        return ""

    logger.debug("Cleaning all known context tags from string...")
    cleaned = system_content
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        # Regex to remove the tag pair and everything in between, including surrounding newlines
        # Use non-greedy match .*? and handle potential surrounding whitespace \s*
        pattern = r"\s*" + re.escape(start_tag) + r".*?" + re.escape(end_tag) + r"\s*"
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.DOTALL | re.IGNORECASE) # Replace with newline to avoid merging lines
        # logger.debug(f"Cleaned tags for '{key}'.") # Noisy

    # Clean up potential multiple consecutive newlines resulted from replacements
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
