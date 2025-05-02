# === COMPLETE CORRECTED BASE FILE: i4_llm_agent/prompting.py (Added Cache Maintainer) ===
# i4_llm_agent/prompting.py

import logging
import re
import asyncio
import json # Used for formatting scene keywords example
from typing import Tuple, Union, Optional, Dict, List, Any, Callable, Coroutine

# --- Existing Imports from i4_llm_agent ---
try:
    from .history import get_recent_turns, format_history_for_llm, DIALOGUE_ROLES
except ImportError:
     DIALOGUE_ROLES = ["user", "assistant"]
     def get_recent_turns(*args, **kwargs): return []
     def format_history_for_llm(*args, **kwargs): return ""
     logging.getLogger(__name__).critical("Failed to import history utils in prompting.py")

try:
    # Import the guideline text constant and the formatter for event hints
    from .event_hints import EVENT_HANDLING_GUIDELINE_TEXT, format_hint_for_query
except ImportError:
    EVENT_HANDLING_GUIDELINE_TEXT = "[EVENT GUIDELINE LOAD FAILED]"
    def format_hint_for_query(hint): return f"[[Hint Load Failed: {hint}]]"
    logging.getLogger(__name__).error("Failed to import event_hints utils in prompting.py")

logger = logging.getLogger(__name__) # 'i4_llm_agent.prompting'

# --- Constants for Context Tags (XML-STYLE focused) ---
# Kept for cleaning old formats if necessary and for combine_background_context
KNOWN_CONTEXT_TAGS = {
    "owi": ("<context>", "</context>"),
    "t1": ("<mempipe_recent_summary>", "</mempipe_recent_summary>"),
    "t2_rag": ("<mempipe_rag_result>", "</mempipe_rag_result>"),
}
EMPTY_CONTEXT_PLACEHOLDER = "<Context type='Empty'>[No Background Information Available]</Context>"


# === Summarizer Prompt Constants (From Base) ===
SUMMARIZER_DIALOGUE_CHUNK_PLACEHOLDER = "{dialogue_chunk}"
DEFAULT_SUMMARIZER_SYSTEM_PROMPT = f"""
[[SYSTEM DIRECTIVE]]

**Role:** Roleplay Dialogue Chunk Summarizer & Memory Extractor

**Objective:**
Analyze the provided DIALOGUE CHUNK (representing recent chat history) and produce a **high-fidelity memory summary** to preserve emotional, practical, relationship, and world realism *expressed within this specific chunk* for future roleplay continuation.

**Primary Goals:**

1.  **Scene Context (from Chunk):**
    *   Capture the basic physical situation: location, time of day, environmental effects *as described EXPLICITLY in the DIALOGUE CHUNK*.
2.  **Emotional State Changes (from Chunk):**
    *   Track emotional shifts expressed *in the DIALOGUE CHUNK*: fear, hope, anger, guilt, trust, resentment, affection. Mention which character expressed them.
3.  **Relationship Developments (from Chunk):**
    *   Describe how trust, distance, dependence, or emotional connections evolved *during this DIALOGUE CHUNK*.
4.  **Practical Developments (from Chunk):**
    *   Capture important practical events *mentioned in the DIALOGUE CHUNK*: travel hardships, fatigue, injury, hunger, gear changes, environmental obstacles.
5.  **World-State Changes (from Chunk):**
    *   Record important plot/world events *stated in the DIALOGUE CHUNK*: route changes, enemy movements, political developments, survival risks.
6.  **Critical Dialogue Fragments (from Chunk):**
    *   Identify and preserve 1–3 **critical quotes** or **key emotional exchanges** *from the DIALOGUE CHUNK*.
    *   These must reflect major emotional turning points, confessions, confrontations, or promises *within this chunk*.
    *   Use near-verbatim phrasing when possible.
7.  **Continuity Anchors (from Chunk):**
    *   Identify important facts, feelings, or decisions *from this DIALOGUE CHUNK* that must be remembered for emotional and logical continuity in future roleplay.

**Compression and Length Policy:**
*   **Do NOT prioritize token-saving compression over realism.** Length is flexible depending on the density of the DIALOGUE CHUNK.
*   Allow **longer outputs naturally** for chunks rich in emotional conflict or tactical discussion.
*   Aggressively compress only if the chunk is mostly trivial small-talk.

**Accuracy Policy:**
*   Only extract facts, emotions, or quotes that are explicitly present or strongly implied *within the provided DIALOGUE CHUNK*.
*   **Do NOT invent or assume information.** Do not refer to context outside the chunk.

**Tone Handling:**
*   Preserve emotional nuance and character complexity expressed *in the DIALOGUE CHUNK*.

---

[[INPUT]]

**DIALOGUE CHUNK TO SUMMARIZE:**
---
{SUMMARIZER_DIALOGUE_CHUNK_PLACEHOLDER}
---

---

[[OUTPUT STRUCTURE]]

**Scene Location and Context:**
(description based *only* on dialogue chunk)

**Emotional State Changes (per character):**
- (Character Name): emotional shifts *expressed in chunk*.

**Relationship Developments:**
- (short descriptions *from chunk*)

**Practical Developments:**
- (details about survival, fatigue, injuries, supplies *mentioned in chunk*)

**World-State Changes:**
- (plot changes, movement of threats, discoveries *stated in chunk*)

**Critical Dialogue Fragments:**
- (List 1–3 key quotes *from this chunk* that define emotional turning points)

**Important Continuity Anchors:**
- (Facts, feelings, or decisions *from this chunk* that must persist.)

---

[[NOTES]]
- Focus **exclusively** on the provided DIALOGUE CHUNK.
- Base the summary *only* on the text within the chunk.
- Prioritize emotional realism and narrative continuity over brevity based on the chunk's content.

"""

# === Memory Aging Prompt Constants (From Base) ===
MEMORY_AGING_BATCH_PLACEHOLDER = "{t1_batch_text}"
DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE = f"""
[[SYSTEM DIRECTIVE]]

**Role:** Roleplay Memory Condenser

**Objective:** Analyze the provided CONSOLIDATED TEXT (representing a sequence of older dialogue summaries) and produce a **single, concise narrative recap** that preserves the essential plot progression, key emotional shifts, critical relationship developments, and crucial continuity anchors from that period.

**Input:** A block of text containing multiple sequential T1 summaries concatenated together.

**Output:** A single paragraph of text summarizing the input block.

**Instructions:**

1.  **Identify Core Narrative Arc:** Read the CONSOLIDATED TEXT to understand the main events, character interactions, and emotional flow across the summarized period. What was the overall journey or change during this time?
2.  **Extract Key Developments:** Identify the most important plot points, decisions, discoveries, relationship changes (positive or negative), and significant emotional moments described in the input text.
3.  **Synthesize, Don't Just List:** Combine the extracted developments into a flowing narrative recap. Focus on cause and effect, character motivations (as described), and the *outcome* of the events in the batch.
4.  **Prioritize Continuity:** Ensure the recap includes details necessary to understand *why* the current situation (following this batch) is the way it is. What information *must* be retained for the story to make sense going forward?
5.  **Condense Ruthlessly:** While preserving essential information, omit minor details, repetitive descriptions, and less critical dialogue snippets found in the original summaries. Aim for significant token reduction compared to the input text.
6.  **Maintain Tone:** Reflect the general emotional tone of the period summarized (e.g., hopeful, tense, tragic).
7.  **Single Paragraph Output:** The final output must be a single block of text (one paragraph).

**Accuracy Note:** Base the recap *only* on the information present in the CONSOLIDATED TEXT. Do not infer or add external knowledge.

---

[[INPUT]]

**CONSOLIDATED TEXT (Sequential T1 Summaries):**
---
{MEMORY_AGING_BATCH_PLACEHOLDER}
---

---

[[OUTPUT]]

**(Single Paragraph Narrative Recap):**
"""

# === RAG Query Prompt Constant (From Base) ===
DEFAULT_RAGQ_LLM_PROMPT = """Based on the latest user message and recent dialogue context, generate a concise search query focusing on the key entities, topics, or questions raised.

Latest Message: {latest_message}

Dialogue Context:
{dialogue_context}

Search Query:"""

# === Inventory Update Prompt Constants (From Base) ===
INVENTORY_UPDATE_RESPONSE_PLACEHOLDER = "{main_llm_response}"
INVENTORY_UPDATE_QUERY_PLACEHOLDER = "{user_query}"
INVENTORY_UPDATE_HISTORY_PLACEHOLDER = "{recent_history_str}"
DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT = f"""
[[SYSTEM DIRECTIVE]]
**Role:** Inventory Log Keeper
**Task:** Analyze the latest interaction (User Query, Assistant Response, Recent History) to identify explicit changes to character inventories, stated via direct commands OR described in dialogue.
**Objective:** Output a structured JSON object detailing ONLY the inventory changes detected. If no changes are detected, output an empty JSON object.

**Supported Direct Command Formats (Priority 1):**
*   `INVENTORY: ADD CharacterName: Item Name=Quantity[, Item Name=Quantity...]`
*   `INVENTORY: REMOVE CharacterName: Item Name=Quantity[, Item Name=Quantity...]`
*   `INVENTORY: SET CharacterName: Item Name=Quantity[, Item Name=Quantity...]`
*   `INVENTORY: CLEAR CharacterName`
*(Note: Use `__USER__` for the player character if their specific name isn't provided)*

**Instructions (Follow in Order):**

1.  **Check for Strict Commands:** Examine the **USER QUERY**. Does it start with `INVENTORY:` followed immediately by `ADD`, `REMOVE`, `SET`, or `CLEAR`?
    *   If YES: Parse the command **strictly** according to the formats above. Generate JSON `updates` based *only* on the parsed command. **Stop processing and output the JSON.**
    *   If NO: Proceed to Instruction 2.

2.  **Check for Natural Language Command:** Examine the **USER QUERY**. Does it start with `INVENTORY:` but is **NOT** followed immediately by `ADD`, `REMOVE`, `SET`, or `CLEAR`?
    *   If YES: Attempt to interpret the text *after* the `INVENTORY:` prefix as a **natural language instruction** about desired inventory changes (e.g., "Emily doesn't need her dress anymore", "Give the health potion to Caldric"). Generate the corresponding JSON `updates` array based on your best interpretation of the instruction. **If the natural language instruction is ambiguous, unclear, or seems unrelated to inventory, output `{{"updates": []}}`. Stop processing and output the JSON.**
    *   If NO: Proceed to Instruction 3.

3.  **Analyze Dialogue (Fallback):** Since no `INVENTORY:` command (strict or natural language) was found in the User Query, analyze the **ASSISTANT RESPONSE** (using User Query and History for context) for narrative descriptions of inventory changes (e.g., picking up, dropping, giving, receiving, using consumables, crafting, buying, selling).
    *   Identify actions, characters (use `__USER__` if needed, resolve pronouns), items, and quantities (default 1) from the dialogue.
    *   Generate JSON `updates` based *only* on these dialogue events.

4.  **Format Output as JSON:** Structure the output STRICTLY as the following JSON format:
    ```json
    {{
      "updates": [
        // One entry for each detected change (from command OR dialogue)
        {{
          "character_name": "Name or __USER__",
          "action": "add | remove | set_quantity", // Use 'set_quantity' for SET command
          "item_name": "Exact Item Name or __ALL_ITEMS__", // Use __ALL_ITEMS__ only for CLEAR
          "quantity": <integer>,
          "description": "<optional string>" // Typically only for 'add' from dialogue
        }}
        // ... more updates if needed
      ]
    }}
    ```
    *   **Important:** The `updates` array should contain entries derived from ONLY ONE of the instructions above (Strict Command, NLP Command, OR Dialogue Analysis), whichever matched first.

5.  **Accuracy is Key:** Only report changes explicitly stated or directly implied. Do NOT infer. Resolve character names and item names as best as possible from context.
6.  **No Change:** If Instructions 1 & 2 didn't match, and Instruction 3 found no dialogue changes, output `{{"updates": []}}`.

**INPUTS:**

**USER QUERY (Check for commands first):**
{INVENTORY_UPDATE_QUERY_PLACEHOLDER}

**ASSISTANT RESPONSE (Analyze for dialogue changes if no command):**
---
{INVENTORY_UPDATE_RESPONSE_PLACEHOLDER}
---

**RECENT CHAT HISTORY (For context, especially pronoun/name resolution):**
---
{INVENTORY_UPDATE_HISTORY_PLACEHOLDER}
---

**OUTPUT (JSON object with detected inventory updates):**
"""

# === Guideline Constants (From Base) ===
SCENE_USAGE_GUIDELINE_TEXT = """<SceneUsageGuideline>
**Environment Awareness:** A static scene description is provided in each turn’s background context (e.g., stable, camp, tavern). This represents passive environmental conditions and available objects.

**Rules:**
1. You may reference or interact with items and features mentioned in the scene (e.g., “the lantern hanging above,” “the stool near the fire”) through NPC dialogue or Narrator tags.
2. Do not assume or invent objects or features not mentioned unless logically implied by the description.
3. Environmental details should influence tone, pacing, and behavior (e.g., quiet → hushed speech, rain → urgency).
4. Avoid re-describing the environment in detail. Instead, use it to guide NPC awareness and interaction choices.
5. If a scene implies limited light, space, or danger, adjust character actions accordingly.
</SceneUsageGuideline>"""

MEMORY_SUMMARY_STRUCTURE_GUIDELINE_TEXT = """<MemorySummaryStructureGuideline>
The background context may contain different types of memory summaries:
*   **Context Recaps (Aged):** These summarize *older* periods of dialogue, condensing previous T1 summaries. They are ordered **Newest Recap First** (the recap covering the most recent *older* period appears first).
*   **Recent Dialogue Summaries (T1):** These summarize *recent* chunks of conversation. They are ordered **Oldest Summary First** to provide a chronological flow leading up to the current turn.
*   **Related Older Information (T2 RAG):** These are retrieved based on relevance to the current query and may not follow a strict chronological order relative to T1/Aged.

Use these summaries collectively to understand the narrative timeline, recent events, and emotional progression.
</MemorySummaryStructureGuideline>"""

WEATHER_SUGGESTION_GUIDELINE_TEXT = """<WeatherSuggestionGuideline>
The background information may contain a "Proposed Weather Change: From X to Y". This indicates a potential shift in the environment suggested by the system. Treat this as context or inspiration. You are NOT required to follow this suggestion if your narrative or character actions dictate different weather. Feel free to describe the weather naturally as the scene unfolds.
</WeatherSuggestionGuideline>"""

# <<< NEW: Cache Maintainer Prompt Constants >>>
CACHE_MAINTAINER_QUERY_PLACEHOLDER = "{query}"
CACHE_MAINTAINER_HISTORY_PLACEHOLDER = "{recent_history_str}"
CACHE_MAINTAINER_PREVIOUS_CACHE_PLACEHOLDER = "{previous_cache_text}"
CACHE_MAINTAINER_CURRENT_OWI_PLACEHOLDER = "{current_owi_context}"
# Flag indicating no update is needed
NO_CACHE_UPDATE_FLAG = "[NO_CACHE_UPDATE]"

DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT = f"""
[[SYSTEM DIRECTIVE]]
**Role:** Roleplay Session Cache Maintainer
**Task:** Analyze the CURRENT OWI CONTEXT and compare it against the PREVIOUS CACHE TEXT in light of the LATEST USER QUERY and RECENT HISTORY. Decide if the cache needs updating.
**Objective:** Maintain a concise, relevant, and persistent cache of key background information (character states, relationships, critical facts, environment details) for the ongoing roleplay session. The cache should provide stable context unless significant new information relevant to the query/history appears in the CURRENT OWI CONTEXT.

**Sources:**
1.  **PREVIOUS CACHE TEXT:** The established background context from the last turn. This is the baseline.
2.  **CURRENT OWI CONTEXT:** New contextual information provided for *this* turn (potentially inconsistent or redundant).
3.  **LATEST USER QUERY:** The user's input driving the current interaction.
4.  **RECENT HISTORY:** The immediate dialogue leading up to the query.

**Decision Logic:**

1.  **Compare OWI to Cache:** Does the CURRENT OWI CONTEXT contain **significant new information** (facts, character status changes, major environmental shifts) that is **relevant** to the LATEST USER QUERY or RECENT HISTORY and is **missing or outdated** in the PREVIOUS CACHE TEXT?
    *   *Minor details, redundant info, or irrelevant OWI context should NOT trigger an update.*
    *   *Focus on information crucial for understanding the current turn or maintaining continuity.*

2.  **If NO Significant New Relevant Information:** Output the exact string:
    `{NO_CACHE_UPDATE_FLAG}`

3.  **If YES, Significant New Relevant Information Exists:** Synthesize a **complete, updated cache text**.
    *   **Start with the PREVIOUS CACHE TEXT.**
    *   **Integrate ONLY the *new, relevant* information** from the CURRENT OWI CONTEXT into the appropriate sections of the cache.
    *   **Correct or replace** outdated information in the cache with the new details from OWI.
    *   **Maintain the existing structure** (headings, sections) of the PREVIOUS CACHE TEXT as much as possible.
    *   **Ensure conciseness.** Do not simply append the entire OWI context. Filter and integrate judiciously.
    *   The output should be the **full, new cache text**, ready to replace the old one.

**INPUTS:**

**LATEST USER QUERY:**
{CACHE_MAINTAINER_QUERY_PLACEHOLDER}

**RECENT HISTORY:**
---
{CACHE_MAINTAINER_HISTORY_PLACEHOLDER}
---

**PREVIOUS CACHE TEXT (Baseline Context):**
---
{CACHE_MAINTAINER_PREVIOUS_CACHE_PLACEHOLDER}
---

**CURRENT OWI CONTEXT (Check for New Info):**
---
{CACHE_MAINTAINER_CURRENT_OWI_PLACEHOLDER}
---

**OUTPUT (Either '{NO_CACHE_UPDATE_FLAG}' or the complete updated cache text):**
"""
# <<< END NEW: Cache Maintainer Prompt Constants >>>

# === Function Implementations ---

# --- Function: Clean Context Tags (From Base) ---
def clean_context_tags(system_content: str) -> str:
    """Removes known context tags (<context>, <mempipe_*>) from the system prompt."""
    if not system_content or not isinstance(system_content, str): return ""
    cleaned = system_content
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = r"\s*" + re.escape(start_tag) + r".*?" + re.escape(end_tag) + r"\s*"
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Also remove the potentially misplaced user query block if found in system prompt text
    cleaned = re.sub(r"<user_query>.*?</user_query>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# --- Function: Process System Prompt (From Base) ---
def process_system_prompt(messages: List[Dict]) -> Tuple[str, Optional[str]]:
    """
    Extracts the base system prompt text and any OWI context block (<context>...</context>).
    Cleans known context tags from the base prompt.
    """
    func_logger = logging.getLogger(__name__ + '.process_system_prompt')
    original_system_prompt_content = ""
    extracted_owi_context = None
    base_system_prompt_text = "You are a helpful assistant." # Default
    if not isinstance(messages, list):
        func_logger.warning("Input 'messages' not a list. Returning default prompt.")
        return base_system_prompt_text, None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            original_system_prompt_content = msg.get("content", "")
            func_logger.debug(f"Found system prompt (len {len(original_system_prompt_content)}).")
            break
    if not original_system_prompt_content:
        func_logger.debug("No system message found in history.")
        return base_system_prompt_text, None
    owi_match = re.search(r"<context>(.*?)</context>", original_system_prompt_content, re.DOTALL | re.IGNORECASE)
    if owi_match:
        extracted_owi_context = owi_match.group(1).strip()
        func_logger.debug(f"Extracted OWI context (len {len(extracted_owi_context)}).")
    cleaned_base_prompt = clean_context_tags(original_system_prompt_content)
    if cleaned_base_prompt:
        base_system_prompt_text = cleaned_base_prompt
        func_logger.debug(f"Cleaned base system prompt text set (len {len(base_system_prompt_text)}).")
    else:
        func_logger.warning("System prompt empty after cleaning tags. Using default base text.")
        base_system_prompt_text = "You are a helpful assistant."
    return base_system_prompt_text, extracted_owi_context


# --- Function: Generate RAG Query (From Base) ---
async def generate_rag_query(
    latest_message_str: str,
    dialogue_context_str: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    api_url: str,
    api_key: str,
    temperature: float,
    caller_info: str = "i4_llm_agent_RAGQueryGen",
) -> Optional[str]:
    """Generates a RAG query using the default prompt."""
    # ** Start of previously truncated code **
    logger.debug(f"[{caller_info}] Generating RAG query using library default prompt...")
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        logger.error(f"[{caller_info}] Invalid llm_call_func.")
        return "[Error: Invalid LLM func]"
    if not api_url or not api_key:
        logger.error(f"[{caller_info}] Missing RAGQ URL/Key.")
        return "[Error: Missing RAGQ URL/Key]"
    if DEFAULT_RAGQ_LLM_PROMPT == "[Prompting Const Load Error]": # Check base constant
        logger.error(f"[{caller_info}] Library default RAGQ prompt constant failed to load.")
        return "[Error: Default RAGQ prompt missing]"

    ragq_prompt_text = None
    formatting_error = None
    # Safely handle potential braces in input text for .format()
    safe_latest_message = latest_message_str.replace("{", "{{").replace("}", "}}") if isinstance(latest_message_str, str) else "[No message]"
    safe_dialogue_context = dialogue_context_str.replace("{", "{{").replace("}", "}}") if isinstance(dialogue_context_str, str) else "[No history]"

    try:
        ragq_prompt_text = DEFAULT_RAGQ_LLM_PROMPT.format(
            latest_message=safe_latest_message,
            dialogue_context=safe_dialogue_context
        )
        if not ragq_prompt_text or not ragq_prompt_text.strip():
            formatting_error = "[Error: Formatted prompt is empty]"
            logger.error(f"[{caller_info}] RAGQ prompt formatting resulted in empty string.")
    except KeyError as e:
        formatting_error = f"[Error: RAGQ key error: {e}]"
        logger.error(f"[{caller_info}] RAGQ prompt key error: {e}.")
    except Exception as e_fmt:
        formatting_error = f"[Error: RAGQ format failed ({type(e_fmt).__name__})]"
        logger.error(f"[{caller_info}] Failed format RAGQ prompt: {e_fmt}", exc_info=True)

    if formatting_error: return formatting_error

    ragq_payload = {"contents": [{"parts": [{"text": ragq_prompt_text}]}]}
    logger.info(f"[{caller_info}] Calling LLM for RAG query generation...")

    try:
        success, response_or_error = await llm_call_func(
            api_url=api_url,
            api_key=api_key,
            payload=ragq_payload,
            temperature=temperature,
            timeout=45, # Short timeout for query generation
            caller_info=caller_info,
        )
    except Exception as e_call:
        logger.error(f"[{caller_info}] Exception calling LLM wrapper for RAGQ: {e_call}", exc_info=True)
        success = False
        response_or_error = f"[Error: LLM Call Exception {type(e_call).__name__}]"

    if success and isinstance(response_or_error, str):
        final_query = response_or_error.strip()
        if final_query:
            logger.info(f"[{caller_info}] Generated RAG query: '{final_query}'")
            return final_query
        else:
            logger.warning(f"[{caller_info}] RAGQ LLM returned empty string.")
            return "[Error: RAGQ empty]"
    else:
        error_msg = str(response_or_error)
        logger.error(f"[{caller_info}] RAGQ failed: {error_msg}")
        if isinstance(response_or_error, dict):
            err_type = response_or_error.get('error_type', 'RAGQ Err')
            err_msg_detail = response_or_error.get('message', 'Unknown')
            return f"[Error: {err_type} - {err_msg_detail}]"
        else:
            return f"[Error: RAGQ Failed - {error_msg[:50]}]"
    # ** End of previously truncated code **


# --- Function: Combine Background Context (From Base) ---
def combine_background_context(
    final_selected_context: Optional[str], # This is just the extracted OWI context in base
    t1_summaries: Optional[List[Tuple[str, Dict[str, Any]]]],
    aged_summaries: Optional[List[Tuple[str, Dict[str, Any]]]],
    t2_rag_results: Optional[List[str]],
    scene_description: Optional[str] = None,
    inventory_context: Optional[str] = None,
    current_day: Optional[int] = None,
    current_time_of_day: Optional[str] = None,
    current_season: Optional[str] = None,
    current_weather: Optional[str] = None,
    weather_proposal: Optional[Dict[str, Optional[str]]] = None,
    labels: Optional[Dict[str, str]] = None # Labels no longer used
) -> str:
    """
    Combines various background context sources into a single XML-style formatted string.
    Prepends static guidelines (Scene Usage, Memory Structure, Weather Suggestion).
    Sorts Aged summaries Newest First, T1 summaries Oldest First.
    """
    # ... (Implementation remains the same as base) ...
    func_logger = logging.getLogger(__name__ + '.combine_background_context')
    context_parts = []
    context_parts.append("<SystemContextGuidelines>")
    context_parts.append(SCENE_USAGE_GUIDELINE_TEXT)
    context_parts.append(MEMORY_SUMMARY_STRUCTURE_GUIDELINE_TEXT)
    context_parts.append(WEATHER_SUGGESTION_GUIDELINE_TEXT)
    context_parts.append("</SystemContextGuidelines>")
    func_logger.debug("Prepended static context guidelines.")
    world_state_xml_parts = []
    if isinstance(current_day, int) and current_day > 0: world_state_xml_parts.append(f"<Day>{current_day}</Day>")
    if isinstance(current_time_of_day, str) and current_time_of_day.strip() and "Unknown" not in current_time_of_day: world_state_xml_parts.append(f"<Time>{current_time_of_day.strip()}</Time>")
    if isinstance(current_season, str) and current_season.strip() and "Unknown" not in current_season: world_state_xml_parts.append(f"<Season>{current_season.strip()}</Season>")
    if isinstance(current_weather, str) and current_weather.strip() and "Unknown" not in current_weather: world_state_xml_parts.append(f"<Weather>{current_weather.strip()}</Weather>")
    if world_state_xml_parts:
        context_parts.append("<WorldState>")
        context_parts.extend(world_state_xml_parts)
        context_parts.append("</WorldState>")
        func_logger.debug(f"Adding World State section: {len(world_state_xml_parts)} parts.")
    if isinstance(weather_proposal, dict):
        prev_w = weather_proposal.get("previous_weather"); new_w = weather_proposal.get("new_weather")
        if isinstance(prev_w, str) and isinstance(new_w, str):
            proposal_string = f"From '{prev_w}' to '{new_w}'"
            context_parts.append(f"<WeatherProposal>{proposal_string}</WeatherProposal>")
            func_logger.debug(f"Adding Weather Proposal section: {proposal_string}")
    safe_scene_description = scene_description.strip() if isinstance(scene_description, str) else None
    if safe_scene_description:
        func_logger.debug(f"Adding scene description (len: {len(safe_scene_description)}).")
        escaped_scene = safe_scene_description.replace('&', '&').replace('<', '<').replace('>', '>')
        context_parts.append(f"<CurrentScene>{escaped_scene}</CurrentScene>")
    safe_inventory_context = inventory_context.strip() if isinstance(inventory_context, str) else None
    is_valid_inventory = safe_inventory_context and not re.search(r"\[(No Inventory|Error|Disabled)\]", safe_inventory_context, re.IGNORECASE)
    if is_valid_inventory:
        func_logger.debug(f"Adding inventory context (len: {len(safe_inventory_context)}).")
        escaped_inventory = safe_inventory_context.replace('&', '&').replace('<', '<').replace('>', '>')
        context_parts.append(f"<Inventories>{escaped_inventory}</Inventories>")
    safe_selected_context = final_selected_context.strip() if isinstance(final_selected_context, str) else None
    if safe_selected_context:
        func_logger.debug(f"Adding raw OWI context (len: {len(safe_selected_context)}).")
        escaped_selected = safe_selected_context.replace('&', '&').replace('<', '<').replace('>', '>')
        context_parts.append(f"<SelectedContext source='OWI_Direct'>{escaped_selected}</SelectedContext>")
    valid_aged_summaries_data = []
    if aged_summaries and isinstance(aged_summaries, list):
        try:
            aged_summaries.sort(key=lambda item: item[1].get('creation_timestamp_utc', 0), reverse=True)
            valid_aged_summaries_data = [item for item in aged_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]
        except Exception as e_sort_aged:
            func_logger.error(f"Error sorting aged summaries: {e_sort_aged}. Using original order.")
            valid_aged_summaries_data = [item for item in aged_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]
    if valid_aged_summaries_data:
        context_parts.append('<AgedSummaries order="newest_recap_first">')
        for summary_text, metadata in valid_aged_summaries_data:
             escaped_summary = summary_text.replace('&', '&').replace('<', '<').replace('>', '>')
             context_parts.append(f"<Summary>{escaped_summary}</Summary>")
        context_parts.append('</AgedSummaries>')
        func_logger.debug(f"Adding {len(valid_aged_summaries_data)} Aged summaries.")
    valid_t1_summaries_data = []
    if t1_summaries and isinstance(t1_summaries, list):
        try:
            t1_summaries.sort(key=lambda item: item[1].get('turn_end_index', -1), reverse=False)
            valid_t1_summaries_data = [item for item in t1_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]
        except Exception as e_sort_t1:
            func_logger.error(f"Error sorting T1 summaries: {e_sort_t1}. Using original order.")
            valid_t1_summaries_data = [item for item in t1_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]
    if valid_t1_summaries_data:
        context_parts.append('<RecentSummaries order="oldest_dialogue_first">')
        for summary_text, metadata in valid_t1_summaries_data:
             escaped_summary = summary_text.replace('&', '&').replace('<', '<').replace('>', '>')
             context_parts.append(f"<Summary>{escaped_summary}</Summary>")
        context_parts.append('</RecentSummaries>')
        func_logger.debug(f"Adding {len(valid_t1_summaries_data)} T1 summaries.")
    valid_t2_results = [s.strip() for s in t2_rag_results if isinstance(s, str) and s.strip()] if t2_rag_results else []
    if valid_t2_results:
        context_parts.append('<RelatedInformation source="T2_RAG">')
        for result_text in valid_t2_results:
             escaped_result = result_text.replace('&', '&').replace('<', '<').replace('>', '>')
             context_parts.append(f"<Info>{escaped_result}</Info>")
        context_parts.append('</RelatedInformation>')
        func_logger.debug(f"Adding {len(valid_t2_results)} T2 RAG results.")
    if len(context_parts) <= 5:
        func_logger.info("No actual background context available beyond guidelines.")
        return EMPTY_CONTEXT_PLACEHOLDER
    else:
        full_context_string = "\n".join(context_parts)
        func_logger.info(f"Combined context created (Total len: {len(full_context_string)}).")
        return full_context_string


# --- Function: Construct Final LLM Payload (From Base) ---
def construct_final_llm_payload(
    system_prompt: str, # Base system prompt text (already cleaned)
    history: List[Dict], # Dialogue history turns (user/model)
    context: Optional[str], # Formatted background context string (XML-style, includes guidelines)
    query: str, # Final user query text for this turn
    long_term_goal: Optional[str] = None, # Dynamic goal text
    event_hint: Optional[str] = None, # Dynamic event hint text
    period_setting: Optional[str] = None, # Dynamic period setting text
    include_ack_turns: bool = True # Whether to include ACK turns
) -> Dict[str, Any]:
    """
    Constructs the final payload for the LLM in Google's 'contents' format,
    following the structure: System Instructions -> Background Context -> History -> Query.
    Injects dynamic guidelines (Goal, Event, Period) into System Instructions.
    Assumes Background Context string already contains static guidelines. BASE VERSION.
    """
    # ... (Implementation remains the same as base) ...
    func_logger = logging.getLogger(__name__ + '.construct_final_llm_payload')
    func_logger.debug(
        f"Constructing final LLM payload (Base Structure). ACKs: {include_ack_turns}, "
        f"Goal Provided: {bool(long_term_goal)}, Event Hint Provided: {bool(event_hint)}, "
        f"Period Setting Provided: '{period_setting or 'None'}'"
    )
    gemini_contents = []
    base_system_prompt_text = system_prompt.strip() if system_prompt else "You are a helpful assistant."
    final_system_instructions = base_system_prompt_text
    safe_long_term_goal = long_term_goal.strip() if isinstance(long_term_goal, str) else None
    if safe_long_term_goal:
        goal_handling_guideline = (
             "This is the persistent, overarching goal guiding the direction of the current session. "
             "**There is no specific deadline or requirement to achieve this goal within a short timeframe; focus on gradual progress and ensuring actions/dialogue remain coherent with this long-term objective.** "
             "Evaluate NPC actions, dialogue, and narrative developments against this objective. Ensure they generally align with or progress towards achieving this goal, "
             "unless immediate context or character realism strongly necessitates a temporary deviation. This goal remains active until explicitly changed in the session settings."
        )
        goal_block = f"""

--- [ SESSION GOAL ] ---

**Objective:**
{safe_long_term_goal}

**Handling Guideline:**
{goal_handling_guideline}

--- [ END SESSION GOAL ] ---"""
        final_system_instructions += goal_block
        func_logger.debug(f"Appended long term goal to system instructions text.")
    if event_hint and isinstance(event_hint, str) and event_hint.strip():
        if EVENT_HANDLING_GUIDELINE_TEXT != "[EVENT GUIDELINE LOAD FAILED]":
            final_system_instructions += f"\n{EVENT_HANDLING_GUIDELINE_TEXT}"
            func_logger.debug(f"Appended event handling guideline to system instructions text.")
        else:
            func_logger.warning("Event hint present but guideline text failed to load. Skipping append.")
    safe_period_setting = period_setting.strip() if isinstance(period_setting, str) else None
    if safe_period_setting:
        period_block = f"""

--- [ Period Setting ] ---
[[Setting Instruction: Generate content appropriate for a '{safe_period_setting}' setting.]]
--- [ END Period Setting ] ---"""
        final_system_instructions += period_block
        func_logger.debug(f"Appended period setting instruction ('{safe_period_setting}') to system instructions text.")
    system_instructions_turn = None; system_ack_turn = None
    if final_system_instructions:
        system_instructions_turn = {"role": "user", "parts": [{"text": f"System Instructions:\n{final_system_instructions}"}]}
        if include_ack_turns:
            ack_text = "Understood. I will follow these instructions."
            if safe_long_term_goal: ack_text += " I will also keep the long-term goal in mind."
            if safe_period_setting: ack_text += f" I will also maintain a '{safe_period_setting}' setting."
            system_ack_turn = {"role": "model", "parts": [{"text": ack_text}]}
    context_turn = None; context_ack_turn = None
    has_real_context = bool(context and context.strip() and not context.strip().startswith("<Context type='Empty'>"))
    if has_real_context:
        context_injection_text = f"Background Information (Use this to inform your response):\n{context.strip()}"
        context_turn = {"role": "user", "parts": [{"text": context_injection_text}]}
        if include_ack_turns: context_ack_turn = {"role": "model", "parts": [{"text": "Understood. I have reviewed the background information."}]}
    history_turns = []
    for msg in history:
        role = msg.get("role"); content = msg.get("content", "").strip()
        if role == "user" and content: history_turns.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant" and content: history_turns.append({"role": "model", "parts": [{"text": content}]})
        elif role == "model" and content: history_turns.append({"role": "model", "parts": [{"text": content}]})
    safe_query = query.strip() if query and query.strip() else "[User query not provided]"
    final_query_text = safe_query
    if event_hint and isinstance(event_hint, str) and event_hint.strip():
        formatted_hint = format_hint_for_query(event_hint)
        if formatted_hint:
            final_query_text = f"{formatted_hint}\n\n{safe_query}"
            func_logger.debug(f"Prepended event hint to final query text.")
    final_query_turn = {"role": "user", "parts": [{"text": final_query_text}]}
    if system_instructions_turn: gemini_contents.append(system_instructions_turn)
    if system_ack_turn: gemini_contents.append(system_ack_turn)
    if context_turn: gemini_contents.append(context_turn)
    if context_ack_turn: gemini_contents.append(context_ack_turn)
    gemini_contents.extend(history_turns)
    gemini_contents.append(final_query_turn)
    final_payload = {"contents": gemini_contents}
    func_logger.info(f"Final payload constructed with {len(gemini_contents)} turns (Base structure).")
    return final_payload


# --- Function: Format Memory Aging Prompt (From Base) ---
def format_memory_aging_prompt(t1_batch_text: str, template: Optional[str] = None) -> str:
    """Formats the prompt for the Memory Aging LLM."""
    # ... (Implementation remains the same as base) ...
    func_logger = logging.getLogger(__name__ + '.format_memory_aging_prompt')
    prompt_template = template if template is not None else DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE
    if not prompt_template or prompt_template == "[Default Memory Aging Prompt Load Failed]":
        return "[Error: Invalid or Missing Template for Memory Aging]"
    safe_batch_text = t1_batch_text.replace("{", "{{").replace("}", "}}") if isinstance(t1_batch_text, str) else ""
    try:
        formatted_prompt = prompt_template.format(
            **{MEMORY_AGING_BATCH_PLACEHOLDER.strip('{}'): safe_batch_text}
        )
        return formatted_prompt
    except KeyError as e:
        func_logger.error(f"Missing placeholder in memory aging prompt: {e}")
        return f"[Error: Missing placeholder '{e}']"
    except Exception as e:
        func_logger.error(f"Error formatting memory aging prompt: {e}", exc_info=True)
        return f"[Error formatting memory aging prompt: {type(e).__name__}]"


# --- Function: Format Inventory Update Prompt (From Base) ---
def format_inventory_update_prompt(
    main_llm_response: str,
    user_query: str,
    recent_history_str: str,
    template: str # Expecting the specific template for this step
) -> str:
    """Formats the prompt for the Inventory Update LLM."""
    # ... (Implementation remains the same as base) ...
    func_logger = logging.getLogger(__name__ + '.format_inventory_update_prompt')
    if not template or not isinstance(template, str) or template == "[Default Inventory Prompt Load Failed]":
        return "[Error: Invalid Template for Inventory Update]"
    try:
        formatted_prompt = template.replace(INVENTORY_UPDATE_RESPONSE_PLACEHOLDER, str(main_llm_response))
        formatted_prompt = formatted_prompt.replace(INVENTORY_UPDATE_QUERY_PLACEHOLDER, str(user_query))
        formatted_prompt = formatted_prompt.replace(INVENTORY_UPDATE_HISTORY_PLACEHOLDER, str(recent_history_str))
        return formatted_prompt
    except Exception as e:
        func_logger.error(f"Error formatting inventory update prompt: {e}", exc_info=True)
        return f"[Error formatting inventory update prompt: {type(e).__name__}]"


# <<< NEW: Cache Maintainer Formatting Function >>>
def format_cache_maintainer_prompt(
    query: str,
    recent_history_str: str,
    previous_cache_text: str,
    current_owi_context: str,
    template: Optional[str] = None
) -> str:
    """Formats the prompt for the Cache Maintainer LLM."""
    func_logger = logging.getLogger(__name__ + '.format_cache_maintainer_prompt')
    prompt_template = template if template is not None else DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT

    # Use the constant added earlier
    if not prompt_template or prompt_template == "[Prompting Const Load Error]":
         # Log an error if the default template constant is missing
         func_logger.error("Default Cache Maintainer template text is missing or failed to load.")
         # Return a more specific error message
         return "[Error: Invalid or Missing Template for Cache Maintainer]"

    # Escape braces in the input data to prevent conflicts with .format()
    safe_query = query.replace("{", "{{").replace("}", "}}") if isinstance(query, str) else ""
    safe_history = recent_history_str.replace("{", "{{").replace("}", "}}") if isinstance(recent_history_str, str) else "[No History]"
    safe_previous_cache = previous_cache_text.replace("{", "{{").replace("}", "}}") if isinstance(previous_cache_text, str) else "[No Previous Cache]"
    safe_current_owi = current_owi_context.replace("{", "{{").replace("}", "}}") if isinstance(current_owi_context, str) else "[No OWI Context]"

    try:
        # Use .format() with placeholder names stripped of braces
        formatted_prompt = prompt_template.format(
            **{
               CACHE_MAINTAINER_QUERY_PLACEHOLDER.strip('{}'): safe_query,
               CACHE_MAINTAINER_HISTORY_PLACEHOLDER.strip('{}'): safe_history,
               CACHE_MAINTAINER_PREVIOUS_CACHE_PLACEHOLDER.strip('{}'): safe_previous_cache,
               CACHE_MAINTAINER_CURRENT_OWI_PLACEHOLDER.strip('{}'): safe_current_owi
               # NO_CACHE_UPDATE_FLAG is part of the literal template text, not formatted in
            }
        )
        # Replace the flag placeholder literally within the formatted string
        # This ensures the exact flag string is present for comparison later
        formatted_prompt = formatted_prompt.replace("{NO_CACHE_UPDATE_FLAG}", NO_CACHE_UPDATE_FLAG)

        return formatted_prompt
    except KeyError as e:
        func_logger.error(f"Missing placeholder in cache maintainer prompt: {e}")
        return f"[Error: Missing placeholder '{e}']"
    except Exception as e:
        func_logger.error(f"Error formatting cache maintainer prompt: {e}", exc_info=True)
        return f"[Error formatting cache maintainer prompt: {type(e).__name__}]"
# <<< END NEW: Cache Maintainer Formatting Function >>>


# --- Legacy/Stub Functions (From Base) ---
def assemble_tagged_context(base_prompt: str, contexts: Dict[str, Union[str, List[str]]]) -> str:
    """Simplified stub for assembling context with old-style tags. May not function correctly."""
    # ... (Implementation remains the same as base) ...
    logger.warning("assemble_tagged_context is a simplified stub and may not function as originally intended.")
    full_context = base_prompt
    for key, value in contexts.items():
        if isinstance(value, list): value = "\n".join(value)
        if value and isinstance(value, str) and value.strip():
            start_tag, end_tag = KNOWN_CONTEXT_TAGS.get(key, (f"<{key}>", f"</{key}>"))
            full_context += f"\n{start_tag}\n{value.strip()}\n{end_tag}\n"
    return full_context

def extract_tagged_context(system_content: str) -> Dict[str, str]:
    """Simplified stub for extracting context with old-style tags. May not function correctly."""
    # ... (Implementation remains the same as base) ...
    logger.warning("extract_tagged_context is a simplified stub and may not function as originally intended.")
    extracted = {}
    if not system_content or not isinstance(system_content, str): return extracted
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        match = re.search(pattern, system_content, re.DOTALL | re.IGNORECASE)
        if match: extracted[key] = match.group(1).strip()
    return extracted

# === END COMPLETE CORRECTED BASE FILE: i4_llm_agent/prompting.py ===