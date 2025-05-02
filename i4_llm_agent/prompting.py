# === START CORRECTED FILE: i4_llm_agent/prompting.py ===
# i4_llm_agent/prompting.py

import logging
import re
import asyncio
import json # Added for formatting scene keywords example
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

# --- Constants for Context Tags (NOW XML-STYLE focused) ---
# KNOWN_CONTEXT_TAGS are less relevant now with XML structure inside the context block
KNOWN_CONTEXT_TAGS = { # Kept for cleaning old formats if necessary
    "owi": ("<context>", "</context>"),
    "t1": ("<mempipe_recent_summary>", "</mempipe_recent_summary>"),
    "t2_rag": ("<mempipe_rag_result>", "</mempipe_rag_result>"),
}
# TAG_LABELS are also less relevant as XML tags are self-descriptive
TAG_LABELS = {} # No longer used by combine_background_context
EMPTY_CONTEXT_PLACEHOLDER = "<Context type='Empty'>[No Background Information Available]</Context>"


# === NEW: Summarizer Prompt Constants (Moved from script.txt - RESTORED) ===
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
# === END NEW Summarizer Constants ===


# === NEW: Memory Aging Prompt Constants (RESTORED) ===
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
# === END NEW Memory Aging Constants ===


# === NEW: RAG Query Prompt Constant (RESTORED - Although unchanged) ===
DEFAULT_RAGQ_LLM_PROMPT = """Based on the latest user message and recent dialogue context, generate a concise search query focusing on the key entities, topics, or questions raised.

Latest Message: {latest_message}

Dialogue Context:
{dialogue_context}

Search Query:"""
# === END NEW RAG Query Constant ===


# --- Constants for Stateless Refiner (RESTORED) ---
STATELESS_REFINER_QUERY_PLACEHOLDER = "{query}"
STATELESS_REFINER_CONTEXT_PLACEHOLDER = "{external_context}"
STATELESS_REFINER_HISTORY_PLACEHOLDER = "{recent_history_str}"
DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE = f"""
[[SYSTEM DIRECTIVE]]
**Role:** Roleplay Context Extractor
**Task:** Analyze the provided CONTEXT DOCUMENTS (character backstories, relationship histories, past events, lore) and the RECENT CHAT HISTORY (dialogue, actions, emotional expressions).
**Objective:** Based ONLY on this information, extract and describe the specific details, memories, relationship dynamics, stated feelings, significant past events, or relevant character traits that are **essential for understanding the full context** of and accurately answering the LATEST USER QUERY from a roleplaying perspective.
**Instructions:**
1.  Identify the core subject of the LATEST USER QUERY and any immediately related contextual elements.
2.  Extract Key Information: Prioritize extracting verbatim sentences or short passages that **directly address** the core subject and related elements.
3.  Describe Key Dynamics: ...extract specific details or events... that illustrate *why* it's complex...
4.  Include Foundational Context: Extract specific details... that **directly led to or fundamentally define** the current situation...
5.  Incorporate Recent Developments: Include details from the RECENT CHAT HISTORY...
6.  Be Descriptive but Focused: Capture the nuance... Avoid overly generic summaries...
7.  Prioritize Relevance over Extreme Brevity: ...ensure that key descriptive details... are included...
8.  Ensure Accuracy: Do not infer, assume, or add information not explicitly present...
9.  Output: Present the extracted points clearly. If no relevant information is found, state clearly: \"No specific details relevant to the query were found in the provided context.\"

**LATEST USER QUERY:** {STATELESS_REFINER_QUERY_PLACEHOLDER}
**CONTEXT DOCUMENTS:**
---
{STATELESS_REFINER_CONTEXT_PLACEHOLDER}
---
**RECENT CHAT HISTORY:**
---
{STATELESS_REFINER_HISTORY_PLACEHOLDER}
---

Concise Relevant Information (for final answer generation):
"""

# --- Constants for Two-Step RAG Cache Refinement (RESTORED) ---
CACHE_UPDATE_QUERY_PLACEHOLDER = "{query}"
CACHE_UPDATE_CURRENT_OWI_PLACEHOLDER = "{current_owi_rag}"
CACHE_UPDATE_PREVIOUS_CACHE_PLACEHOLDER = "{previous_cache}"
CACHE_UPDATE_HISTORY_PLACEHOLDER = "{recent_history_str}"
# Default prompt template for Step 1: Cache Update
DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT = f"""
[[SYSTEM DIRECTIVE]]
**Role:** Session Background Cache Maintainer
**Task:** Intelligently update the SESSION CACHE using relevant information from the CURRENT OWI RETRIEVAL. Prioritize maintaining accurate core character profiles and lore while integrating **new facts, significant details, clarifications, or elaborations**.
**Objective:** Maintain an accurate and structured cache for long-term context, balancing stability with incorporating meaningful updates from OWI.

**Inputs:**
- LATEST USER QUERY (for context)
- CURRENT OWI RETRIEVAL (source of potential new info, details, profiles)
- PREVIOUSLY REFINED CACHE (base for update)
- RECENT CHAT HISTORY (for context only)

**Core Instructions:**

1.  **Identify & Update/Preserve Character Profiles:**
    *   Scan CURRENT OWI RETRIEVAL for `character_profile` documents or significant descriptive passages about characters.
    *   Scan PREVIOUSLY REFINED CACHE for existing character profile summaries (e.g., under `# Character: Name` headings).
    *   **Action:** If a character profile exists in PREVIOUS CACHE:
        *   **KEEP** the core identity and established traits.
        *   **MERGE/ADD** concise, relevant new details, clarifications, or significant trait elaborations found in CURRENT OWI RETRIEVAL into the existing profile summary. Do not drastically alter core traits based only on RECENT CHAT HISTORY.
        *   Only *replace* major parts of a profile if CURRENT OWI provides clearly contradictory or superseding *factual* information (not just nuanced descriptions).
    *   **Action:** If a NEW profile is found in CURRENT OWI for a character NOT in PREVIOUS CACHE, summarize its essential details (Identity, Traits, Role) and ADD it to the output under a new heading.

2.  **Integrate NEW or Elaborated Factual Lore:**
    *   Scan CURRENT OWI RETRIEVAL (excluding profiles processed above) for background facts (lore, world details, established events, locations).
    *   **Action:** Compare these facts against the PREVIOUSLY REFINED CACHE.
        *   If a fact is genuinely **NEW** and relevant, ADD it.
        *   If a fact **ELABORATES significantly** on, **CLARIFIES**, or provides important **new DETAILS** for an existing topic in the cache, integrate these details concisely into the relevant section.
        *   If a fact provides a clear **CORRECTION/UPDATE** to existing lore, MODIFY the cache accordingly.
        *   Do NOT add minor rephrasing or clearly redundant facts.

3.  **Minimal Pruning:**
    *   **Action:** Only remove sections from the PREVIOUS CACHE if they are *explicitly contradicted* by newer info in CURRENT OWI or are clearly no longer relevant (e.g., a character definitively removed from the story). **Default to keeping existing information unless strong evidence dictates removal.**

4.  **Use Query/History for CONTEXT ONLY:** Use LATEST USER QUERY and RECENT CHAT HISTORY primarily to understand focus and relevance when deciding *what information* from OWI to add/update/elaborate on. **DO NOT summarize the history itself in the cache.**

5.  **Output Format:**
    *   Produce the complete, updated SESSION CACHE text.
    *   **Maintain Structure:** Use clear headings (e.g., `# Character: Name`, `# Lore: Topic`). Preserve existing headings/structure where possible.
    *   **No Change:** If analysis shows no significant additions/updates/removals/elaborations are needed based on the OWI input, output ONLY the exact text: `[NO_CACHE_UPDATE]`
    *   **Empty/Irrelevant:** If PREVIOUS CACHE was empty and CURRENT OWI contains no relevant profiles or facts, output: `[No relevant background context found]`

**INPUTS:**

**LATEST USER QUERY:**
{CACHE_UPDATE_QUERY_PLACEHOLDER}

**CURRENT OWI RETRIEVAL:**
---
{CACHE_UPDATE_CURRENT_OWI_PLACEHOLDER}
---

**PREVIOUSLY REFINED CACHE:**
---
{CACHE_UPDATE_PREVIOUS_CACHE_PLACEHOLDER}
---

**RECENT CHAT HISTORY:**
---
{CACHE_UPDATE_HISTORY_PLACEHOLDER}
---

**OUTPUT (Updated Session Cache Text - Structured, or [NO_CACHE_UPDATE], or [No relevant background context found]):**
"""

# Final Context Selector Prompt Template (v1.1 - RESTORED)
FINAL_SELECT_QUERY_PLACEHOLDER = "{query}"
FINAL_SELECT_UPDATED_CACHE_PLACEHOLDER = "{updated_cache}"
FINAL_SELECT_CURRENT_OWI_PLACEHOLDER = "{current_owi_rag}"
FINAL_SELECT_HISTORY_PLACEHOLDER = "{recent_history_str}"

DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT = f"""
[[SYSTEM DIRECTIVE]]
**Role:** Context Selector for Roleplay Response
**Task:** Analyze available background sources (CACHE, OWI, HISTORY) and select details **relevant and helpful** for generating the *next* narrative response based on the LATEST USER QUERY and RECENT HISTORY.
**Objective:** Provide **sufficient background context** from the Cache and OWI Retrieval to enable a coherent, nuanced, and contextually grounded response, without overwhelming the final LLM with irrelevant data.

**Sources:**
1.  **UPDATED SESSION CACHE:** Long-term facts, character profiles, established lore. (Primary Source)
2.  **CURRENT OWI RETRIEVAL:** General contextual information provided for the current turn. (Secondary Source)
3.  **RECENT CHAT HISTORY:** Immediate conversational context (dialogue, actions, involved characters).
4.  **LATEST USER QUERY:** The user's specific input for this turn.

**Instructions:**

1.  **Analyze Query & History:** Determine the core subject, actions, **locations**, and **characters actively or passively present** in the LATEST USER QUERY and the last 2–3 turns of RECENT CHAT HISTORY. Include unresolved threads referenced implicitly (e.g., emotional fallout, mentioned past decisions).

2.  **Select Helpful Cache/OWI Context:** Examine the CACHE and OWI RETRIEVAL. Extract only sentences/passages that:
    * Explain the **current situation** or **immediate environment**.
    * Provide insight into the **motivations, relationships, or core traits** of involved or emotionally connected characters.
    * Offer relevant **background lore** about current locations, objects, or unresolved past events that inform the *current* moment.

3.  **Relational Awareness Rule:** If a character is not directly mentioned but is **strongly emotionally tied** to a currently involved character (e.g., a sister, parent, lost companion), include minimal relevant memory snippets for continuity.
    *Example: If Emily is active, and her sister Julia is emotionally tied to her current mood or goal, include brief but relevant Julia context—even if Julia is not present or named.*

4.  **Emotional Theme Continuity (Optional):** If the query implies a recurring emotional thread (fear, guilt, protection), select context that reinforces that tone, even if the source isn't directly mentioned.

5.  **Prioritize CACHE Over OWI:** CACHE represents canonical memory and character development. Use OWI RETRIEVAL to fill in situational details not found in cache, or provide short reminders about location and setting.

6.  **Filter Irrelevant Content:** Exclude long lore blocks, full profiles, or unrelated past facts. Favor **compressed, emotionally relevant facts** that reinforce the present turn.

7.  **Assemble Output:** Combine the selected Cache/OWI context snippets into a short, clean block. Use optional headings (e.g., `=== Character Note: Julia ===`, `=== Location Memory ===`) to group content.
    - Avoid excessive length.
    - Use only what’s helpful for this turn.

8.  **Empty Fallback:** If no relevant Cache/OWI content is found, respond with:
    `[No relevant background context found for the current query]`

**INPUTS:**

**LATEST USER QUERY:**
{FINAL_SELECT_QUERY_PLACEHOLDER}

**UPDATED SESSION CACHE (Primary Source - Long-Term Facts/Profiles):**
---
{FINAL_SELECT_UPDATED_CACHE_PLACEHOLDER}
---

**CURRENT OWI RETRIEVAL (Secondary Source - General Context):**
---
{FINAL_SELECT_CURRENT_OWI_PLACEHOLDER}
---

**RECENT CHAT HISTORY (for relevance & involved characters):**
---
{FINAL_SELECT_HISTORY_PLACEHOLDER}
---

**OUTPUT (Selected Relevant Cache/OWI Snippets):**
"""

# Placeholders for Inventory Update LLM (RESTORED)
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

# <<< NEW/MOVED CONSTANTS FOR GUIDELINES >>>
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
# <<< END NEW/MOVED CONSTANTS >>>


# --- Function Implementations ---

# --- Function: Clean Context Tags (Existing - Unchanged from latest) ---
def clean_context_tags(system_content: str) -> str:
    # This function might need adjustment if old formats with <context> tags are still possible
    # For now, keep it as is to handle potential legacy formats in the base prompt.
    if not system_content or not isinstance(system_content, str): return ""
    cleaned = system_content
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = r"\s*" + re.escape(start_tag) + r".*?" + re.escape(end_tag) + r"\s*"
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Also remove the potentially misplaced user query block if found in system prompt text
    cleaned = re.sub(r"<user_query>.*?</user_query>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# --- Function: Process System Prompt (Modified - Enhanced Cleaning from latest) ---
def process_system_prompt(messages: List[Dict]) -> Tuple[str, Optional[str]]:
    """
    Extracts the base system prompt text and any OWI context block.
    Cleans known context tags AND the misplaced user_query tag from the base prompt.
    """
    func_logger = logging.getLogger(__name__ + '.process_system_prompt')
    original_system_prompt_content = ""
    extracted_owi_context = None
    base_system_prompt_text = "You are a helpful assistant." # Default

    if not isinstance(messages, list):
        func_logger.warning("Input 'messages' not a list. Returning default prompt.")
        return base_system_prompt_text, None

    # Find the system message
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            original_system_prompt_content = msg.get("content", "")
            func_logger.debug(f"Found system prompt (len {len(original_system_prompt_content)}).")
            break # Assume only one system message

    if not original_system_prompt_content:
        func_logger.debug("No system message found in history.")
        return base_system_prompt_text, None # Return default if no system message

    # Extract OWI context block (<context>...</context>) if present
    owi_match = re.search(r"<context>(.*?)</context>", original_system_prompt_content, re.DOTALL | re.IGNORECASE)
    if owi_match:
        extracted_owi_context = owi_match.group(1).strip()
        func_logger.debug(f"Extracted OWI context (len {len(extracted_owi_context)}).")

    # Clean known tags AND the misplaced <user_query> tag from the system prompt content
    cleaned_base_prompt = clean_context_tags(original_system_prompt_content)

    if cleaned_base_prompt:
        base_system_prompt_text = cleaned_base_prompt
        func_logger.debug(f"Cleaned base system prompt text set (len {len(base_system_prompt_text)}).")
    else:
        # If cleaning resulted in an empty string, use the default.
        func_logger.warning("System prompt empty after cleaning tags. Using default base text.")
        base_system_prompt_text = "You are a helpful assistant."

    return base_system_prompt_text, extracted_owi_context

# --- Function: Format Stateless Refiner Prompt (Existing - Unchanged from latest) ---
def format_stateless_refiner_prompt(external_context: str, recent_history_str: str, query: str, template: Optional[str] = None) -> str:
    func_logger = logging.getLogger(__name__ + '.format_stateless_refiner_prompt')
    prompt_template = template if template is not None else DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE
    safe_context = external_context.replace("{", "{{").replace("}", "}}") if isinstance(external_context, str) else ""
    safe_history = recent_history_str.replace("{", "{{").replace("}", "}}") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("{", "{{").replace("}", "}}") if isinstance(query, str) else ""
    try:
        formatted_prompt = prompt_template.format(
            **{
               STATELESS_REFINER_CONTEXT_PLACEHOLDER.strip('{}'): safe_context,
               STATELESS_REFINER_HISTORY_PLACEHOLDER.strip('{}'): safe_history,
               STATELESS_REFINER_QUERY_PLACEHOLDER.strip('{}'): safe_query
            }
        )
        return formatted_prompt
    except KeyError as e:
        func_logger.error(f"Missing placeholder in stateless refiner prompt: {e}")
        return f"[Error: Missing placeholder '{e}']"
    except Exception as e:
        func_logger.error(f"Error formatting stateless refiner prompt: {e}", exc_info=True)
        return f"[Error formatting: {type(e).__name__}]"

# --- Function: Refine External Context (Stateless - Existing - Unchanged from latest) ---
async def refine_external_context(external_context: str, history_messages: List[Dict], latest_user_query: str, llm_call_func: Callable, refiner_llm_config: Dict, skip_threshold: int, history_count: int, dialogue_only_roles: List[str] = DIALOGUE_ROLES, caller_info: str = "StatelessRefiner") -> str:
    func_logger = logging.getLogger(__name__ + '.refine_external_context')
    func_logger.debug(f"[{caller_info}] Entered refine_external_context (stateless).")
    if not external_context or not external_context.strip():
        func_logger.debug(f"[{caller_info}] Skipping stateless refinement: External context is empty.")
        return external_context
    if not latest_user_query or not latest_user_query.strip():
        func_logger.debug(f"[{caller_info}] Skipping stateless refinement: Latest query is empty.")
        return external_context
    if not llm_call_func:
        func_logger.error(f"[{caller_info}] Skipping stateless refinement: llm_call_func not provided.")
        return external_context
    required_keys = ['url', 'key', 'temp']; template = refiner_llm_config.get('prompt_template')
    if not refiner_llm_config or not all(k in refiner_llm_config for k in required_keys):
        missing_keys = [k for k in required_keys if k not in (refiner_llm_config or {})]
        func_logger.error(f"[{caller_info}] Skipping stateless refinement: Config missing keys: {missing_keys}.")
        return external_context
    if not refiner_llm_config.get('url') or not refiner_llm_config.get('key'):
        func_logger.error(f"[{caller_info}] Skipping stateless refinement: Refiner URL/Key missing.")
        return external_context
    context_length = len(external_context)
    if skip_threshold > 0 and context_length < skip_threshold:
        func_logger.info(f"[{caller_info}] Skipping stateless refinement: Length ({context_length}) < Threshold ({skip_threshold}).")
        return external_context
    func_logger.info(f"[{caller_info}] Proceeding with stateless refinement (Length {context_length}).")
    recent_history_list = get_recent_turns(history_messages, history_count, dialogue_only_roles, True)
    recent_chat_history_str = format_history_for_llm(recent_history_list) if recent_history_list else "[No Recent History]"
    refiner_prompt_text = format_stateless_refiner_prompt(external_context=external_context, recent_history_str=recent_chat_history_str, query=latest_user_query, template=template)
    if not refiner_prompt_text or refiner_prompt_text.startswith("[Error:"):
        func_logger.error(f"[{caller_info}] Failed format stateless prompt: {refiner_prompt_text}. Aborting refinement.")
        return external_context
    refiner_payload = {"contents": [{"parts": [{"text": refiner_prompt_text}]}]}
    func_logger.info(f"[{caller_info}] Calling Stateless Refiner LLM...")
    try:
        success, response_or_error = await llm_call_func(
            api_url=refiner_llm_config['url'], api_key=refiner_llm_config['key'],
            payload=refiner_payload, temperature=refiner_llm_config['temp'],
            timeout=90, caller_info=caller_info
        )
    except Exception as e_call: func_logger.error(f"[{caller_info}] Exception during llm_call_func: {e_call}", exc_info=True); success = False; response_or_error = "LLM Call Exception"
    if success and isinstance(response_or_error, str) and response_or_error.strip():
        refined_context = response_or_error.strip()
        if refined_context.lower() == "no specific details relevant to the query were found in the provided context.":
            func_logger.info(f"[{caller_info}] Stateless refinement indicated no relevant details found. Returning original context.")
            return external_context
        else:
            func_logger.info(f"[{caller_info}] Stateless refinement successful (Length: {len(refined_context)}).")
            return refined_context
    else:
        error_details = str(response_or_error);
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.warning(f"[{caller_info}] Stateless refinement failed. Error: '{error_details}'. Returning original context.")
        return external_context

# --- Function: Format Cache Update Prompt (Existing - Unchanged from latest) ---
def format_cache_update_prompt(
    previous_cache: str,
    current_owi_rag: str,
    recent_history_str: str,
    query: str,
    template: str # Expecting the specific template for this step
) -> str:
    func_logger = logging.getLogger(__name__ + '.format_cache_update_prompt')
    if not template or not isinstance(template, str): return "[Error: Invalid Template for Cache Update]"
    safe_prev_cache = previous_cache.replace("{", "{{").replace("}", "}}") if isinstance(previous_cache, str) else ""
    safe_current_owi = current_owi_rag.replace("{", "{{").replace("}", "}}") if isinstance(current_owi_rag, str) else ""
    safe_history = recent_history_str.replace("{", "{{").replace("}", "}}") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("{", "{{").replace("}", "}}") if isinstance(query, str) else ""
    try:
        formatted_prompt = template.format(
             **{
                 CACHE_UPDATE_PREVIOUS_CACHE_PLACEHOLDER.strip('{}'): safe_prev_cache,
                 CACHE_UPDATE_CURRENT_OWI_PLACEHOLDER.strip('{}'): safe_current_owi,
                 CACHE_UPDATE_HISTORY_PLACEHOLDER.strip('{}'): safe_history,
                 CACHE_UPDATE_QUERY_PLACEHOLDER.strip('{}'): safe_query
             }
        )
        return formatted_prompt
    except KeyError as e: func_logger.error(f"Missing placeholder in cache update prompt: {e}"); return f"[Error: Missing placeholder '{e}']"
    except Exception as e: func_logger.error(f"Error formatting cache update prompt: {e}", exc_info=True); return f"[Error formatting: {type(e).__name__}]"

# --- Function: Format Final Context Selection Prompt (Existing - Unchanged from latest) ---
def format_final_context_selection_prompt(
    updated_cache: str,
    current_owi_rag: str, # Include current OWI for secondary check
    recent_history_str: str,
    query: str,
    template: str # Expecting the specific template for this step
) -> str:
    func_logger = logging.getLogger(__name__ + '.format_final_context_selection_prompt')
    if not template or not isinstance(template, str): return "[Error: Invalid Template for Context Selection]"
    safe_updated_cache = updated_cache.replace("{", "{{").replace("}", "}}") if isinstance(updated_cache, str) else ""
    safe_current_owi = current_owi_rag.replace("{", "{{").replace("}", "}}") if isinstance(current_owi_rag, str) else ""
    safe_history = recent_history_str.replace("{", "{{").replace("}", "}}") if isinstance(recent_history_str, str) else ""
    safe_query = query.replace("{", "{{").replace("}", "}}") if isinstance(query, str) else ""
    try:
        formatted_prompt = template.format(
             **{
                 FINAL_SELECT_UPDATED_CACHE_PLACEHOLDER.strip('{}'): safe_updated_cache,
                 FINAL_SELECT_CURRENT_OWI_PLACEHOLDER.strip('{}'): safe_current_owi,
                 FINAL_SELECT_HISTORY_PLACEHOLDER.strip('{}'): safe_history,
                 FINAL_SELECT_QUERY_PLACEHOLDER.strip('{}'): safe_query
             }
        )
        return formatted_prompt
    except KeyError as e: func_logger.error(f"Missing placeholder in final selection prompt: {e}"); return f"[Error: Missing placeholder '{e}']"
    except Exception as e: func_logger.error(f"Error formatting final selection prompt: {e}", exc_info=True); return f"[Error formatting: {type(e).__name__}]"

# === Function: Generate RAG Query (Existing - Unchanged structure from latest) ===
async def generate_rag_query(
    latest_message_str: str,
    dialogue_context_str: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    api_url: str,
    api_key: str,
    temperature: float,
    caller_info: str = "i4_llm_agent_RAGQueryGen",
) -> Optional[str]:
    logger.debug(f"[{caller_info}] Generating RAG query using library default prompt...")
    if not llm_call_func or not asyncio.iscoroutinefunction(llm_call_func):
        logger.error(f"[{caller_info}] Invalid llm_call_func.")
        return "[Error: Invalid LLM func]"
    if not api_url or not api_key:
        logger.error(f"[{caller_info}] Missing RAGQ URL/Key.")
        return "[Error: Missing RAGQ URL/Key]"
    if DEFAULT_RAGQ_LLM_PROMPT == "[Default RAGQ Prompt Load Failed]":
        logger.error(f"[{caller_info}] Library default RAGQ prompt constant failed to load.")
        return "[Error: Default RAGQ prompt missing]"
    ragq_prompt_text = None
    formatting_error = None
    safe_latest_message = latest_message_str.replace("{", "{{").replace("}", "}}") if isinstance(latest_message_str, str) else "[No message]"
    safe_dialogue_context = dialogue_context_str.replace("{", "{{").replace("}", "}}") if isinstance(dialogue_context_str, str) else "[No history]"
    try:
        ragq_prompt_text = DEFAULT_RAGQ_LLM_PROMPT.format( latest_message=safe_latest_message, dialogue_context=safe_dialogue_context )
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
        success, response_or_error = await llm_call_func( api_url=api_url, api_key=api_key, payload=ragq_payload, temperature=temperature, timeout=45, caller_info=caller_info, )
    except Exception as e_call:
        logger.error(f"[{caller_info}] Exception calling LLM wrapper for RAGQ: {e_call}", exc_info=True)
        success = False; response_or_error = f"[Error: LLM Call Exception {type(e_call).__name__}]"
    if success and isinstance(response_or_error, str):
        final_query = response_or_error.strip()
        if final_query: logger.info(f"[{caller_info}] Generated RAG query: '{final_query}'"); return final_query
        else: logger.warning(f"[{caller_info}] RAGQ LLM returned empty string."); return "[Error: RAGQ empty]"
    else:
        error_msg = str(response_or_error); logger.error(f"[{caller_info}] RAGQ failed: {error_msg}")
        if isinstance(response_or_error, dict): err_type = response_or_error.get('error_type', 'RAGQ Err'); err_msg_detail = response_or_error.get('message', 'Unknown'); return f"[Error: {err_type} - {err_msg_detail}]"
        else: return f"[Error: RAGQ Failed - {error_msg[:50]}]"

# --- Function: Construct Final LLM Payload (MODIFIED - New Structure & Moved Guidelines from latest) ---
def construct_final_llm_payload(
    system_prompt: str, # Base system prompt text (already cleaned)
    history: List[Dict], # Dialogue history turns (user/model)
    context: Optional[str], # Formatted background context string (XML-style, includes guidelines)
    query: str, # Final user query text for this turn
    long_term_goal: Optional[str] = None, # Dynamic goal text
    event_hint: Optional[str] = None, # Dynamic event hint text
    period_setting: Optional[str] = None, # Dynamic period setting text
    strategy: str = 'standard', # Strategy 'standard'/'advanced' (affects context placement relative to history - NOW context always comes before history)
    include_ack_turns: bool = True # Whether to include ACK turns
) -> Dict[str, Any]:
    """
    Constructs the final payload for the LLM in Google's 'contents' format,
    following the structure: System Instructions -> Background Context -> History -> Query.
    Injects dynamic guidelines (Goal, Event, Period) into System Instructions.
    Assumes Background Context string already contains static guidelines (Scene, Memory, Weather).
    """
    func_logger = logging.getLogger(__name__ + '.construct_final_llm_payload')
    func_logger.debug(
        f"Constructing final LLM payload (New Structure). ACKs: {include_ack_turns}, "
        f"Goal Provided: {bool(long_term_goal)}, Event Hint Provided: {bool(event_hint)}, "
        f"Period Setting Provided: '{period_setting or 'None'}'"
    )

    gemini_contents = []

    # 1. Prepare the System Instructions block (Base + Dynamic Guidelines)
    base_system_prompt_text = system_prompt.strip() if system_prompt else "You are a helpful assistant."
    # Start with the base prompt (already cleaned by process_system_prompt)
    final_system_instructions = base_system_prompt_text

    # --- Append Dynamic Guidelines ---
    # Append Long Term Goal (if provided)
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

    # Append Event Handling Guideline (if hint provided)
    # Note: EVENT_HANDLING_GUIDELINE_TEXT constant is imported from .event_hints
    if event_hint and isinstance(event_hint, str) and event_hint.strip():
        if EVENT_HANDLING_GUIDELINE_TEXT != "[EVENT GUIDELINE LOAD FAILED]":
            final_system_instructions += f"\n{EVENT_HANDLING_GUIDELINE_TEXT}" # Add newline separator
            func_logger.debug(f"Appended event handling guideline to system instructions text.")
        else:
            func_logger.warning("Event hint present but guideline text failed to load. Skipping append.")

    # Append Period Setting (if provided)
    safe_period_setting = period_setting.strip() if isinstance(period_setting, str) else None
    if safe_period_setting:
        period_block = f"""

--- [ Period Setting ] ---
[[Setting Instruction: Generate content appropriate for a '{safe_period_setting}' setting.]]
--- [ END Period Setting ] ---"""
        final_system_instructions += period_block
        func_logger.debug(f"Appended period setting instruction ('{safe_period_setting}') to system instructions text.")

    # NOTE: Scene, Memory, Weather guidelines are NOT appended here - they are expected inside the 'context' string.

    # 2. Add the System Instructions turn and optional ACK
    system_instructions_turn = None
    system_ack_turn = None
    if final_system_instructions:
        system_instructions_turn = {"role": "user", "parts": [{"text": f"System Instructions:\n{final_system_instructions}"}]}
        if include_ack_turns:
            ack_text = "Understood. I will follow these instructions."
            if safe_long_term_goal: ack_text += " I will also keep the long-term goal in mind."
            if safe_period_setting: ack_text += f" I will also maintain a '{safe_period_setting}' setting."
            system_ack_turn = {"role": "model", "parts": [{"text": ack_text}]}

    # 3. Prepare Background Context turn and optional ACK
    context_turn = None
    context_ack_turn = None
    has_real_context = bool(context and context.strip() and not context.strip().startswith("<Context type='Empty'>"))
    if has_real_context:
        # Context string now already contains XML tags and prepended guidelines
        context_injection_text = f"Background Information (Use this to inform your response):\n{context.strip()}"
        context_turn = {"role": "user", "parts": [{"text": context_injection_text}]}
        if include_ack_turns: context_ack_turn = {"role": "model", "parts": [{"text": "Understood. I have reviewed the background information."}]}

    # 4. Prepare History Turns (Filter for valid roles/content)
    history_turns = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        # Use standard Gemini roles user/model
        if role == "user" and content: history_turns.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant" and content: history_turns.append({"role": "model", "parts": [{"text": content}]})
        elif role == "model" and content: history_turns.append({"role": "model", "parts": [{"text": content}]})

    # 5. Prepare Final Query Turn (Inject event hint if provided)
    safe_query = query.strip() if query and query.strip() else "[User query not provided]"
    final_query_text = safe_query # Start with the base query

    # Inject Event Hint into Query Text if provided
    if event_hint and isinstance(event_hint, str) and event_hint.strip():
        formatted_hint = format_hint_for_query(event_hint) # e.g., "[[Event Suggestion: ...]]"
        if formatted_hint:
            final_query_text = f"{formatted_hint}\n\n{safe_query}" # Prepend the hint
            func_logger.debug(f"Prepended event hint to final query text.")

    final_query_turn = {"role": "user", "parts": [{"text": final_query_text}]} # Use the potentially modified text

    # 6. Assemble Payload in the New Order: [Sys] -> [Context] -> [History] -> [Query]
    # Strategy ('standard'/'advanced') is no longer relevant for context placement.
    if system_instructions_turn: gemini_contents.append(system_instructions_turn)
    if system_ack_turn: gemini_contents.append(system_ack_turn)
    if context_turn: gemini_contents.append(context_turn)
    if context_ack_turn: gemini_contents.append(context_ack_turn)
    gemini_contents.extend(history_turns)
    gemini_contents.append(final_query_turn)

    final_payload = {"contents": gemini_contents}
    func_logger.info(f"Final payload constructed with {len(gemini_contents)} turns using NEW structure.")
    return final_payload

# --- Function: Format Memory Aging Prompt (Existing - Unchanged from latest) ---
def format_memory_aging_prompt(t1_batch_text: str, template: Optional[str] = None) -> str:
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


# --- Function: Combine Background Context (MODIFIED - XML Tags & Prepend Guidelines from latest) ---
def combine_background_context(
    final_selected_context: Optional[str],
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
    labels: Optional[Dict[str, str]] = None # Labels no longer used for formatting
) -> str:
    """
    Combines various background context sources into a single XML-style formatted string.
    Prepends static guidelines (Scene Usage, Memory Structure, Weather Suggestion).
    Sorts Aged summaries Newest First, T1 summaries Oldest First.

    Args:
        final_selected_context: Context from OWI/Cache/Stateless refinement.
        t1_summaries: List of (text, metadata_dict) tuples for recent T1 summaries.
        aged_summaries: List of (text, metadata_dict) tuples for recent Aged summaries.
        t2_rag_results: List of retrieved Tier 2 RAG result strings.
        scene_description: The description text for the current scene.
        inventory_context: Formatted string of current character inventories.
        current_day: The current day number for the session.
        current_time_of_day: The current time of day string (e.g., "Morning").
        current_season: The current season string (e.g., "Summer").
        current_weather: The current weather string (e.g., "Clear").
        weather_proposal: Dict from Hint LLM, e.g., {"previous_weather": "X", "new_weather": "Y"}.
        labels: (Deprecated) No longer used for formatting.

    Returns:
        A single formatted string containing all valid context parts in XML structure,
        or an empty context placeholder tag if no context is available.
    """
    func_logger = logging.getLogger(__name__ + '.combine_background_context')
    context_parts = []

    # --- 0. Prepend Static Guidelines ---
    # These are now defined as constants in this file
    context_parts.append("<SystemContextGuidelines>")
    context_parts.append(SCENE_USAGE_GUIDELINE_TEXT)
    context_parts.append(MEMORY_SUMMARY_STRUCTURE_GUIDELINE_TEXT)
    context_parts.append(WEATHER_SUGGESTION_GUIDELINE_TEXT)
    context_parts.append("</SystemContextGuidelines>")
    func_logger.debug("Prepended static context guidelines.")

    # --- 1. World State ---
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

    # --- 2. Proposed Weather Change ---
    if isinstance(weather_proposal, dict):
        prev_w = weather_proposal.get("previous_weather"); new_w = weather_proposal.get("new_weather")
        if isinstance(prev_w, str) and isinstance(new_w, str):
            proposal_string = f"From '{prev_w}' to '{new_w}'"
            context_parts.append(f"<WeatherProposal>{proposal_string}</WeatherProposal>")
            func_logger.debug(f"Adding Weather Proposal section: {proposal_string}")

    # --- 3. Scene Description ---
    safe_scene_description = scene_description.strip() if isinstance(scene_description, str) else None
    if safe_scene_description:
        func_logger.debug(f"Adding scene description (len: {len(safe_scene_description)}).")
        # Basic XML escaping for content - adjust if more complex escaping needed
        escaped_scene = safe_scene_description.replace('<', '<').replace('>', '>').replace('&', '&')
        context_parts.append(f"<CurrentScene>{escaped_scene}</CurrentScene>")

    # --- 4. Inventory Context ---
    safe_inventory_context = inventory_context.strip() if isinstance(inventory_context, str) else None
    # Check content more robustly
    is_valid_inventory = safe_inventory_context and not re.search(r"\[(No Inventory|Error|Disabled)\]", safe_inventory_context, re.IGNORECASE)
    if is_valid_inventory:
        func_logger.debug(f"Adding inventory context (len: {len(safe_inventory_context)}).")
        escaped_inventory = safe_inventory_context.replace('<', '<').replace('>', '>').replace('&', '&')
        context_parts.append(f"<Inventories>{escaped_inventory}</Inventories>")

    # --- 5. Final Selected Context (Refined/OWI) ---
    safe_selected_context = final_selected_context.strip() if isinstance(final_selected_context, str) else None
    is_valid_selected = safe_selected_context and not re.search(r"\[(No relevant background context found|NO_CACHE_UPDATE)\]", safe_selected_context, re.IGNORECASE)
    if is_valid_selected:
        func_logger.debug(f"Adding selected context (len: {len(safe_selected_context)}).")
        escaped_selected = safe_selected_context.replace('<', '<').replace('>', '>').replace('&', '&')
        context_parts.append(f"<SelectedContext>{escaped_selected}</SelectedContext>")

    # --- 6. Aged Summaries ---
    valid_aged_summaries_data = []
    if aged_summaries and isinstance(aged_summaries, list):
        try:
            # Sort by creation timestamp DESC (most recent first)
            aged_summaries.sort(key=lambda item: item[1].get('creation_timestamp_utc', 0), reverse=True)
            valid_aged_summaries_data = [item for item in aged_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]
        except Exception as e_sort_aged:
            func_logger.error(f"Error sorting aged summaries: {e_sort_aged}. Using original order.")
            valid_aged_summaries_data = [item for item in aged_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]

    if valid_aged_summaries_data:
        context_parts.append('<AgedSummaries order="newest_recap_first">')
        for summary_text, metadata in valid_aged_summaries_data:
             escaped_summary = summary_text.replace('<', '<').replace('>', '>').replace('&', '&')
             # Add metadata attributes if needed, e.g., timestamp or span
             context_parts.append(f"<Summary>{escaped_summary}</Summary>")
        context_parts.append('</AgedSummaries>')
        func_logger.debug(f"Adding {len(valid_aged_summaries_data)} Aged summaries.")

    # --- 7. Recent T1 Summaries ---
    valid_t1_summaries_data = []
    if t1_summaries and isinstance(t1_summaries, list):
        try:
            # Sort by turn_end_index ASC (oldest first)
            t1_summaries.sort(key=lambda item: item[1].get('turn_end_index', -1), reverse=False)
            valid_t1_summaries_data = [item for item in t1_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]
        except Exception as e_sort_t1:
            func_logger.error(f"Error sorting T1 summaries: {e_sort_t1}. Using original order.")
            valid_t1_summaries_data = [item for item in t1_summaries if isinstance(item, tuple) and len(item) > 1 and isinstance(item[0], str) and item[0].strip() and isinstance(item[1], dict)]

    if valid_t1_summaries_data:
        context_parts.append('<RecentSummaries order="oldest_dialogue_first">')
        for summary_text, metadata in valid_t1_summaries_data:
             escaped_summary = summary_text.replace('<', '<').replace('>', '>').replace('&', '&')
             # Add metadata attributes if needed, e.g., turn index
             context_parts.append(f"<Summary>{escaped_summary}</Summary>")
        context_parts.append('</RecentSummaries>')
        func_logger.debug(f"Adding {len(valid_t1_summaries_data)} T1 summaries.")

    # --- 8. T2 RAG Results ---
    valid_t2_results = [s.strip() for s in t2_rag_results if isinstance(s, str) and s.strip()] if t2_rag_results else []
    if valid_t2_results:
        context_parts.append('<RelatedInformation source="T2_RAG">')
        for result_text in valid_t2_results:
             escaped_result = result_text.replace('<', '<').replace('>', '>').replace('&', '&')
             context_parts.append(f"<Info>{escaped_result}</Info>")
        context_parts.append('</RelatedInformation>')
        func_logger.debug(f"Adding {len(valid_t2_results)} T2 RAG results.")

    # --- Combine and Return ---
    # Check if we only have the guidelines part
    if len(context_parts) == 5 and context_parts[0] == "<SystemContextGuidelines>" and context_parts[-1] == "</SystemContextGuidelines>":
        func_logger.info("No actual background context available beyond guidelines.")
        return EMPTY_CONTEXT_PLACEHOLDER
    elif len(context_parts) > 0:
        full_context_string = "\n".join(context_parts) # Use newline as separator for readability in logs
        # Basic XML escaping for the final combined string might be needed if passing directly to XML parser elsewhere,
        # but for LLM injection, internal escaping of content is usually sufficient. Review if issues arise.
        func_logger.info(f"Combined context created (Total len: {len(full_context_string)}).")
        return full_context_string
    else:
        # This case should ideally not be reached if guidelines are always added
        func_logger.warning("Combine Background Context resulted in zero parts. Returning empty placeholder.")
        return EMPTY_CONTEXT_PLACEHOLDER


# --- Function: Format Inventory Update Prompt (Existing - Unchanged from latest) ---
def format_inventory_update_prompt(
    main_llm_response: str,
    user_query: str,
    recent_history_str: str,
    template: str # Expecting the specific template for this step
) -> str:
    func_logger = logging.getLogger(__name__ + '.format_inventory_update_prompt')
    if not template or not isinstance(template, str): return "[Error: Invalid Template for Inventory Update]"
    try:
        # Use replace for simpler substitution here
        formatted_prompt = template.replace(INVENTORY_UPDATE_RESPONSE_PLACEHOLDER, str(main_llm_response))
        formatted_prompt = formatted_prompt.replace(INVENTORY_UPDATE_QUERY_PLACEHOLDER, str(user_query))
        formatted_prompt = formatted_prompt.replace(INVENTORY_UPDATE_HISTORY_PLACEHOLDER, str(recent_history_str))
        return formatted_prompt
    except Exception as e:
        func_logger.error(f"Error formatting inventory update prompt: {e}", exc_info=True)
        return f"[Error formatting inventory update prompt: {type(e).__name__}]"

# --- Less Relevant Functions (Stubs - Unchanged from latest) ---
def assemble_tagged_context(base_prompt: str, contexts: Dict[str, Union[str, List[str]]]) -> str:
    logger.warning("assemble_tagged_context is a simplified stub and may not function as originally intended.")
    return base_prompt

def extract_tagged_context(system_content: str) -> Dict[str, str]:
    logger.warning("extract_tagged_context is a simplified stub and may not function as originally intended.")
    extracted = {}
    if not system_content or not isinstance(system_content, str): return extracted
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        match = re.search(pattern, system_content, re.DOTALL | re.IGNORECASE)
        if match: extracted[key] = match.group(1).strip()
    return extracted

# === END CORRECTED FILE: i4_llm_agent/prompting.py ===