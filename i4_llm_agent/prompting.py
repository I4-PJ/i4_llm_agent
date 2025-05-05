# === START OF COMPLETE FILE: i4_llm_agent/prompting.py (Revised Inventory Prompt) ===
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

# <<< Event Hint Guideline is now defined HERE directly >>>
# Import format_hint_for_query only, the guideline text is defined below
try:
    from .event_hints import format_hint_for_query
except ImportError:
    def format_hint_for_query(hint): return f"[[Hint Load Failed: {hint}]]"
    logging.getLogger(__name__).error("Failed to import format_hint_for_query from event_hints in prompting.py")

logger = logging.getLogger(__name__) # 'i4_llm_agent.prompting'

# --- Constants for Context Tags (XML-STYLE focused) ---
# Kept for cleaning old formats if necessary and for combine_background_context
KNOWN_CONTEXT_TAGS = {
    "owi": ("<context>", "</context>"),
    "t1": ("<mempipe_recent_summary>", "</mempipe_recent_summary>"),
    "t2_rag": ("<mempipe_rag_result>", "</mempipe_rag_result>"),
}
EMPTY_CONTEXT_PLACEHOLDER = "<Context type='Empty'>[No Background Information Available]</Context>"


# === Summarizer Prompt Constants ===
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

# === Memory Aging Prompt Constants ===
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

# === RAG Query Prompt Constant ===
DEFAULT_RAGQ_LLM_PROMPT = """Based on the latest user message and recent dialogue context, generate a concise search query focusing on the key entities, topics, or questions raised.

Latest Message: {latest_message}

Dialogue Context:
{dialogue_context}

Search Query:"""

# === Inventory Update Prompt Constants ===
INVENTORY_UPDATE_RESPONSE_PLACEHOLDER = "{main_llm_response}"
INVENTORY_UPDATE_QUERY_PLACEHOLDER = "{user_query}"
INVENTORY_UPDATE_HISTORY_PLACEHOLDER = "{recent_history_str}"

# <<< START REVISED INVENTORY PROMPT (v3 - Stricter Reuse) >>>
DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT = f"""
[[SYSTEM DIRECTIVE]]
Role: Narrative Inventory Analyst & Maintainer
Task: Analyze the latest interaction (User Query, Assistant Response, Recent History) to identify changes to character inventories. This includes explicit commands, narrative descriptions of acquiring, losing, using, breaking, replacing, or storing items. Maintain item relevance by updating status based on narrative context. **Accurate and consistent canonical naming is CRITICAL for tracking.**
Objective: Output a structured JSON object detailing ONLY the inventory items that were added or modified during this turn, reflecting their intended final state according to the new schema. If no changes are detected, output {{"character_updates": []}}.

**Item Schema (Required Structure for items in output):**
{{
  "quantity": <integer>, // Current count of the item
  "category": "Clothing" | "Consumable" | "Tool" | "Material" | "Quest Item" | "Currency" | "Other", // Best fit category
  "status": "active" | "equipped" | "stored" | "broken" | "consumed" | "obsolete", // Reflects current relevance/state
  "properties": {{ // Object for descriptive details
    "material": "<string>", // e.g., "wool", "iron", "leather"
    "color": "<string>", // e.g., "blue", "dark green"
    "quality": "<string>", // e.g., "sturdy", "worn", "fine", "new", "mended"
    // Add other observed properties as key-value pairs
    "special_tags": ["<string>", ...] // List for flags like "waterproofed", "lined", "magical"
  }},
  "origin_notes": "<string>" // BRIEF note on acquisition/significance if relevant, otherwise empty string ""
}}

**Output JSON Format (Required Structure for overall output):**
{{
  "character_updates": [
    {{
      "character_name": "Name or __USER__", // Character whose inventory changed
      "updated_items": {{
        // Key: Canonical Name of the item. **THIS IS THE CRITICAL UNIQUE IDENTIFIER.**
        // Value: The *complete item schema object* reflecting the item's
        //        intended final state after this turn's events.
        //        Include items here if they were added OR if any field
        //        (quantity, status, properties) was modified.
        "Canonical Name 1": {{ ...item schema... }}, // Use TitleCase, prefer singular
        "Canonical Name 2": {{ ...item schema... }}
        // ... only include items ADDED or MODIFIED this turn.
      }}
    }}
    // ... include another object in the list if a second character's inventory was also affected.
  ]
}}

**Instructions (Follow in Order):**

1.  **Identify Characters:** Determine which character(s) inventories were affected (use `__USER__` for the player if name unclear).
2.  **Check for Strict Commands:** Examine the **USER QUERY**. Does it start with `INVENTORY: ADD/REMOVE/SET/CLEAR`?
    *   If YES: Parse the command **strictly**. Apply the command to items identified by their **Canonical Name**. Generate the `updated_items` reflecting the command's direct effect (e.g., `ADD` creates an item with `status: "active"`, `REMOVE` might imply setting quantity to 0 or status to `"obsolete"` if quantity drops to zero, `SET` updates quantity, `CLEAR` implies setting all items for that character to `status: "obsolete"`). **Stop processing other inputs and output the JSON.**
    *   If NO: Proceed to Instruction 3.

3.  **Analyze Dialogue & Narrative:** Analyze the **ASSISTANT RESPONSE** and **USER QUERY** (using History for context) for narrative events affecting inventory:
    *   **Acquisition:** Picking up, receiving, buying, finding, crafting items. -> Identify/establish the item's **Canonical Name**. Create/update the item entry. Set `status: "active"` (or `"equipped"` if worn/wielded immediately). Extract `properties`, `category`.
    *   **Loss/Removal:** Dropping, giving away, selling items. -> Identify the item by its **Canonical Name**. Update its `status` to `"obsolete"` or `"stored"`. If quantity becomes 0, reflect that.
    *   **Consumption/Usage:** Using potions, food rations, arrows, materials for crafting. -> Identify the item by its **Canonical Name**. Update `status` to `"consumed"` or decrease `quantity`. If quantity becomes 0, status can also be `"consumed"`.
    *   **Breakage:** Item explicitly described as breaking. -> Identify the item by its **Canonical Name**. Update its `status` to `"broken"`.
    *   **Replacement:** Receiving a new item that clearly replaces an older one (e.g., "new sturdy boots" replace "old worn boots"). -> Identify the *old* item by its **Canonical Name** and update its `status` to `"stored"` or `"obsolete"`. Identify/establish the **Canonical Name** for the *new* item (it might be the *same* name if only properties changed, or a *new distinct* name if significantly different) and add/update it with `status: "active"` or `"equipped"`.
    *   **Status Change:** Item being equipped, unequipped, stored. -> Identify the item by its **Canonical Name**. Update its `status` accordingly. Properties might change (e.g., `quality: "worn"` after long use).
    *   **Quest Item Obsolescence:** Item tied to a completed task. -> Identify the item by its **Canonical Name**. Update its `status` to `"obsolete"`.

4.  **Determine Canonical Names (CRITICAL STEP):**
    *   The `canonical_name` (used as the JSON key in `updated_items`) **MUST** be the unique identifier for a specific *type* of item.
    *   **Consistency is MANDATORY.** You **MUST** reuse the *exact* same canonical name string (including case) for all instances of the same item type.
    *   **Naming Convention:**
        *   Use **Title Case** (e.g., "Iron Sword", "Leather Boots", "Health Potion").
        *   **Prefer Singular** names (e.g., "Leather Boot", "Wool Stocking", "Coin") even if multiple exist (use the `quantity` field for count). Exceptions: intrinsically plural items like "Rations", "Arrows".
        *   Be descriptive enough to differentiate (e.g., "Steel Sword" vs. "Iron Dagger").
    *   **Mapping Variations:** Map variations in dialogue ("the blue cloak", "her wool wrap", "a sword") to the **established Canonical Name**. If dialogue mentions "a wool stocking" and the canonical name "Wool Stocking" already exists, **update the existing entry**.
    *   **Resolving Ambiguity:** If unsure whether a mentioned item (e.g., "a cloak") refers to an existing canonical item (e.g., "Wool Cloak"), **assume it refers to the existing item** unless properties *clearly* differ (e.g., dialogue mentions a "silk cloak" - this would likely need a new canonical name like "Silk Cloak").
    *   **CRITICAL - Reuse Existing Names:** Before creating a NEW canonical name, you MUST check if any existing item for that character is a close match (e.g., differing only by adjectives like color or quality, or using simpler phrasing like 'the dress' instead of 'Velvet Dress'). If a close match exists, ALWAYS update the existing item using its established canonical name. DO NOT create slightly different names for the same core item (e.g., do not create 'Velvet Dress' if 'Sapphire Velvet Dress' exists and refers to the same object). Prioritize updating over creating duplicates.
    *   **Creating New Names:** Only create a *new* canonical name if the item is genuinely distinct from all existing items for that character (e.g., finding a "Ruby Ring" when no rings existed before).

5.  **Apply Schema & Status Logic:** For every item identified as added or modified:
    *   Use the **final, chosen Canonical Name** (from Step 4) as the key in the `updated_items` dictionary.
    *   Construct the full item schema object (value) reflecting its **final state** after the turn's events.
    *   When updating an existing item identified by its Canonical Name, **preserve its existing properties** unless the narrative explicitly states a change (e.g., item becomes 'worn', 'mended', 'damaged'). Only add *new* properties mentioned in the current turn.
    *   Pay close attention to updating the `status` field based on the narrative analysis in Step 3.
    *   Extract `properties` (color, material, quality, tags) mentioned in the text. Assign a `category`. Add brief `origin_notes` if significant context exists.
    *   Ensure `quantity` is correct. If status becomes `"consumed"` or `"broken"`, quantity often becomes 0, but include the item entry in the output with the final status.

6.  **Handle Ambiguity:** If it's unclear whether an old item was discarded or just stored after being replaced, default to setting its `status: "stored"`. If narrative details are insufficient to populate properties, use reasonable defaults or omit them.

7.  **Ignore Transient Items:** Do **not** create inventory entries for items clearly not intended for possession (e.g., food eaten immediately from a plate, scenery objects not taken, temporary tools used and left behind like a borrowed hammer).

8.  **Construct Final JSON:** Assemble the final JSON according to the specified **Output JSON Format** (Section 3). Only include characters who had updates, and only include items whose state was added or modified in the `updated_items` dictionary for that character.

9.  **No Change:** If no commands were found and no narrative changes to any character's inventory were detected, output `{{"character_updates": []}}`.

**INPUTS:**

**USER QUERY (Check for commands first):**
{INVENTORY_UPDATE_QUERY_PLACEHOLDER}

**ASSISTANT RESPONSE (Analyze for narrative changes if no command):**
---
{INVENTORY_UPDATE_RESPONSE_PLACEHOLDER}
---

**RECENT CHAT HISTORY (For context, especially pronoun/name resolution and narrative flow):**
---
{INVENTORY_UPDATE_HISTORY_PLACEHOLDER}
---

**OUTPUT (JSON object with detected inventory additions/modifications):**
"""
# <<< END REVISED INVENTORY PROMPT (v3 - Stricter Reuse) >>>


# === Guideline Constants ===
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

# <<< NEW EVENT HINT GUIDELINE TEXT >>>
EVENT_HANDLING_GUIDELINE_TEXT = """--- [ EVENT HANDLING GUIDELINE ] ---
**Optional Event Hints:** You may occasionally find an `<EventHint>...</EventHint>` tag within the Background Information block. This contains a **strictly optional** suggestion for a minor environmental detail or sensory input.

**Handling Rules (Low Priority):**
1.  **Optionality:** You are **NOT required** to use the hint. Prioritize responding to the user's query, maintaining character voice/focus, and describing the core scene context **above** incorporating the hint.
2.  **Subtlety:** If you *choose* to use the hint, weave it in subtly and naturally. It should feel like an organic observation, not a forced event.
3.  **Ignore If:** You **SHOULD ignore** the hint if:
    *   It feels forced, unnatural, or contradicts the established mood/scene.
    *   It would disrupt the flow of dialogue or the character's current focus/actions.
    *   The scene is already detailed enough, and the hint adds little value.
    *   It seems irrelevant to the immediate situation.
4.  **Do Not:** Do not treat the hint as a command. Do not announce that you are using the hint. Do not let it derail the conversation.

**In summary: Treat the `<EventHint>` as a low-priority, optional atmospheric suggestion. Use it sparingly and only when it genuinely enhances the scene without disrupting the core narrative flow.**
--- [ END EVENT HANDLING GUIDELINE ] ---"""
# <<< END NEW EVENT HINT GUIDELINE TEXT >>>


# === Cache Maintainer Prompt Constants ===
CACHE_MAINTAINER_QUERY_PLACEHOLDER = "{query}"
CACHE_MAINTAINER_HISTORY_PLACEHOLDER = "{recent_history_str}"
CACHE_MAINTAINER_PREVIOUS_CACHE_PLACEHOLDER = "{previous_cache_text}"
CACHE_MAINTAINER_CURRENT_OWI_PLACEHOLDER = "{current_owi_context}"
NO_CACHE_UPDATE_FLAG = "[NO_CACHE_UPDATE]"
# <<< REFINED TEMPLATE V2 (Less Aggressive Pruning) >>>
DEFAULT_CACHE_MAINTAINER_TEMPLATE_TEXT = f"""
[[SYSTEM DIRECTIVE]]
**Role:** Session Background Cache Maintainer & Synthesizer

**Task:** Critically evaluate the PREVIOUSLY REFINED CACHE against **both** the CURRENT OWI RETRIEVAL **and** the RECENT CHAT HISTORY, considering the LATEST USER QUERY for immediate relevance. Decide if a cache update is warranted. If yes, perform an intelligent merge/update, prioritizing **conciseness and relevance** to the ongoing narrative and immediate plans.

**Objective:** Maintain an accurate, concise, and consistently structured SESSION CACHE containing **essential and currently relevant** background information (active character states/goals, key plot points, immediately relevant lore/setting). The cache should provide stable context but **evolve** with the narrative. **Prune or summarize** information that is less central to the current plot arc or character interactions to maintain an efficient size target (aim for roughly 25,000 characters).

**Inputs:**
- **LATEST USER QUERY:** Provides immediate context for relevance checking.
- **RECENT CHAT HISTORY:** **Primary source for recent plot developments, character decisions, and relationship shifts.**
- **PREVIOUSLY REFINED CACHE:** The baseline state of the background context.
- **CURRENT OWI RETRIEVAL:** Secondary source for *genuinely new*, significant factual details, clarifications, or corrections *not* already reflected adequately in the cache or history.

**Core Logic & Instructions:**

1.  **Analyze Recent Dialogue:**
    *   Identify key decisions, plot advancements, emotional shifts, or relationship changes revealed in the RECENT CHAT HISTORY.
    *   Determine if these developments necessitate updates to the 'Key Plot/Goal', character profiles (current state/motivations), or relationship summaries in the cache.

2.  **Analyze OWI vs. Cache (Conditional):**
    *   Compare CURRENT OWI RETRIEVAL against the PREVIOUSLY REFINED CACHE.
    *   **Only consider OWI if it provides genuinely NEW, significant, and non-redundant factual information** (e.g., a new character reveal, a major world event correction, significant lore elaboration *directly relevant* to the current query/plot).
    *   **IGNORE** OWI if it's merely redundant, rephrased, or contains details irrelevant to the current focus.

3.  **Decision Point:**
    *   Update is needed if:
        *   Recent Dialogue revealed significant plot/character state changes needing integration.
        *   *OR* OWI provided genuinely new, significant, relevant factual updates.
    *   **If NEITHER condition is met** -> **Output ONLY the exact text:** `{NO_CACHE_UPDATE_FLAG}` and stop processing.
    *   **If YES, proceed to Step 4 (Update/Merge/Refine).**

4.  **Update/Merge/Refine Process (If Step 3 determined an update is needed):**
    *   Start with the full text of the PREVIOUSLY REFINED CACHE as the base.
    *   **Integrate Dialogue Updates:** Update the 'Key Plot/Goal' section, character states (e.g., Caldric's confirmed reluctance, agreed departure timing), or relationship notes based *directly* on the analysis from Step 1. Be concise.
    *   **Integrate OWI Updates (If Applicable):** If Step 2 identified valuable NEW info in OWI, merge it concisely:
        *   Character Profiles: Add significant new *facts* or *major* status changes. Avoid minor elaborations unless crucial. Prefer established cache profiles unless OWI offers clear correction. Add summaries for genuinely new, relevant characters.
        *   Factual Lore/Background: Add NEW relevant facts or MAJOR clarifications/corrections identified in OWI.
    *   **Pruning & Summarization (For Efficiency):**
        *   Review **all sections** of the cache (including existing parts).
        *   **Prune:** Remove details or sections that are clearly no longer relevant to the *current* plot arc or active character goals (e.g., fully resolved minor quests, details of inactive characters).
        *   **Summarize:** Condense lengthy descriptions or established background facts that are stable but not immediately critical. Retain the core information but reduce word count where appropriate. (e.g., summarize detailed servant descriptions unless one is actively involved; consider summarizing detailed world history if not currently relevant).
        *   **Prioritize:** Focus retention on: Current Plot/Goals/Plans, Active Character States & Relationships (including recent emotional or planning nuances), Immediately Relevant Setting/Lore.
    *   **Maintain Structure & Target Length:** Preserve heading structure. **Prune and summarize less immediately relevant information to keep the total output concise and focused, aiming for roughly 25,000 characters.** Ensure the most important info (Plot, Active Characters) is prominent.

5.  **Final Output Construction:**
    *   If Step 3 resulted in `{NO_CACHE_UPDATE_FLAG}`, that is the entire output.
    *   If Step 4 was performed, the output is the **complete, updated, and REFINED SESSION CACHE text**.
    *   **Empty/Irrelevant Case:** If the PREVIOUS CACHE was empty AND Recent Dialogue/OWI contain no significant relevant info -> Output: `[No relevant background context found]`

**INPUTS:**

**LATEST USER QUERY:**
{CACHE_MAINTAINER_QUERY_PLACEHOLDER}

**RECENT CHAT HISTORY:**
---
{CACHE_MAINTAINER_HISTORY_PLACEHOLDER}
---

**PREVIOUSLY REFINED CACHE (Baseline Context):**
---
{CACHE_MAINTAINER_PREVIOUS_CACHE_PLACEHOLDER}
---

**CURRENT OWI RETRIEVAL (Check for New/Updated Info):**
---
{CACHE_MAINTAINER_CURRENT_OWI_PLACEHOLDER}
---

**OUTPUT (Either '{NO_CACHE_UPDATE_FLAG}', the refined cache text, or '[No relevant background context found]'):**
"""
# <<< END REFINED TEMPLATE V2 >>>

# === Function Implementations ===

# --- Function: Clean Context Tags ---
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

# --- Function: Process System Prompt ---
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


# --- Function: Generate RAG Query ---
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


# --- Function: Combine Background Context (MODIFIED) ---
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
    event_hint_text: Optional[str] = None, # <<< NEW PARAMETER
    labels: Optional[Dict[str, str]] = None # Labels no longer used
) -> str:
    """
    Combines various background context sources into a single XML-style formatted string.
    Prepends static guidelines (Scene Usage, Memory Structure, Weather Suggestion).
    Includes event hint within an <EventHint> tag if provided. <<< MODIFIED
    Sorts Aged summaries Newest First, T1 summaries Oldest First.
    """
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

    # <<< START NEW HINT LOGIC >>>
    safe_event_hint = event_hint_text.strip() if isinstance(event_hint_text, str) else None
    if safe_event_hint:
        # Basic XML escaping for the hint text itself
        escaped_hint = safe_event_hint.replace('&', '&').replace('<', '<').replace('>', '>')
        # Wrap the raw (but escaped) hint text in the new tag
        context_parts.append(f"<EventHint>{escaped_hint}</EventHint>")
        func_logger.debug(f"Adding Event Hint section: {escaped_hint[:80]}...")
    # <<< END NEW HINT LOGIC >>>

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
        func_logger.debug(f"Adding raw OWI/Cache context (len: {len(safe_selected_context)}).")
        escaped_selected = safe_selected_context.replace('&', '&').replace('<', '<').replace('>', '>')
        context_parts.append(f"<SelectedContext source='OWI_Direct'>{escaped_selected}</SelectedContext>") # Assuming OWI/Cache for label

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

    if len(context_parts) <= 5: # Adjusted count for only guidelines
        func_logger.info("No actual background context available beyond guidelines.")
        return EMPTY_CONTEXT_PLACEHOLDER
    else:
        full_context_string = "\n".join(context_parts)
        func_logger.info(f"Combined context created (Total len: {len(full_context_string)}).")
        return full_context_string


# --- Function: Construct Final LLM Payload (MODIFIED) ---
def construct_final_llm_payload(
    system_prompt: str, # Base system prompt text (already cleaned)
    history: List[Dict], # Dialogue history turns (user/model)
    context: Optional[str], # Formatted background context string (XML-style, includes guidelines, MAY include hint)
    query: str, # Final user query text for this turn
    long_term_goal: Optional[str] = None, # Dynamic goal text
    event_hint: Optional[str] = None, # Parameter still received but no longer used *here*
    period_setting: Optional[str] = None, # Dynamic period setting text
    include_ack_turns: bool = True # Whether to include ACK turns
) -> Dict[str, Any]:
    """
    Constructs the final payload for the LLM in Google's 'contents' format.
    Event Hint is now expected *within* the 'context' string if present. <<< MODIFIED
    Assumes Background Context string already contains static guidelines.
    """
    func_logger = logging.getLogger(__name__ + '.construct_final_llm_payload')
    func_logger.debug(
        f"Constructing final LLM payload (Event Hint in Context). ACKs: {include_ack_turns}, "
        f"Goal Provided: {bool(long_term_goal)}, "
        # Event hint existence no longer logged here as it's inside context
        f"Period Setting Provided: '{period_setting or 'None'}'"
    )
    gemini_contents = []

    # --- System Instructions Block (Includes Goal, Period Setting, Event Handling Guideline) ---
    base_system_prompt_text = system_prompt.strip() if system_prompt else "You are a helpful assistant."
    final_system_instructions = base_system_prompt_text

    # Append Goal if present
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

    # Append Event Handling Guideline (uses the *updated* constant)
    if EVENT_HANDLING_GUIDELINE_TEXT != "[EVENT GUIDELINE LOAD FAILED]": # Check constant load status
        final_system_instructions += f"\n{EVENT_HANDLING_GUIDELINE_TEXT}" # Constant is updated
        func_logger.debug(f"Appended event handling guideline (Hint in Context version) to system instructions text.")
    else:
        func_logger.warning("Event handling guideline text failed to load. Skipping append.")

    # Append Period Setting if present
    safe_period_setting = period_setting.strip() if isinstance(period_setting, str) else None
    if safe_period_setting:
        period_block = f"""

--- [ Period Setting ] ---
[[Setting Instruction: Generate content appropriate for a '{safe_period_setting}' setting.]]
--- [ END Period Setting ] ---"""
        final_system_instructions += period_block
        func_logger.debug(f"Appended period setting instruction ('{safe_period_setting}') to system instructions text.")

    # --- System Instructions Turn(s) ---
    system_instructions_turn = None; system_ack_turn = None
    if final_system_instructions:
        system_instructions_turn = {"role": "user", "parts": [{"text": f"System Instructions:\n{final_system_instructions}"}]}
        if include_ack_turns:
            # Construct ACK text based on included elements
            ack_text_parts = ["Understood. I will follow these instructions."]
            if safe_long_term_goal: ack_text_parts.append("I will also keep the long-term goal in mind.")
            if safe_period_setting: ack_text_parts.append(f"I will also maintain a '{safe_period_setting}' setting.")
            ack_text = " ".join(ack_text_parts)
            system_ack_turn = {"role": "model", "parts": [{"text": ack_text}]}

    # --- Context Block Turn(s) ---
    context_turn = None; context_ack_turn = None
    has_real_context = bool(context and context.strip() and not context.strip().startswith("<Context type='Empty'>"))
    if has_real_context:
        context_injection_text = f"Background Information (Use this to inform your response):\n{context.strip()}" # Context now includes hint tag if present
        context_turn = {"role": "user", "parts": [{"text": context_injection_text}]}
        if include_ack_turns:
             context_ack_turn = {"role": "model", "parts": [{"text": "Understood. I have reviewed the background information."}]}

    # --- History Turns ---
    history_turns = []
    for msg in history:
        role = msg.get("role"); content = msg.get("content", "").strip()
        if role == "user" and content: history_turns.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant" and content: history_turns.append({"role": "model", "parts": [{"text": content}]})
        elif role == "model" and content: history_turns.append({"role": "model", "parts": [{"text": content}]}) # Include model role

    # --- Final Query Turn ---
    safe_query = query.strip() if query and query.strip() else "[User query not provided]"
    final_query_text = safe_query # <<< HINT PREPENDING IS REMOVED HERE >>>

    final_query_turn = {"role": "user", "parts": [{"text": final_query_text}]}

    # --- Assemble Payload ---
    if system_instructions_turn: gemini_contents.append(system_instructions_turn)
    if system_ack_turn: gemini_contents.append(system_ack_turn)
    if context_turn: gemini_contents.append(context_turn)
    if context_ack_turn: gemini_contents.append(context_ack_turn)
    gemini_contents.extend(history_turns)
    gemini_contents.append(final_query_turn)

    final_payload = {"contents": gemini_contents}
    func_logger.info(f"Final payload constructed with {len(gemini_contents)} turns (Hint moved to Context).")
    return final_payload


# --- Function: Format Memory Aging Prompt ---
def format_memory_aging_prompt(t1_batch_text: str, template: Optional[str] = None) -> str:
    """Formats the prompt for the Memory Aging LLM."""
    func_logger = logging.getLogger(__name__ + '.format_memory_aging_prompt')
    prompt_template = template if template is not None else DEFAULT_MEMORY_AGING_PROMPT_TEMPLATE
    if not prompt_template or prompt_template == "[Default Memory Aging Prompt Load Failed]":
        return "[Error: Invalid or Missing Template for Memory Aging]"
    safe_batch_text = t1_batch_text.replace("{", "{{").replace("}", "}}") if isinstance(t1_batch_text, str) else ""
    try:
        # Use .format() with placeholder name stripped of braces
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


# --- Function: Format Inventory Update Prompt ---
def format_inventory_update_prompt(
    main_llm_response: str,
    user_query: str,
    recent_history_str: str,
    template: str # Expecting the specific template for this step
) -> str:
    """Formats the prompt for the Inventory Update LLM."""
    func_logger = logging.getLogger(__name__ + '.format_inventory_update_prompt')
    if not template or not isinstance(template, str) or template == "[Default Inventory Prompt Load Failed]": # Check against the constant name
        return "[Error: Invalid or Missing Template for Inventory Update]"
    try:
        # Use str.replace for simple placeholders if format causes issues
        formatted_prompt = template.replace(INVENTORY_UPDATE_RESPONSE_PLACEHOLDER, str(main_llm_response))
        formatted_prompt = formatted_prompt.replace(INVENTORY_UPDATE_QUERY_PLACEHOLDER, str(user_query))
        formatted_prompt = formatted_prompt.replace(INVENTORY_UPDATE_HISTORY_PLACEHOLDER, str(recent_history_str))
        return formatted_prompt
    except Exception as e:
        func_logger.error(f"Error formatting inventory update prompt: {e}", exc_info=True)
        return f"[Error formatting inventory update prompt: {type(e).__name__}]"


# --- Function: Format Cache Maintainer Prompt ---
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
    if not prompt_template or prompt_template == "[Default Cache Maintainer Prompt Load Failed]": # Check if default failed
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
               # NO_CACHE_UPDATE_FLAG is handled by literal replacement below
            }
        )
        # Replace the flag placeholder literally within the formatted string
        # This ensures the exact flag string is present for comparison later
        # Ensure the placeholder `{NO_CACHE_UPDATE_FLAG}` is exactly as written here in the template text
        formatted_prompt = formatted_prompt.replace("{NO_CACHE_UPDATE_FLAG}", NO_CACHE_UPDATE_FLAG)

        return formatted_prompt
    except KeyError as e:
        func_logger.error(f"Missing placeholder in cache maintainer prompt: {e}")
        return f"[Error: Missing placeholder '{e}']"
    except Exception as e:
        func_logger.error(f"Error formatting cache maintainer prompt: {e}", exc_info=True)
        return f"[Error formatting cache maintainer prompt: {type(e).__name__}]"


# --- Legacy/Stub Functions ---
def assemble_tagged_context(base_prompt: str, contexts: Dict[str, Union[str, List[str]]]) -> str:
    """Simplified stub for assembling context with old-style tags. May not function correctly."""
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
    logger.warning("extract_tagged_context is a simplified stub and may not function as originally intended.")
    extracted = {}
    if not system_content or not isinstance(system_content, str): return extracted
    for key, (start_tag, end_tag) in KNOWN_CONTEXT_TAGS.items():
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        match = re.search(pattern, system_content, re.DOTALL | re.IGNORECASE)
        if match: extracted[key] = match.group(1).strip()
    return extracted

# === END OF COMPLETE FILE: i4_llm_agent/prompting.py (Revised Inventory Prompt) ===