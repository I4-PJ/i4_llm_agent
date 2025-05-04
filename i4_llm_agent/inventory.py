# [[START FINAL COMPLETE inventory.py]]
# i4_llm_agent/inventory.py

import logging
import json
import asyncio
import re
import sqlite3
from typing import Dict, Optional, Callable, Any, Tuple, Union, Coroutine, List

# Assuming prompting constants are available via import or defined elsewhere
# from .prompting import DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT # Needed by caller
# from .prompting import format_inventory_update_prompt # Imported dynamically below

logger = logging.getLogger(__name__) # i4_llm_agent.inventory

# ==============================================================================
# === 1. Inventory Formatting (Refactored for New Schema)                   ===
# ==============================================================================

def format_inventory_for_prompt(
    session_inventories: Dict[str, str], # char_name -> inventory_json_string
    include_statuses: List[str] = ["active", "equipped"], # Default statuses to show
    exclude_categories: List[str] = ["Currency"], # Default categories to hide
    show_stored: bool = False # Optionally include 'stored' items
) -> str:
    """
    Formats inventory data (new structured schema) retrieved from the database
    into a single, human-readable string suitable for context injection.

    Filters items based on status and category. Groups by category.
    """
    if not session_inventories:
        return "[No Inventory Data Available]"

    # Add 'stored' to include_statuses if show_stored is True
    statuses_to_show = include_statuses[:] # Create a copy
    if show_stored and "stored" not in statuses_to_show:
        statuses_to_show.append("stored")

    formatted_parts = []
    sorted_char_names = sorted(session_inventories.keys())

    for char_name in sorted_char_names:
        inventory_json = session_inventories[char_name]
        char_header = f"{char_name}'s Inventory:"
        items_by_category: Dict[str, List[str]] = {}

        if not inventory_json or inventory_json.strip() == '{}':
            formatted_parts.append(f"{char_header}\n[Empty]")
            continue

        try:
            # Expecting inventory_data to be: {"canonical_name": {item_schema}, ...}
            inventory_data: Dict[str, Dict[str, Any]] = json.loads(inventory_json)
            if not isinstance(inventory_data, dict) or not inventory_data:
                formatted_parts.append(f"{char_header}\n[Empty]")
                continue

            # Sort items by canonical name for consistent order within categories
            sorted_item_names = sorted(inventory_data.keys())

            for item_name in sorted_item_names:
                item_details = inventory_data[item_name]
                if not isinstance(item_details, dict):
                    logger.warning(f"Skipping invalid item data format for '{item_name}' in {char_name}'s inventory.")
                    continue

                item_status = item_details.get("status", "active") # Default to active if missing
                item_category = item_details.get("category", "Other") # Default category
                item_quantity = item_details.get("quantity", 0)

                # --- Filtering Logic ---
                if item_status not in statuses_to_show:
                    continue # Skip items with non-display statuses
                if item_category in exclude_categories:
                    continue # Skip excluded categories

                # Skip items with zero quantity unless specifically equipped (edge case)
                if item_quantity <= 0 and item_status != "equipped":
                     continue

                # --- Formatting Logic ---
                item_line_parts = [f"- {item_name}"] # Start with canonical name
                if item_quantity > 1:
                    item_line_parts.append(f"({item_quantity})")

                # Indicate status clearly if relevant
                if item_status == "equipped":
                    item_line_parts.append("[Equipped]")
                elif item_status == "stored":
                     item_line_parts.append("[Stored]") # Only shown if show_stored=True allows it

                # Add properties concisely
                properties = item_details.get("properties")
                prop_strings = []
                if isinstance(properties, dict):
                    prop_order = ['quality', 'color', 'material'] # Define preferred order
                    for prop_key in prop_order:
                        if properties.get(prop_key):
                            prop_strings.append(f"{properties[prop_key]}") # Just value for conciseness

                    # Add special tags if they exist
                    if isinstance(properties.get("special_tags"), list) and properties["special_tags"]:
                        prop_strings.append(f"({', '.join(properties['special_tags'])})")

                if prop_strings:
                    item_line_parts.append(f"{{{', '.join(prop_strings)}}}") # Join props with comma

                # Add the formatted line to the correct category
                if item_category not in items_by_category:
                    items_by_category[item_category] = []
                items_by_category[item_category].append(" ".join(item_line_parts))

            # Assemble output for the character
            if not items_by_category:
                formatted_parts.append(f"{char_header}\n[No Relevant Items]") # Changed from Empty
            else:
                char_output_parts = [char_header]
                # Sort categories for consistent output
                sorted_categories = sorted(items_by_category.keys())
                for category in sorted_categories:
                    # Add category headers for clarity
                    char_output_parts.append(f"  [{category}]")
                    char_output_parts.extend([f"    {line}" for line in items_by_category[category]]) # Indent items under category
                formatted_parts.append("\n".join(char_output_parts))

        except json.JSONDecodeError:
            logger.error(f"Failed to decode inventory JSON for character '{char_name}'. Data: '{inventory_json[:100]}...'")
            formatted_parts.append(f"{char_header}\n[Error Decoding Inventory Data (New Format)]")
        except Exception as e:
            logger.error(f"Unexpected error formatting inventory for '{char_name}': {e}", exc_info=True)
            formatted_parts.append(f"{char_header}\n[Error Formatting Inventory]")

    if not formatted_parts:
        return "[No Inventory Data Available]"

    return "\n\n".join(formatted_parts)


# ==============================================================================
# === 2. Inventory State Modification Helper (Refactored for New Schema)    ===
# ==============================================================================

async def _apply_inventory_updates(
    cursor: sqlite3.Cursor,
    session_id: str,
    db_get_func: Callable[..., Coroutine[Any, Any, Optional[str]]],
    db_update_func: Callable[..., Coroutine[Any, Any, bool]],
    character_name: str,
    llm_updated_items: Dict[str, Dict[str, Any]], # Dict {"canonical_name": {item_schema}}
    inventory_log_func: Optional[Callable[[str, str, str], Coroutine[Any, Any, None]]] = None # Added logger
) -> bool:
    """
    Fetches current inventory state, merges updates from the LLM, logs changes,
    and saves the new state. Handles the new structured item schema.
    """
    func_logger = logging.getLogger(__name__ + '._apply_inventory_updates')
    if not cursor: func_logger.error(f"[{session_id}] Missing cursor for {character_name}."); return False
    if not llm_updated_items: func_logger.debug(f"[{session_id}] No LLM updates provided for {character_name}."); return True

    try:
        func_logger.debug(f"[{session_id}] Applying updates for {character_name}: {list(llm_updated_items.keys())}")

        # 1. Fetch Current State
        current_json_state: Optional[str] = await db_get_func(cursor=cursor, session_id=session_id, character_name=character_name)

        # 2. Parse Current State (or initialize empty)
        current_inventory_state: Dict[str, Dict[str, Any]] = {}
        if current_json_state:
            try:
                parsed_state = json.loads(current_json_state)
                if isinstance(parsed_state, dict) and all(isinstance(k, str) and isinstance(v, dict) for k, v in parsed_state.items()):
                     current_inventory_state = parsed_state
                     func_logger.debug(f"[{session_id}] Parsed existing inventory state for {character_name} ({len(current_inventory_state)} items).")
                else:
                     func_logger.warning(f"[{session_id}] Existing inventory data for {character_name} is not in the expected format. Starting fresh for merge.")
                     current_inventory_state = {}
            except json.JSONDecodeError:
                func_logger.error(f"[{session_id}] Failed to decode existing inventory JSON for {character_name}. Starting fresh for merge.")
                current_inventory_state = {}

        # --- Log Changes Before Applying ---
        log_messages = []
        if inventory_log_func and llm_updated_items:
            log_messages.append(f"Applying updates for character: {character_name}")
            for canonical_name, item_data in llm_updated_items.items():
                action = "Updating" if canonical_name in current_inventory_state else "Adding"
                status_change = f"Status -> {item_data.get('status', 'N/A')}"
                quantity_change = f"Qty -> {item_data.get('quantity', 'N/A')}"
                log_messages.append(f"  - {action} item: '{canonical_name}' ({quantity_change}, {status_change})")

            try:
                full_log_message = "\n".join(log_messages)
                # Use the specific log_type expected by the utility function
                await inventory_log_func(session_id, full_log_message, "Applied_Changes")
            except Exception as e_log:
                func_logger.error(f"[{session_id}] Failed to call inventory_log_func for {character_name}: {e_log}", exc_info=True)
        # --- End Logging Changes ---

        # 3. Merge LLM Updates
        for canonical_name, item_data in llm_updated_items.items():
            if isinstance(item_data, dict) and "quantity" in item_data and "category" in item_data and "status" in item_data:
                current_inventory_state[canonical_name] = item_data # Overwrite/add with final state from LLM
            else:
                 func_logger.warning(f"[{session_id}] Skipping invalid item data structure from LLM for '{canonical_name}' for {character_name}.")

        # 4. Serialize New State
        try:
            updated_inventory_json = json.dumps(current_inventory_state, separators=(',', ':'))
        except Exception as e_dump:
            func_logger.error(f"[{session_id}] Failed to serialize updated inventory for {character_name}: {e_dump}", exc_info=True)
            return False

        # 5. Save New State
        save_success = await db_update_func(
            cursor=cursor,
            session_id=session_id,
            character_name=character_name,
            inventory_data_json=updated_inventory_json
        )

        if not save_success:
            func_logger.error(f"[{session_id}] Failed to save updated inventory to DB for {character_name}.")
            return False

        func_logger.debug(f"[{session_id}] Successfully saved updated DB inventory for {character_name}.")
        return True

    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected exception in _apply_inventory_updates for {character_name}: {e}", exc_info=True)
        return False


# ==============================================================================
# === 3. Inventory Update Orchestration (Refactored for New Schema/Helper)  ===
# ==============================================================================

async def update_inventories_from_llm(
    cursor: sqlite3.Cursor,
    session_id: str,
    main_llm_response: str,
    user_query: str,
    recent_history_str: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    db_get_inventory_func: Callable[..., Coroutine[Any, Any, Optional[str]]],
    db_update_inventory_func: Callable[..., Coroutine[Any, Any, bool]],
    inventory_llm_config: Dict[str, Any],
    inventory_log_func: Optional[Callable[[str, str, str], Coroutine[Any, Any, None]]] = None # Added logger
) -> bool:
    """
    Orchestrates the inventory update process for a turn using the new structured schema.
    Calls the LLM to analyze changes, logs the raw response, parses the structured JSON,
    and calls the helper function to apply updates sequentially per character.
    """
    func_logger = logging.getLogger(__name__ + '.update_inventories_from_llm')
    func_logger.info(f"[{session_id}] Starting post-turn inventory update analysis (New Schema)...")
    if not cursor: func_logger.error(f"[{session_id}] SQLite cursor missing."); return False
    if not all([session_id, main_llm_response, user_query, llm_call_func, db_get_inventory_func, db_update_inventory_func, inventory_llm_config]):
        func_logger.error(f"[{session_id}] Missing required arguments."); return False

    # --- Call Inventory LLM ---
    inv_template = inventory_llm_config.get("prompt_template")
    inv_url = inventory_llm_config.get("url"); inv_key = inventory_llm_config.get("key")
    inv_temp = inventory_llm_config.get("temp", 0.3)

    if not inv_url or not inv_key: func_logger.error(f"[{session_id}] Inventory LLM URL/Key missing."); return False
    if not inv_template: func_logger.error(f"[{session_id}] Inventory LLM prompt_template missing."); return False

    # Dynamically import formatter to avoid circular dependency issues if utils imports inventory
    try: from .prompting import format_inventory_update_prompt
    except ImportError: func_logger.error(f"[{session_id}] Cannot import format_inventory_update_prompt."); return False

    try:
        prompt_text = format_inventory_update_prompt(
            main_llm_response=main_llm_response, user_query=user_query,
            recent_history_str=recent_history_str, template=inv_template
        )
    except Exception as e_fmt: func_logger.error(f"[{session_id}] Error formatting inventory prompt: {e_fmt}"); return False

    if not prompt_text or "[Error" in prompt_text: func_logger.error(f"[{session_id}] Failed format inventory prompt: {prompt_text}"); return False

    inv_payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    caller_info = f"InventoryUpdater_{session_id}"

    func_logger.info(f"[{session_id}] Calling Inventory LLM (New Schema)...")
    success, response_or_error = await llm_call_func(
        api_url=inv_url, api_key=inv_key, payload=inv_payload,
        temperature=inv_temp, timeout=90, caller_info=caller_info
    )

    if not success or not isinstance(response_or_error, str):
        error_details = str(response_or_error);
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"[{session_id}] Inventory LLM call failed. Details: {error_details}"); return False

    inventory_llm_output = response_or_error.strip()
    func_logger.debug(f"[{session_id}] Inventory LLM raw output: {inventory_llm_output[:500]}...")

    # --- Log the RAW LLM Response ---
    if inventory_log_func:
        try:
            # Use the specific log_type expected by the utility function
            await inventory_log_func(session_id, inventory_llm_output, "LLM_Raw_Output")
        except Exception as e_log_raw:
            func_logger.error(f"[{session_id}] Failed to call inventory_log_func for raw LLM output: {e_log_raw}", exc_info=True)
    # --- End Logging Raw Response ---

    # --- Parse LLM Response ---
    parsed_updates: Optional[Dict] = None
    try:
        json_string_to_parse = inventory_llm_output
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_string_to_parse, re.IGNORECASE)
        if match: json_string_to_parse = match.group(1).strip(); func_logger.debug(f"[{session_id}] Stripped markdown fences.")
        else: func_logger.debug(f"[{session_id}] No markdown fences found.")

        if json_string_to_parse == '{}': parsed_updates = {"character_updates": []}
        else: parsed_updates = json.loads(json_string_to_parse)

        if not isinstance(parsed_updates, dict) or "character_updates" not in parsed_updates or not isinstance(parsed_updates.get("character_updates"), list):
             if isinstance(parsed_updates, list): # Allow list as fallback
                  func_logger.warning(f"[{session_id}] Inventory LLM output was a list, wrapping.")
                  parsed_updates = {"character_updates": parsed_updates}
             else:
                  func_logger.error(f"[{session_id}] Inventory LLM JSON missing 'character_updates' list or not a dict. Type: {type(parsed_updates)}")
                  return False
    except json.JSONDecodeError as e:
        func_logger.error(f"[{session_id}] Failed decode Inventory LLM JSON: {e}. String: '{json_string_to_parse[:500]}...'"); return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error processing Inventory LLM response: {e}", exc_info=True); return False

    # --- Apply Updates Sequentially ---
    updates_to_process = parsed_updates.get("character_updates", [])
    if not updates_to_process:
        func_logger.info(f"[{session_id}] Inventory LLM indicated no inventory changes.")
        return True

    func_logger.info(f"[{session_id}] Inventory LLM detected updates for {len(updates_to_process)} character(s). Processing...")

    overall_success = True; successful_updates_count = 0; total_updates_attempted = 0

    for update_block in updates_to_process:
        if not isinstance(update_block, dict): func_logger.warning(f"[{session_id}] Skipping invalid item in list: {update_block}"); continue

        char_name = update_block.get("character_name")
        updated_items_dict = update_block.get("updated_items")

        if not char_name or not isinstance(updated_items_dict, dict): func_logger.warning(f"[{session_id}] Skipping update block, missing name or items: {update_block}"); continue

        total_updates_attempted += 1
        try:
            # Pass inventory_log_func down to the helper
            update_success = await _apply_inventory_updates(
                cursor=cursor, session_id=session_id,
                db_get_func=db_get_inventory_func, db_update_func=db_update_inventory_func,
                character_name=char_name, llm_updated_items=updated_items_dict,
                inventory_log_func=inventory_log_func # Pass it here
            )
            if update_success: successful_updates_count += 1
            else: overall_success = False; func_logger.warning(f"[{session_id}] Failed apply update for {char_name}.")
        except Exception as e_apply:
            func_logger.error(f"[{session_id}] Exception during _apply_inventory_updates for {char_name}: {e_apply}", exc_info=True)
            overall_success = False

    func_logger.info(f"[{session_id}] Finished processing inventory updates. Attempted: {total_updates_attempted}, Successful Saves: {successful_updates_count}.")

    return overall_success and (successful_updates_count == total_updates_attempted)

# [[END FINAL COMPLETE inventory.py]]