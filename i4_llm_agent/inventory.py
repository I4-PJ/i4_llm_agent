# [[START MODIFIED inventory.py - Update Function Implementation]]
# i4_llm_agent/inventory.py

import logging
import json
import asyncio # Keep asyncio for the main function signature
import re
import sqlite3
from typing import Dict, Optional, Callable, Any, Tuple, Union, Coroutine

from .prompting import format_inventory_update_prompt, DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT

logger = logging.getLogger(__name__) # i4_llm_agent.inventory

# ==============================================================================
# === 1. Inventory Formatting (No changes needed)                           ===
# ==============================================================================
def format_inventory_for_prompt(session_inventories: Dict[str, str]) -> str:
    """
    Formats the inventory data retrieved from the database into a single,
    human-readable string suitable for context injection.
    """
    if not session_inventories:
        return "[No Inventory Data Available]"
    formatted_parts = []
    sorted_char_names = sorted(session_inventories.keys())
    for char_name in sorted_char_names:
        inventory_json = session_inventories[char_name]
        char_header = f"{char_name}'s Inventory:"
        formatted_items = []
        if not inventory_json or inventory_json.strip() == '{}':
            formatted_parts.append(f"{char_header}\n[Empty]")
            continue
        try:
            inventory_data = json.loads(inventory_json)
            if not isinstance(inventory_data, dict) or not inventory_data:
                formatted_parts.append(f"{char_header}\n[Empty]")
                continue
            sorted_item_names = sorted(inventory_data.keys())
            for item_name in sorted_item_names:
                item_details = inventory_data[item_name]
                if isinstance(item_details, dict):
                    quantity = item_details.get("quantity", 0)
                    description = item_details.get("description")
                    if quantity > 0:
                        item_line = f"- {item_name}: {quantity}"
                        if description and isinstance(description, str) and description.strip():
                            item_line += f" ({description.strip()})"
                        formatted_items.append(item_line)
                elif isinstance(item_details, int) and item_details > 0:
                     formatted_items.append(f"- {item_name}: {item_details}")
                else:
                    logger.warning(f"Unexpected item detail format for '{item_name}' in {char_name}'s inventory: {item_details}")
            if formatted_items:
                formatted_parts.append(f"{char_header}\n" + "\n".join(formatted_items))
            else:
                formatted_parts.append(f"{char_header}\n[Empty]")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode inventory JSON for character '{char_name}'. Data: '{inventory_json[:100]}...'")
            formatted_parts.append(f"{char_header}\n[Error Decoding Inventory Data]")
        except Exception as e:
            logger.error(f"Unexpected error formatting inventory for '{char_name}': {e}", exc_info=True)
            formatted_parts.append(f"{char_header}\n[Error Formatting Inventory]")
    if not formatted_parts:
        return "[No Inventory Data Available]"
    return "\n\n".join(formatted_parts)

# ==============================================================================
# === 2. Inventory State Modification (No changes needed)                    ===
# ==============================================================================
def _modify_inventory_json(
    current_inventory_json: Optional[str],
    action: str,
    item_name: str,
    quantity: int = 1,
    description: Optional[str] = None
) -> Optional[str]:
    """
    Internal helper to modify inventory data stored as a JSON string.
    """
    if not item_name or not isinstance(item_name, str) or not item_name.strip():
        logger.error("_modify_inventory_json: Invalid or empty item_name provided.")
        return None
    safe_item_name = item_name.strip()
    valid_actions = ["add", "remove", "set_quantity"]
    if action not in valid_actions:
        logger.error(f"_modify_inventory_json: Invalid action '{action}'. Must be one of {valid_actions}.")
        return None
    if not isinstance(quantity, int) or quantity < 0:
        logger.error(f"_modify_inventory_json: Invalid quantity '{quantity}'. Must be a non-negative integer.")
        return None
    inventory_data: Dict[str, Dict[str, Any]] = {}
    if current_inventory_json and current_inventory_json.strip() != '{}':
        try:
            inventory_data = json.loads(current_inventory_json)
            if not isinstance(inventory_data, dict):
                logger.error("_modify_inventory_json: Current inventory JSON is not a dictionary.")
                return None
        except json.JSONDecodeError:
            logger.error("_modify_inventory_json: Failed to decode current inventory JSON.")
            return None
    item_exists = safe_item_name in inventory_data
    if action == "add":
        if item_exists:
            if isinstance(inventory_data[safe_item_name], dict):
                current_qty = inventory_data[safe_item_name].get("quantity", 0)
                if isinstance(current_qty, int):
                    inventory_data[safe_item_name]["quantity"] = current_qty + quantity
                    if description is not None:
                        inventory_data[safe_item_name]["description"] = description.strip() if isinstance(description, str) else description
                else:
                    logger.warning(f"Invalid existing quantity for '{safe_item_name}'. Overwriting with new quantity.")
                    inventory_data[safe_item_name]["quantity"] = quantity
            else:
                 logger.warning(f"Unexpected existing data format for '{safe_item_name}'. Overwriting.")
                 inventory_data[safe_item_name] = {"quantity": quantity}
                 if description is not None:
                    inventory_data[safe_item_name]["description"] = description.strip() if isinstance(description, str) else description
        else:
            inventory_data[safe_item_name] = {"quantity": quantity}
            if description is not None:
                inventory_data[safe_item_name]["description"] = description.strip() if isinstance(description, str) else description
    elif action == "remove":
        if item_exists:
            if isinstance(inventory_data[safe_item_name], dict):
                current_qty = inventory_data[safe_item_name].get("quantity", 0)
                if isinstance(current_qty, int):
                    new_qty = current_qty - quantity
                    if new_qty <= 0:
                        del inventory_data[safe_item_name]
                    else:
                        inventory_data[safe_item_name]["quantity"] = new_qty
                else:
                    logger.warning(f"Cannot remove from '{safe_item_name}': Invalid existing quantity. Removing item.")
                    del inventory_data[safe_item_name]
            else:
                 logger.warning(f"Cannot remove from '{safe_item_name}': Unexpected existing data format. Removing item.")
                 del inventory_data[safe_item_name]
        else:
            logger.warning(f"Cannot remove '{safe_item_name}': Item not found in inventory.")
    elif action == "set_quantity":
        if quantity <= 0:
            if item_exists:
                del inventory_data[safe_item_name]
        else:
            inventory_data[safe_item_name] = {"quantity": quantity}
            if description is not None:
                inventory_data[safe_item_name]["description"] = description.strip() if isinstance(description, str) else description
    try:
        updated_inventory_json = json.dumps(inventory_data, indent=None, separators=(',', ':'))
        return updated_inventory_json
    except Exception as e:
        logger.error(f"Failed to dump updated inventory data to JSON: {e}", exc_info=True)
        return None

# ==============================================================================
# === 3. Post-Turn Inventory Update Orchestration (MODIFIED SECTION)        ===
# ==============================================================================

# --- 3.1: Update Inventories from LLM Response (MODIFIED: Sequential Processing) ---
async def update_inventories_from_llm(
    cursor: sqlite3.Cursor, # Still needs cursor for DB calls
    session_id: str,
    main_llm_response: str,
    user_query: str,
    recent_history_str: str,
    llm_call_func: Callable[..., Coroutine[Any, Any, Tuple[bool, Union[str, Dict]]]],
    db_get_inventory_func: Callable[..., Coroutine[Any, Any, Optional[str]]],
    db_update_inventory_func: Callable[..., Coroutine[Any, Any, bool]],
    inventory_llm_config: Dict[str, Any]
) -> bool:
    func_logger = logging.getLogger(__name__ + '.update_inventories_from_llm')
    func_logger.info(f"[{session_id}] Starting post-turn inventory update analysis...")
    if not cursor: func_logger.error(f"[{session_id}] SQLite cursor missing for inventory update."); return False
    if not all([session_id, main_llm_response, user_query, llm_call_func, db_get_inventory_func, db_update_inventory_func, inventory_llm_config]):
        func_logger.error(f"[{session_id}] Missing required arguments for inventory update."); return False

    inv_template = inventory_llm_config.get("prompt_template", DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT)
    inv_url = inventory_llm_config.get("url"); inv_key = inventory_llm_config.get("key"); inv_temp = inventory_llm_config.get("temp", 0.3)
    if not inv_url or not inv_key: func_logger.error(f"[{session_id}] Inventory LLM URL or Key missing in config."); return False
    prompt_text = format_inventory_update_prompt(main_llm_response=main_llm_response, user_query=user_query, recent_history_str=recent_history_str, template=inv_template)
    if not prompt_text or "[Error" in prompt_text: func_logger.error(f"[{session_id}] Failed to format inventory update prompt: {prompt_text}"); return False
    inv_payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    caller_info = f"InventoryUpdater_{session_id}"
    func_logger.info(f"[{session_id}] Calling Inventory LLM...");
    success, response_or_error = await llm_call_func(api_url=inv_url, api_key=inv_key, payload=inv_payload, temperature=inv_temp, timeout=60, caller_info=caller_info)
    if not success or not isinstance(response_or_error, str):
        error_details = str(response_or_error);
        if isinstance(response_or_error, dict): error_details = f"Type: {response_or_error.get('error_type')}, Msg: {response_or_error.get('message')}"
        func_logger.error(f"[{session_id}] Inventory LLM call failed. Details: {error_details}"); return False
    inventory_llm_output = response_or_error.strip(); func_logger.debug(f"[{session_id}] Inventory LLM raw output: {inventory_llm_output[:500]}...")

    updates_processed_count = 0
    successful_updates_count = 0 # <<< Track successful updates
    total_actions_processed = 0   # <<< Track total actions attempted

    try:
        json_string_to_parse = inventory_llm_output
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_string_to_parse, re.IGNORECASE)
        if match: json_string_to_parse = match.group(1).strip(); func_logger.debug(f"[{session_id}] Stripped markdown fences.")
        else: func_logger.debug(f"[{session_id}] No markdown fences found.")
        parsed_data = json.loads(json_string_to_parse)
        if not isinstance(parsed_data, dict): func_logger.error(f"[{session_id}] Inventory LLM parsed output is not a JSON dictionary. Type: {type(parsed_data)}"); return False
        updates_list = parsed_data.get("updates")
        if not isinstance(updates_list, list):
            func_logger.error(f"[{session_id}] Inventory LLM JSON missing 'updates' list or not a list.")
            if updates_list is None and len(parsed_data.keys()) == 0: func_logger.info(f"[{session_id}] Inventory LLM indicated no changes (empty JSON object)."); return True
            elif updates_list == []: func_logger.info(f"[{session_id}] Inventory LLM indicated no changes (empty updates list)."); return True
            else: return False
        if not updates_list: func_logger.info(f"[{session_id}] Inventory LLM detected no inventory changes."); return True
        func_logger.info(f"[{session_id}] Inventory LLM detected {len(updates_list)} potential update(s). Processing sequentially...")

        # --- MODIFICATION START: Sequential Loop ---
        for update_action in updates_list:
            total_actions_processed += 1 # Count attempt
            if not isinstance(update_action, dict): func_logger.warning(f"[{session_id}] Skipping invalid item in updates list: {update_action}"); continue
            char_name = update_action.get("character_name"); action = update_action.get("action")
            item_name = update_action.get("item_name"); quantity = update_action.get("quantity")
            description = update_action.get("description")
            if not all([char_name, action, item_name]) or not isinstance(quantity, int):
                 func_logger.warning(f"[{session_id}] Skipping update due to missing/invalid fields: {update_action}"); continue

            # Call helper sequentially and wait for it to complete
            try:
                update_success = await _process_single_inventory_update(
                     cursor, # 1
                     session_id, # 2
                     db_get_inventory_func, # 3
                     db_update_inventory_func, # 4
                     char_name, # 5
                     action, # 6
                     item_name, # 7
                     quantity, # 8
                     description # 9
                 )
                if update_success:
                    successful_updates_count += 1
                else:
                    func_logger.warning(f"[{session_id}] Failed to process update for {char_name} ({action} {item_name}).")
            except Exception as e_process:
                 # Catch exceptions from the helper itself
                 func_logger.error(f"[{session_id}] Error processing update for {char_name} ({action} {item_name}): {e_process}", exc_info=True)

        # --- MODIFICATION END: Sequential Loop ---

        # No need for gather or processing results list
        updates_processed_count = successful_updates_count # Use the counter from the loop
        func_logger.info(f"[{session_id}] Processed {total_actions_processed} update actions sequentially. Successful DB updates: {successful_updates_count}.")

    except json.JSONDecodeError as e: func_logger.error(f"[{session_id}] Failed to decode JSON response from Inventory LLM after cleanup: {e}. String was: '{json_string_to_parse[:500]}...'"); return False
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error processing Inventory LLM response: {e}", exc_info=True); return False
    # Return True if at least one update was successfully processed
    return successful_updates_count > 0

# --- Helper for processing a single update action (Corrected Definition) ---
async def _process_single_inventory_update(
    cursor: sqlite3.Cursor,
    session_id: str,
    db_get_func: Callable[..., Coroutine[Any, Any, Optional[str]]],
    db_update_func: Callable[..., Coroutine[Any, Any, bool]],
    char_name: str,
    action: str,
    item_name: str,
    quantity: int,
    description: Optional[str]
) -> bool:
    """ Fetches current state, modifies JSON, saves new state for one update. """
    func_logger = logging.getLogger(__name__ + '._process_single_inventory_update')
    if not cursor: func_logger.error(f"[{session_id}] Missing cursor in _process_single_inventory_update for {char_name}."); return False
    try:
        func_logger.debug(f"[{session_id}] Processing update for {char_name}: {action} {quantity} x {item_name}")
        # Call DB functions *with* the cursor
        current_json_state = await db_get_func(cursor=cursor, session_id=session_id, character_name=char_name)

        new_json_state = _modify_inventory_json(
            current_inventory_json=current_json_state,
            action=action, item_name=item_name, quantity=quantity, description=description
        )

        if new_json_state is None:
            func_logger.error(f"[{session_id}] Failed to generate new inventory JSON for {char_name} ({item_name}).")
            return False

        save_success = await db_update_func(
            cursor=cursor, # Pass cursor
            session_id=session_id, character_name=char_name, inventory_data_json=new_json_state
        )

        if not save_success: func_logger.error(f"[{session_id}] Failed to save updated inventory to DB for {char_name}."); return False

        func_logger.debug(f"[{session_id}] Successfully updated DB inventory for {char_name}.")
        return True

    except TypeError as te:
        func_logger.error(f"[{session_id}] TypeError in _process_single_inventory_update for {char_name}: {te}. Args passed: cursor={bool(cursor)}, sid={session_id}, char={char_name}", exc_info=True)
        return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Exception in _process_single_inventory_update for {char_name}: {e}", exc_info=True)
        return False