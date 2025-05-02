# === START MODIFIED FILE: i4_llm_agent/database.py ===
# i4_llm_agent/database.py

import logging
import sqlite3
import asyncio
import re
import os
import json # Added for world state, scene state, and aged summary metadata
from datetime import datetime, timezone # Added for inventory/world/scene/aged timestamp
from typing import (
    Optional, Dict, List, Tuple, Union, Any, Callable, Coroutine, Sequence
)

# --- ChromaDB Optional Import ---
try:
    import chromadb
    from chromadb.api.models.Collection import Collection as ChromaCollectionType
    from chromadb.errors import InvalidDimensionException
    ChromaEmbeddingFunction = Callable[[Sequence[str]], List[List[float]]]
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    ChromaCollectionType = None # type: ignore
    InvalidDimensionException = Exception # type: ignore
    ChromaEmbeddingFunction = Callable # type: ignore
    logging.getLogger(__name__).warning(
        "chromadb library not found. ChromaDB functions will not be available."
    )

logger = logging.getLogger(__name__) # i4_llm_agent.database

# --- Constants ---
T1_SUMMARY_TABLE_NAME = "tier1_text_summaries"
RAG_CACHE_TABLE_NAME = "session_rag_cache"
INVENTORY_TABLE_NAME = "character_inventory"
WORLD_STATE_TABLE_NAME = "session_world_state"
SCENE_STATE_TABLE_NAME = "session_scene_state"
# <<< NEW: Aged Summary Table Name >>>
AGED_SUMMARY_TABLE_NAME = "aged_summaries"

# ==============================================================================
# === 1. SQLite Initialization (Modified)                                    ===
# ==============================================================================

# --- 1.1: Sync - Initialize Inventory Table (Existing - Unchanged) ---
def _sync_initialize_inventory_table(cursor: sqlite3.Cursor) -> bool:
    """
    Synchronously initializes the character_inventory table.
    Stores inventory data as JSON per character per session.
    """
    func_logger = logging.getLogger(__name__ + '._sync_initialize_inventory_table')
    if not cursor:
        func_logger.error("SQLite cursor is not available for inventory table init.")
        return False
    try:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {INVENTORY_TABLE_NAME} (
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                inventory_data TEXT,
                last_updated_utc REAL NOT NULL,
                PRIMARY KEY (session_id, character_name)
            )""")
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_inv_session ON {INVENTORY_TABLE_NAME} (session_id)"
        )
        func_logger.debug(f"Table '{INVENTORY_TABLE_NAME}' initialized successfully.")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"SQLite error initializing table '{INVENTORY_TABLE_NAME}': {e}")
        return False
    except Exception as e:
        func_logger.error(f"Unexpected error initializing table '{INVENTORY_TABLE_NAME}': {e}")
        return False

# --- 1.2: Async - Initialize Inventory Table (Existing - Unchanged) ---
async def initialize_inventory_table(cursor: sqlite3.Cursor) -> bool:
    """Async wrapper to initialize the character_inventory table."""
    return await asyncio.to_thread(_sync_initialize_inventory_table, cursor)


# --- 1.2.1: Sync - Initialize World State Table (Existing - Unchanged) ---
def _sync_initialize_world_state_table(cursor: sqlite3.Cursor) -> bool:
    """
    Synchronously initializes the session_world_state table.
    Stores key-value pairs for world state per session (e.g., season, weather, day, time).
    """
    func_logger = logging.getLogger(__name__ + '._sync_initialize_world_state_table')
    if not cursor:
        func_logger.error("SQLite cursor is not available for world state table init.")
        return False
    try:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {WORLD_STATE_TABLE_NAME} (
                session_id TEXT PRIMARY KEY,
                current_season TEXT,
                current_weather TEXT,
                current_day INTEGER,
                time_of_day TEXT,
                last_updated_utc REAL NOT NULL
            )""")
        func_logger.debug(f"Table '{WORLD_STATE_TABLE_NAME}' initialized successfully.")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"SQLite error initializing table '{WORLD_STATE_TABLE_NAME}': {e}")
        return False
    except Exception as e:
        func_logger.error(f"Unexpected error initializing table '{WORLD_STATE_TABLE_NAME}': {e}")
        return False

# --- 1.2.2: Async - Initialize World State Table (Existing - Unchanged wrapper) ---
async def initialize_world_state_table(cursor: sqlite3.Cursor) -> bool:
    """Async wrapper to initialize the session_world_state table."""
    return await asyncio.to_thread(_sync_initialize_world_state_table, cursor)


# --- 1.2.3: Sync - Initialize Scene State Table (Existing - Unchanged) ---
def _sync_initialize_scene_state_table(cursor: sqlite3.Cursor) -> bool:
    """
    Synchronously initializes the session_scene_state table.
    Stores scene keywords (JSON) and description per session.
    """
    func_logger = logging.getLogger(__name__ + '._sync_initialize_scene_state_table')
    if not cursor:
        func_logger.error("SQLite cursor is not available for scene state table init.")
        return False
    try:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {SCENE_STATE_TABLE_NAME} (
                session_id TEXT PRIMARY KEY,
                scene_keywords_json TEXT,
                scene_description TEXT,
                last_updated_utc REAL NOT NULL
            )""")
        func_logger.debug(f"Table '{SCENE_STATE_TABLE_NAME}' initialized successfully.")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"SQLite error initializing table '{SCENE_STATE_TABLE_NAME}': {e}")
        return False
    except Exception as e:
        func_logger.error(f"Unexpected error initializing table '{SCENE_STATE_TABLE_NAME}': {e}")
        return False

# --- 1.2.4: Async - Initialize Scene State Table (Existing - Unchanged) ---
async def initialize_scene_state_table(cursor: sqlite3.Cursor) -> bool:
    """Async wrapper to initialize the session_scene_state table."""
    return await asyncio.to_thread(_sync_initialize_scene_state_table, cursor)


# --- 1.2.5: Sync - Initialize Aged Summaries Table (NEW) ---
def _sync_initialize_aged_summaries_table(cursor: sqlite3.Cursor) -> bool:
    """
    Synchronously initializes the aged_summaries table.
    Stores condensed summaries representing batches of older T1 summaries.
    """
    func_logger = logging.getLogger(__name__ + '._sync_initialize_aged_summaries_table')
    if not cursor:
        func_logger.error("SQLite cursor is not available for aged summaries table init.")
        return False
    try:
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {AGED_SUMMARY_TABLE_NAME} (
                id TEXT PRIMARY KEY,              -- Unique ID for the aged summary
                session_id TEXT NOT NULL,         -- Session this belongs to
                aged_summary_text TEXT NOT NULL,  -- The condensed summary text
                creation_timestamp_utc REAL NOT NULL, -- When this aged summary was created
                original_batch_start_index INTEGER, -- Turn start index of the first T1 in the batch
                original_batch_end_index INTEGER,   -- Turn end index of the last T1 in the batch
                original_t1_count INTEGER,          -- How many T1 summaries were condensed
                original_t1_ids_json TEXT           -- Optional: JSON list of original T1 IDs
            )""")
        # Index for efficient retrieval by session and time/batch end index
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_aged_session_ts ON {AGED_SUMMARY_TABLE_NAME} (session_id, creation_timestamp_utc)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_aged_session_end_idx ON {AGED_SUMMARY_TABLE_NAME} (session_id, original_batch_end_index)"
        )
        func_logger.debug(f"Table '{AGED_SUMMARY_TABLE_NAME}' initialized successfully.")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"SQLite error initializing table '{AGED_SUMMARY_TABLE_NAME}': {e}")
        return False
    except Exception as e:
        func_logger.error(f"Unexpected error initializing table '{AGED_SUMMARY_TABLE_NAME}': {e}")
        return False

# --- 1.2.6: Async - Initialize Aged Summaries Table (NEW) ---
async def initialize_aged_summaries_table(cursor: sqlite3.Cursor) -> bool:
    """Async wrapper to initialize the aged_summaries table."""
    return await asyncio.to_thread(_sync_initialize_aged_summaries_table, cursor)


# --- 1.3: Sync - Initialize ALL Tables (MODIFIED) ---
def _sync_initialize_sqlite_tables(cursor: sqlite3.Cursor) -> bool:
    """
    Initializes all necessary SQLite tables (T1, RAG Cache, Inventory, World State, Scene State, Aged Summaries).
    MODIFIED to include aged summary table initialization.
    """
    func_logger = logging.getLogger(__name__ + '._sync_initialize_sqlite_tables')
    if not cursor:
        func_logger.error("SQLite cursor is not available.")
        return False
    all_success = True
    try:
        # --- Initialize T1 Summary Table (Existing) ---
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {T1_SUMMARY_TABLE_NAME} (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, user_id TEXT,
                summary_text TEXT NOT NULL, timestamp_utc REAL NOT NULL,
                timestamp_iso TEXT, turn_start_index INTEGER, turn_end_index INTEGER,
                char_length INTEGER, config_t0_token_limit INTEGER,
                config_t1_chunk_target INTEGER, calculated_prompt_tokens INTEGER,
                t0_end_index_at_summary INTEGER
            )""")
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_t1_session_ts ON {T1_SUMMARY_TABLE_NAME} (session_id, timestamp_utc)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_t1_session_end_idx ON {T1_SUMMARY_TABLE_NAME} (session_id, turn_end_index)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_t1_session_start_end ON {T1_SUMMARY_TABLE_NAME} (session_id, turn_start_index, turn_end_index)"
        )
        func_logger.debug(f"Table '{T1_SUMMARY_TABLE_NAME}' initialized successfully.")

        # --- Initialize RAG Cache Table (Existing) ---
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {RAG_CACHE_TABLE_NAME} (
                session_id TEXT PRIMARY KEY, cached_context TEXT NOT NULL,
                last_updated_utc REAL NOT NULL, last_updated_iso TEXT
            )""")
        func_logger.debug(f"Table '{RAG_CACHE_TABLE_NAME}' initialized successfully.")

        # --- Initialize Inventory Table (Existing Call) ---
        inventory_init_success = _sync_initialize_inventory_table(cursor)
        if not inventory_init_success:
             func_logger.error("Failed to initialize inventory table during main init.")
             all_success = False # Mark failure but continue

        # --- Initialize World State Table (Existing Call) ---
        world_state_init_success = _sync_initialize_world_state_table(cursor)
        if not world_state_init_success:
             func_logger.error("Failed to initialize world state table during main init.")
             all_success = False # Mark failure

        # --- Initialize Scene State Table (Existing Call) ---
        scene_state_init_success = _sync_initialize_scene_state_table(cursor)
        if not scene_state_init_success:
             func_logger.error("Failed to initialize scene state table during main init.")
             all_success = False # Mark failure

        # --- Initialize Aged Summaries Table (NEW CALL) ---
        aged_summaries_init_success = _sync_initialize_aged_summaries_table(cursor)
        if not aged_summaries_init_success:
             func_logger.error("Failed to initialize aged summaries table during main init.")
             all_success = False # Mark failure

        # --- Return True only if all initializations were okay ---
        if all_success:
            func_logger.info("All required SQLite tables initialized (or confirmed existed).")
        else:
            func_logger.error("One or more SQLite tables failed to initialize.")
        return all_success

    except sqlite3.Error as e:
        func_logger.error(f"SQLite error initializing tables: {e}")
        return False
    except Exception as e:
        func_logger.error(f"Unexpected error initializing tables: {e}")
        return False

# --- 1.4: Async - Initialize ALL Tables (Modified - Calls modified sync version) ---
async def initialize_sqlite_tables(cursor: sqlite3.Cursor) -> bool:
    """Async wrapper to initialize all necessary SQLite tables."""
    return await asyncio.to_thread(_sync_initialize_sqlite_tables, cursor)


# ==============================================================================
# === 2. SQLite Tier 1 Summary Operations (MODIFIED get_recent)              ===
# ==============================================================================
# --- 2.1: Sync - Add T1 Summary (Unchanged) ---
def _sync_add_tier1_summary(
    cursor: sqlite3.Cursor,
    summary_id: str, session_id: str, user_id: str, summary_text: str, metadata: Dict
) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_add_tier1_summary')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return False
    try:
        cursor.execute(
            f"""INSERT INTO {T1_SUMMARY_TABLE_NAME} (id, session_id, user_id, summary_text, timestamp_utc, timestamp_iso, turn_start_index, turn_end_index, char_length, config_t0_token_limit, config_t1_chunk_target, calculated_prompt_tokens, t0_end_index_at_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary_id, session_id, user_id, summary_text,
                metadata.get("timestamp_utc"), metadata.get("timestamp_iso"),
                metadata.get("turn_start_index"), metadata.get("turn_end_index"),
                metadata.get("char_length"), metadata.get("config_t0_token_limit"),
                metadata.get("config_t1_chunk_target"), metadata.get("calculated_prompt_tokens"),
                metadata.get("t0_end_index_at_summary")
            )
        )
        return True
    except sqlite3.IntegrityError as e: func_logger.error(f"[{session_id}] IntegrityError adding T1 {summary_id} (duplicate?): {e}"); return False
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error adding T1 {summary_id}: {e}"); return False
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error adding T1 {summary_id}: {e}"); return False

# --- 2.2: Async - Add T1 Summary (Unchanged) ---
async def add_tier1_summary(
    cursor: sqlite3.Cursor,
    summary_id: str, session_id: str, user_id: str, summary_text: str, metadata: Dict
) -> bool:
    """Async wrapper to add a Tier 1 summary."""
    return await asyncio.to_thread(_sync_add_tier1_summary, cursor, summary_id, session_id, user_id, summary_text, metadata)

# --- 2.3: Sync - Get Recent T1 Summaries (MODIFIED Return Type) ---
def _sync_get_recent_tier1_summaries(
    cursor: sqlite3.Cursor, session_id: str, limit: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieves recent T1 summaries with metadata for sorting.
    Returns list of (summary_text, metadata_dict) tuples, ordered chronologically (oldest first).
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_recent_tier1_summaries')
    results: List[Tuple[str, Dict[str, Any]]] = []
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return results
    if limit <= 0: return results
    try:
        # Fetch needed columns: summary_text and turn_end_index for sorting
        cursor.execute(
            f"SELECT summary_text, turn_end_index FROM {T1_SUMMARY_TABLE_NAME} WHERE session_id = ? ORDER BY timestamp_utc DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        for row in rows:
            summary_text, turn_end_index = row
            metadata = {"turn_end_index": turn_end_index}
            results.append((summary_text, metadata))

        results.reverse() # Return in chronological order (oldest first in list)
        return results
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error getting recent T1: {e}"); return []
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error getting recent T1: {e}"); return []

# --- 2.4: Async - Get Recent T1 Summaries (MODIFIED Return Type) ---
async def get_recent_tier1_summaries(
    cursor: sqlite3.Cursor, session_id: str, limit: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Async wrapper to get recent Tier 1 summaries with metadata.
    Returns list of (summary_text, metadata_dict) tuples, ordered chronologically.
    """
    return await asyncio.to_thread(_sync_get_recent_tier1_summaries, cursor, session_id, limit)

# --- 2.5: Sync - Get T1 Summary Count (Unchanged) ---
def _sync_get_tier1_summary_count(cursor: sqlite3.Cursor, session_id: str) -> int:
    func_logger = logging.getLogger(__name__ + '._sync_get_tier1_summary_count')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return -1
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {T1_SUMMARY_TABLE_NAME} WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        return result[0] if result else 0
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error counting T1: {e}"); return -1
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error counting T1: {e}"); return -1

# --- 2.6: Async - Get T1 Summary Count (Unchanged) ---
async def get_tier1_summary_count(cursor: sqlite3.Cursor, session_id: str) -> int:
    """Async wrapper to count Tier 1 summaries."""
    return await asyncio.to_thread(_sync_get_tier1_summary_count, cursor, session_id)

# --- 2.7: Sync - Get Oldest T1 Summary (Unchanged - used by T2 Push) ---
def _sync_get_oldest_tier1_summary(cursor: sqlite3.Cursor, session_id: str) -> Optional[Tuple[str, str, Dict]]:
    func_logger = logging.getLogger(__name__ + '._sync_get_oldest_tier1_summary')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return None
    try:
        cursor.execute(
            f"""SELECT id, summary_text, session_id, user_id, timestamp_utc, timestamp_iso, turn_start_index, turn_end_index, char_length, config_t0_token_limit, config_t1_chunk_target, calculated_prompt_tokens, t0_end_index_at_summary FROM {T1_SUMMARY_TABLE_NAME} WHERE session_id = ? ORDER BY timestamp_utc ASC LIMIT 1""",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            sid, s_text, sess_id, u_id, ts_utc, ts_iso, s_idx, e_idx, length, t0_lim, t1_targ, p_tok, t0_end = row
            metadata = {
                "session_id": sess_id, "user_id": u_id, "timestamp_utc": ts_utc, "timestamp_iso": ts_iso,
                "turn_start_index": s_idx, "turn_end_index": e_idx, "char_length": length,
                "config_t0_token_limit": t0_lim, "config_t1_chunk_target": t1_targ,
                "calculated_prompt_tokens": p_tok, "t0_end_index_at_summary": t0_end,
                "doc_type": "llm_summary" # Add standard field
            }
            return sid, s_text, metadata
        else:
            return None
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error getting oldest T1: {e}"); return None
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error getting oldest T1: {e}"); return None

# --- 2.8: Async - Get Oldest T1 Summary (Unchanged) ---
async def get_oldest_tier1_summary(cursor: sqlite3.Cursor, session_id: str) -> Optional[Tuple[str, str, Dict]]:
    """Async wrapper to get the oldest Tier 1 summary."""
    return await asyncio.to_thread(_sync_get_oldest_tier1_summary, cursor, session_id)

# --- 2.9: Sync - Delete T1 Summary (Unchanged - used by T2 Push) ---
def _sync_delete_tier1_summary(cursor: sqlite3.Cursor, summary_id: str) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_delete_tier1_summary')
    if not cursor: func_logger.error(f"Cursor unavailable for deleting T1 {summary_id}."); return False
    try:
        cursor.execute(f"DELETE FROM {T1_SUMMARY_TABLE_NAME} WHERE id = ?", (summary_id,))
        return cursor.rowcount > 0
    except sqlite3.Error as e: func_logger.error(f"SQLite error deleting T1 {summary_id}: {e}"); return False
    except Exception as e: func_logger.error(f"Unexpected error deleting T1 {summary_id}: {e}"); return False

# --- 2.10: Async - Delete T1 Summary (Unchanged) ---
async def delete_tier1_summary(cursor: sqlite3.Cursor, summary_id: str) -> bool:
    """Async wrapper to delete a Tier 1 summary."""
    return await asyncio.to_thread(_sync_delete_tier1_summary, cursor, summary_id)

# --- 2.11: Sync - Get Max T1 End Index (Unchanged) ---
def _sync_get_max_t1_end_index(cursor: sqlite3.Cursor, session_id: str) -> Optional[int]:
    """
    Synchronously retrieves the maximum turn_end_index for a given session_id
    from the Tier 1 summaries table.
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_max_t1_end_index')
    if not cursor:
        func_logger.error(f"[{session_id}] Cursor unavailable for getting max T1 index.")
        return None
    if not session_id or not isinstance(session_id, str):
        func_logger.error("Invalid session_id provided.")
        return None

    try:
        cursor.execute(
            f"SELECT MAX(turn_end_index) FROM {T1_SUMMARY_TABLE_NAME} WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        if result and result[0] is not None:
            max_index = int(result[0])
            func_logger.debug(f"[{session_id}] Max T1 end index found in DB: {max_index}")
            return max_index
        else:
            func_logger.debug(f"[{session_id}] No T1 summaries found in DB. Max index is None.")
            return None
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error getting max T1 end index: {e}")
        return None
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error getting max T1 end index: {e}")
        return None

# --- 2.12: Async - Get Max T1 End Index (Unchanged) ---
async def get_max_t1_end_index(cursor: sqlite3.Cursor, session_id: str) -> Optional[int]:
    """
    Async wrapper to retrieve the maximum turn_end_index for a given session_id
    from the Tier 1 summaries table.
    """
    return await asyncio.to_thread(_sync_get_max_t1_end_index, cursor, session_id)

# --- 2.13: Sync - Check T1 Summary Exists (Unchanged) ---
def _sync_check_t1_summary_exists(
    cursor: sqlite3.Cursor, session_id: str, start_index: int, end_index: int
) -> bool:
    """
    Synchronously checks if a Tier 1 summary already exists for the exact
    session ID, start index, and end index.
    """
    func_logger = logging.getLogger(__name__ + '._sync_check_t1_summary_exists')
    if not cursor:
        func_logger.error(f"[{session_id}] Cursor unavailable for checking T1 existence.")
        return False # Assume doesn't exist if cursor fails
    if not session_id or not isinstance(session_id, str):
        func_logger.error("Invalid session_id provided for T1 check.")
        return False
    if not isinstance(start_index, int) or not isinstance(end_index, int):
        func_logger.error(f"[{session_id}] Invalid start/end index type for T1 check.")
        return False
    try:
        cursor.execute(
            f"""SELECT EXISTS (
                    SELECT 1 FROM {T1_SUMMARY_TABLE_NAME}
                    WHERE session_id = ? AND turn_start_index = ? AND turn_end_index = ?
                    LIMIT 1
                )""",
            (session_id, start_index, end_index)
        )
        result = cursor.fetchone()
        exists = bool(result and result[0])
        if exists:
             func_logger.debug(f"[{session_id}] Found existing T1 summary for indices {start_index}-{end_index}.")
        else:
             func_logger.debug(f"[{session_id}] No existing T1 summary found for indices {start_index}-{end_index}.")
        return exists
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error checking for T1 summary ({start_index}-{end_index}): {e}")
        return False # Assume doesn't exist on error
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error checking for T1 summary ({start_index}-{end_index}): {e}")
        return False # Assume doesn't exist on error

# --- 2.14: Async - Check T1 Summary Exists (Unchanged) ---
async def check_t1_summary_exists(
    cursor: sqlite3.Cursor, session_id: str, start_index: int, end_index: int
) -> bool:
    """
    Async wrapper to check if a T1 summary exists for the exact indices.
    """
    return await asyncio.to_thread(
        _sync_check_t1_summary_exists, cursor, session_id, start_index, end_index
    )

# --- 2.15: Sync - Get Oldest T1 Batch (NEW) ---
def _sync_get_oldest_t1_batch(
    cursor: sqlite3.Cursor, session_id: str, batch_size: int
) -> List[Dict[str, Any]]:
    """
    Synchronously retrieves the `batch_size` oldest T1 summaries for aging.
    Returns a list of dictionaries, each containing id, summary_text,
    turn_start_index, and turn_end_index. Ordered oldest first.
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_oldest_t1_batch')
    results: List[Dict[str, Any]] = []
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return results
    if batch_size <= 0: return results
    try:
        # Fetch needed columns: id, summary_text, turn_start_index, turn_end_index
        cursor.execute(
            f"""SELECT id, summary_text, turn_start_index, turn_end_index
                FROM {T1_SUMMARY_TABLE_NAME}
                WHERE session_id = ?
                ORDER BY timestamp_utc ASC
                LIMIT ?""",
            (session_id, batch_size)
        )
        rows = cursor.fetchall()
        for row in rows:
            t1_id, summary_text, start_idx, end_idx = row
            results.append({
                "id": t1_id,
                "summary_text": summary_text,
                "turn_start_index": start_idx,
                "turn_end_index": end_idx
            })
        func_logger.debug(f"[{session_id}] Fetched {len(results)} oldest T1 summaries for aging batch.")
        return results # Already ordered oldest first by SQL query
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error getting oldest T1 batch: {e}"); return []
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error getting oldest T1 batch: {e}"); return []

# --- 2.16: Async - Get Oldest T1 Batch (NEW) ---
async def get_oldest_t1_batch(
    cursor: sqlite3.Cursor, session_id: str, batch_size: int
) -> List[Dict[str, Any]]:
    """
    Async wrapper to retrieve the `batch_size` oldest T1 summaries for aging.
    Returns list of dicts: [{'id': str, 'summary_text': str, 'turn_start_index': int, 'turn_end_index': int}]
    """
    return await asyncio.to_thread(_sync_get_oldest_t1_batch, cursor, session_id, batch_size)

# --- 2.17: Sync - Delete T1 Batch (NEW) ---
def _sync_delete_t1_batch(cursor: sqlite3.Cursor, session_id: str, t1_ids: List[str]) -> bool:
    """
    Synchronously deletes multiple T1 summaries by their IDs.
    """
    func_logger = logging.getLogger(__name__ + '._sync_delete_t1_batch')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return False
    if not t1_ids: func_logger.warning(f"[{session_id}] No T1 IDs provided for deletion batch."); return True # No work needed

    try:
        # Use parameterized query to avoid SQL injection with IN clause
        placeholders = ','.join('?' for _ in t1_ids)
        sql = f"DELETE FROM {T1_SUMMARY_TABLE_NAME} WHERE session_id = ? AND id IN ({placeholders})"
        params = [session_id] + t1_ids

        cursor.execute(sql, params)
        rows_deleted = cursor.rowcount
        func_logger.info(f"[{session_id}] Deleted {rows_deleted} T1 summaries from batch (requested {len(t1_ids)}).")
        # It's successful even if rows_deleted < len(t1_ids), as some might have been deleted already.
        return True
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error deleting T1 batch: {e}")
        return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error deleting T1 batch: {e}")
        return False

# --- 2.18: Async - Delete T1 Batch (NEW) ---
async def delete_t1_batch(cursor: sqlite3.Cursor, session_id: str, t1_ids: List[str]) -> bool:
    """
    Async wrapper to delete multiple T1 summaries by ID.
    """
    return await asyncio.to_thread(_sync_delete_t1_batch, cursor, session_id, t1_ids)


# ==============================================================================
# === 3. SQLite RAG Cache Operations (Unchanged)                             ===
# ==============================================================================
# --- 3.1: Sync - Add/Update RAG Cache ---
def _sync_add_or_update_rag_cache(cursor: sqlite3.Cursor, session_id: str, context_text: str) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_add_or_update_rag_cache')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return False
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return False
    if not isinstance(context_text, str): func_logger.warning(f"[{session_id}] RAG cache context_text not a string. Storing empty."); context_text = ""
    try:
        now_utc = datetime.now(timezone.utc); timestamp_utc = now_utc.timestamp(); timestamp_iso = now_utc.isoformat()
    except Exception as e_dt:
        timestamp_utc = 0.0; timestamp_iso = "1970-01-01T00:00:00+00:00"; func_logger.error(f"Could not get timestamp for RAG cache update: {e_dt}")
    try:
        cursor.execute(f"""INSERT OR REPLACE INTO {RAG_CACHE_TABLE_NAME} (session_id, cached_context, last_updated_utc, last_updated_iso) VALUES (?, ?, ?, ?)""",
                       (session_id, context_text, timestamp_utc, timestamp_iso))
        return True
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error updating RAG cache: {e}"); return False
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error updating RAG cache: {e}"); return False

# --- 3.2: Async - Add/Update RAG Cache ---
async def add_or_update_rag_cache(cursor: sqlite3.Cursor, session_id: str, context_text: str) -> bool:
    return await asyncio.to_thread(_sync_add_or_update_rag_cache, cursor, session_id, context_text)

# --- 3.3: Sync - Get RAG Cache ---
def _sync_get_rag_cache(cursor: sqlite3.Cursor, session_id: str) -> Optional[str]:
    func_logger = logging.getLogger(__name__ + '._sync_get_rag_cache')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return None
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return None
    try:
        cursor.execute(f"""SELECT cached_context FROM {RAG_CACHE_TABLE_NAME} WHERE session_id = ?""", (session_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error retrieving RAG cache: {e}"); return None
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error retrieving RAG cache: {e}"); return None

# --- 3.4: Async - Get RAG Cache ---
async def get_rag_cache(cursor: sqlite3.Cursor, session_id: str) -> Optional[str]:
    return await asyncio.to_thread(_sync_get_rag_cache, cursor, session_id)


# ==============================================================================
# === 4. SQLite Inventory Operations (Existing - Unchanged)                  ===
# ==============================================================================
# --- 4.1: Sync - Get Character Inventory Data ---
def _sync_get_character_inventory_data(cursor: sqlite3.Cursor, session_id: str, character_name: str) -> Optional[str]:
    """
    Synchronously retrieves the inventory data (JSON string) for a specific character
    in a specific session.
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_character_inventory_data')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return None
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return None
    if not character_name or not isinstance(character_name, str): func_logger.error("Invalid character_name."); return None

    try:
        cursor.execute(
            f"SELECT inventory_data FROM {INVENTORY_TABLE_NAME} WHERE session_id = ? AND LOWER(character_name) = LOWER(?)",
            (session_id, character_name)
        )
        result = cursor.fetchone()
        if result:
            func_logger.debug(f"[{session_id}] Found inventory for character: {character_name}")
            return result[0] # Return the JSON string
        else:
            func_logger.debug(f"[{session_id}] No inventory found for character: {character_name}")
            return None
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error retrieving inventory for {character_name}: {e}")
        return None
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error retrieving inventory for {character_name}: {e}")
        return None

# --- 4.2: Async - Get Character Inventory Data ---
async def get_character_inventory_data(cursor: sqlite3.Cursor, session_id: str, character_name: str) -> Optional[str]:
    """
    Async wrapper to retrieve the inventory data (JSON string) for a specific character.
    """
    return await asyncio.to_thread(_sync_get_character_inventory_data, cursor, session_id, character_name)

# --- 4.3: Sync - Add or Update Character Inventory ---
def _sync_add_or_update_character_inventory(cursor: sqlite3.Cursor, session_id: str, character_name: str, inventory_data_json: str) -> bool:
    """
    Synchronously adds a new character inventory record or updates an existing one
    using INSERT OR REPLACE. Stores the entire inventory state as a JSON string.
    Uses the provided character_name directly (case might matter on insert/replace).
    """
    func_logger = logging.getLogger(__name__ + '._sync_add_or_update_character_inventory')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return False
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return False
    if not character_name or not isinstance(character_name, str): func_logger.error("Invalid character_name."); return False
    if not isinstance(inventory_data_json, str): func_logger.warning(f"[{session_id}] Inventory data for {character_name} not a string."); inventory_data_json = "{}"

    now_utc_timestamp = datetime.now(timezone.utc).timestamp()

    try:
        cursor.execute(
             f"""INSERT OR REPLACE INTO {INVENTORY_TABLE_NAME}
                (session_id, character_name, inventory_data, last_updated_utc)
                VALUES (?, ?, ?, ?)""",
             (session_id, character_name, inventory_data_json, now_utc_timestamp)
        )
        func_logger.info(f"[{session_id}] Successfully added/updated inventory for: {character_name}")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error adding/updating inventory for {character_name}: {e}")
        return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error adding/updating inventory for {character_name}: {e}")
        return False

# --- 4.4: Async - Add or Update Character Inventory ---
async def add_or_update_character_inventory(cursor: sqlite3.Cursor, session_id: str, character_name: str, inventory_data_json: str) -> bool:
    """
    Async wrapper to add or update a character's inventory JSON data.
    """
    return await asyncio.to_thread(_sync_add_or_update_character_inventory, cursor, session_id, character_name, inventory_data_json)

# --- 4.5: Sync - Get All Inventories for Session ---
def _sync_get_all_inventories_for_session(cursor: sqlite3.Cursor, session_id: str) -> Dict[str, str]:
    """
    Synchronously retrieves all character inventories (as JSON strings) for a given session.
    Returns a dictionary mapping character_name to inventory_data JSON string.
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_all_inventories_for_session')
    inventories: Dict[str, str] = {}
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return inventories
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return inventories

    try:
        cursor.execute(
            f"SELECT character_name, inventory_data FROM {INVENTORY_TABLE_NAME} WHERE session_id = ?",
            (session_id,)
        )
        rows = cursor.fetchall()
        for row in rows:
            char_name, inv_data = row
            if char_name and isinstance(inv_data, str):
                inventories[char_name] = inv_data
        func_logger.debug(f"[{session_id}] Retrieved inventories for {len(inventories)} characters.")
        return inventories
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error retrieving all inventories: {e}")
        return {}
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error retrieving all inventories: {e}")
        return {}

# --- 4.6: Async - Get All Inventories for Session ---
async def get_all_inventories_for_session(cursor: sqlite3.Cursor, session_id: str) -> Dict[str, str]:
    """
    Async wrapper to retrieve all character inventories for a session.
    Returns: Dict[character_name: str, inventory_data_json: str]
    """
    return await asyncio.to_thread(_sync_get_all_inventories_for_session, cursor, session_id)


# ==============================================================================
# === 5. SQLite World State Operations (Existing - Unchanged)              ===
# ==============================================================================
# --- 5.1: Sync - Get World State ---
def _sync_get_world_state(cursor: sqlite3.Cursor, session_id: str) -> Optional[Dict[str, Any]]:
    """
    Synchronously retrieves the world state (season, weather, day, time) for a specific session.
    Returns a dictionary or None if not found or error.
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_world_state')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable for world state."); return None
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id for world state."); return None

    try:
        cursor.execute(
            f"SELECT current_season, current_weather, current_day, time_of_day FROM {WORLD_STATE_TABLE_NAME} WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        if result:
            season, weather, day, time = result
            day_val = int(day) if day is not None else None
            func_logger.debug(f"[{session_id}] Found world state: Season='{season}', Weather='{weather}', Day='{day_val}', Time='{time}'")
            return {"season": season, "weather": weather, "day": day_val, "time_of_day": time}
        else:
            func_logger.debug(f"[{session_id}] No world state found in DB for this session.")
            return None
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error retrieving world state: {e}")
        return None
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error retrieving world state: {e}")
        return None

# --- 5.2: Async - Get World State ---
async def get_world_state(cursor: sqlite3.Cursor, session_id: str) -> Optional[Dict[str, Any]]:
    """
    Async wrapper to retrieve the world state (season, weather, day, time) for a specific session.
    Returns a dictionary {'season': str, 'weather': str, 'day': int, 'time_of_day': str} or None.
    """
    return await asyncio.to_thread(_sync_get_world_state, cursor, session_id)

# --- 5.3: Sync - Set World State ---
def _sync_set_world_state(
    cursor: sqlite3.Cursor,
    session_id: str,
    season: Optional[str],
    weather: Optional[str],
    day: Optional[int],
    time_of_day: Optional[str]
) -> bool:
    """
    Synchronously sets or updates the world state (season, weather, day, time) for a session
    using INSERT OR REPLACE. Accepts None for values that shouldn't be updated explicitly,
    though INSERT OR REPLACE will overwrite the whole row.
    """
    func_logger = logging.getLogger(__name__ + '._sync_set_world_state')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable for setting world state."); return False
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id for world state set."); return False

    final_season = season if isinstance(season, str) and season.strip() else "Unknown"
    final_weather = weather if isinstance(weather, str) and weather.strip() else "Unknown"
    final_day = day if isinstance(day, int) and day > 0 else 1
    final_time = time_of_day if isinstance(time_of_day, str) and time_of_day.strip() else "Unknown"

    now_utc_timestamp = datetime.now(timezone.utc).timestamp()

    try:
        cursor.execute(
             f"""INSERT OR REPLACE INTO {WORLD_STATE_TABLE_NAME}
                (session_id, current_season, current_weather, current_day, time_of_day, last_updated_utc)
                VALUES (?, ?, ?, ?, ?, ?)""",
             (session_id, final_season, final_weather, final_day, final_time, now_utc_timestamp)
        )
        func_logger.info(f"[{session_id}] Successfully set/updated world state: Season='{final_season}', Weather='{final_weather}', Day='{final_day}', Time='{final_time}'")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error setting/updating world state: {e}")
        return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error setting/updating world state: {e}")
        return False

# --- 5.4: Async - Set World State ---
async def set_world_state(
    cursor: sqlite3.Cursor,
    session_id: str,
    season: Optional[str],
    weather: Optional[str],
    day: Optional[int],
    time_of_day: Optional[str]
) -> bool:
    """
    Async wrapper to set or update the world state (season, weather, day, time) for a session.
    """
    return await asyncio.to_thread(_sync_set_world_state, cursor, session_id, season, weather, day, time_of_day)


# ==============================================================================
# === 6. SQLite Scene State Operations (Existing - Unchanged)                ===
# ==============================================================================
# --- 6.1: Sync - Get Scene State ---
def _sync_get_scene_state(cursor: sqlite3.Cursor, session_id: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Synchronously retrieves the scene state (keywords JSON, description) for a specific session.
    Returns a dictionary {'keywords_json': str|None, 'description': str|None} or None if not found or error.
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_scene_state')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable for scene state."); return None
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id for scene state."); return None

    try:
        cursor.execute(
            f"SELECT scene_keywords_json, scene_description FROM {SCENE_STATE_TABLE_NAME} WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        if result:
            keywords_json, description = result
            func_logger.debug(f"[{session_id}] Found scene state. Keywords JSON length: {len(keywords_json) if keywords_json else 0}, Desc length: {len(description) if description else 0}")
            # Return None for values if they are NULL in the DB
            return {
                "keywords_json": keywords_json if keywords_json is not None else None,
                "description": description if description is not None else None
            }
        else:
            func_logger.debug(f"[{session_id}] No scene state found in DB for this session.")
            return None # Indicate not found
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error retrieving scene state: {e}")
        return None
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error retrieving scene state: {e}")
        return None

# --- 6.2: Async - Get Scene State ---
async def get_scene_state(cursor: sqlite3.Cursor, session_id: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Async wrapper to retrieve the scene state (keywords JSON, description) for a session.
    Returns a dictionary {'keywords_json': str|None, 'description': str|None} or None.
    """
    return await asyncio.to_thread(_sync_get_scene_state, cursor, session_id)

# --- 6.3: Sync - Set Scene State ---
def _sync_set_scene_state(
    cursor: sqlite3.Cursor,
    session_id: str,
    scene_keywords_json: Optional[str], # Allow None
    scene_description: Optional[str]    # Allow None
) -> bool:
    """
    Synchronously sets or updates the scene state (keywords JSON, description) for a session
    using INSERT OR REPLACE.
    """
    func_logger = logging.getLogger(__name__ + '._sync_set_scene_state')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable for setting scene state."); return False
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id for scene state set."); return False

    # Ensure JSON is either a valid string or None (allow storing None if LLM provides no keywords)
    final_keywords_json = scene_keywords_json if isinstance(scene_keywords_json, str) else None
    # Ensure description is either a valid string or None
    final_description = scene_description if isinstance(scene_description, str) else None

    now_utc_timestamp = datetime.now(timezone.utc).timestamp()

    try:
        cursor.execute(
             f"""INSERT OR REPLACE INTO {SCENE_STATE_TABLE_NAME}
                (session_id, scene_keywords_json, scene_description, last_updated_utc)
                VALUES (?, ?, ?, ?)""",
             (session_id, final_keywords_json, final_description, now_utc_timestamp)
        )
        func_logger.info(f"[{session_id}] Successfully set/updated scene state. Keywords JSON length: {len(final_keywords_json) if final_keywords_json else 0}, Desc length: {len(final_description) if final_description else 0}")
        return True
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error setting/updating scene state: {e}")
        return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error setting/updating scene state: {e}")
        return False

# --- 6.4: Async - Set Scene State ---
async def set_scene_state(
    cursor: sqlite3.Cursor,
    session_id: str,
    scene_keywords_json: Optional[str],
    scene_description: Optional[str]
) -> bool:
    """
    Async wrapper to set or update the scene state (keywords JSON, description) for a session.
    """
    return await asyncio.to_thread(_sync_set_scene_state, cursor, session_id, scene_keywords_json, scene_description)


# ==============================================================================
# === 7. SQLite Aged Summary Operations (NEW SECTION)                        ===
# ==============================================================================

# --- 7.1: Sync - Add Aged Summary (NEW) ---
def _sync_add_aged_summary(
    cursor: sqlite3.Cursor,
    aged_summary_id: str,
    session_id: str,
    aged_summary_text: str,
    original_batch_start_index: Optional[int],
    original_batch_end_index: Optional[int],
    original_t1_count: Optional[int],
    original_t1_ids: Optional[List[str]] = None # Made optional
) -> bool:
    """
    Synchronously adds a new aged summary record to the aged_summaries table.
    """
    func_logger = logging.getLogger(__name__ + '._sync_add_aged_summary')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return False
    if not aged_summary_id or not session_id or not aged_summary_text:
        func_logger.error(f"[{session_id}] Missing required fields for adding aged summary.")
        return False

    creation_timestamp_utc = datetime.now(timezone.utc).timestamp()
    # Convert list of T1 IDs to JSON string if provided, otherwise store NULL
    original_t1_ids_json_str: Optional[str] = None
    if original_t1_ids and isinstance(original_t1_ids, list):
        try:
            original_t1_ids_json_str = json.dumps(original_t1_ids)
        except (TypeError, ValueError) as e_json:
             func_logger.warning(f"[{session_id}] Could not serialize original T1 IDs to JSON for aged summary {aged_summary_id}: {e_json}. Storing NULL.")

    try:
        cursor.execute(
            f"""INSERT INTO {AGED_SUMMARY_TABLE_NAME} (
                    id, session_id, aged_summary_text, creation_timestamp_utc,
                    original_batch_start_index, original_batch_end_index,
                    original_t1_count, original_t1_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                aged_summary_id, session_id, aged_summary_text, creation_timestamp_utc,
                original_batch_start_index, original_batch_end_index,
                original_t1_count, original_t1_ids_json_str
            )
        )
        func_logger.info(f"[{session_id}] Added aged summary {aged_summary_id} (condensing {original_t1_count} T1s, turns {original_batch_start_index}-{original_batch_end_index}).")
        return True
    except sqlite3.IntegrityError as e:
        # This likely means the aged_summary_id was reused, which shouldn't happen with UUIDs
        func_logger.error(f"[{session_id}] IntegrityError adding aged summary {aged_summary_id} (duplicate ID?): {e}")
        return False
    except sqlite3.Error as e:
        func_logger.error(f"[{session_id}] SQLite error adding aged summary {aged_summary_id}: {e}")
        return False
    except Exception as e:
        func_logger.error(f"[{session_id}] Unexpected error adding aged summary {aged_summary_id}: {e}")
        return False

# --- 7.2: Async - Add Aged Summary (NEW) ---
async def add_aged_summary(
    cursor: sqlite3.Cursor,
    aged_summary_id: str,
    session_id: str,
    aged_summary_text: str,
    original_batch_start_index: Optional[int],
    original_batch_end_index: Optional[int],
    original_t1_count: Optional[int],
    original_t1_ids: Optional[List[str]] = None
) -> bool:
    """
    Async wrapper to add a new aged summary record.
    """
    return await asyncio.to_thread(
        _sync_add_aged_summary, cursor, aged_summary_id, session_id, aged_summary_text,
        original_batch_start_index, original_batch_end_index, original_t1_count, original_t1_ids
    )

# --- 7.3: Sync - Get Recent Aged Summaries (NEW) ---
def _sync_get_recent_aged_summaries(
    cursor: sqlite3.Cursor, session_id: str, limit: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Synchronously retrieves the `limit` most recent aged summaries based on
    creation time, returning text and metadata for sorting.
    Returns list of (aged_summary_text, metadata_dict) tuples, ordered most recent first.
    """
    func_logger = logging.getLogger(__name__ + '._sync_get_recent_aged_summaries')
    results: List[Tuple[str, Dict[str, Any]]] = []
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return results
    if limit <= 0: return results
    try:
        # Fetch needed columns: text, timestamp, and batch end index for potential sorting
        cursor.execute(
            f"""SELECT aged_summary_text, creation_timestamp_utc, original_batch_end_index
                FROM {AGED_SUMMARY_TABLE_NAME}
                WHERE session_id = ?
                ORDER BY creation_timestamp_utc DESC
                LIMIT ?""",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        for row in rows:
            aged_text, created_ts, batch_end_idx = row
            metadata = {
                "creation_timestamp_utc": created_ts,
                "original_batch_end_index": batch_end_idx
                # Add other metadata fields if needed for context display/logic later
            }
            results.append((aged_text, metadata))

        func_logger.debug(f"[{session_id}] Retrieved {len(results)} recent aged summaries.")
        # Results are already ordered most recent first by SQL query
        return results
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error getting recent aged summaries: {e}"); return []
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error getting recent aged summaries: {e}"); return []

# --- 7.4: Async - Get Recent Aged Summaries (NEW) ---
async def get_recent_aged_summaries(
    cursor: sqlite3.Cursor, session_id: str, limit: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Async wrapper to retrieve recent aged summaries with metadata.
    Returns list of (aged_summary_text, metadata_dict) tuples, ordered most recent first.
    """
    return await asyncio.to_thread(_sync_get_recent_aged_summaries, cursor, session_id, limit)


# ==============================================================================
# === 8. ChromaDB Tier 2 Operations (Unchanged - Renumbered Section)         ===
# ==============================================================================
# (Functions remain unchanged, only section number updated)
# --- 8.1: Sync - Get/Create Chroma Collection ---
def _sync_get_or_create_chroma_collection(
    chroma_client: Any, # Expects chromadb.ClientAPI but use Any for optional import
    collection_name: str,
    embedding_function: Optional[ChromaEmbeddingFunction] = None,
    metadata_config: Optional[Dict] = None,
) -> Optional[Any]: # Returns ChromaCollectionType or None
    func_logger = logging.getLogger(__name__ + '._sync_get_or_create_chroma_collection')
    if not CHROMADB_AVAILABLE: func_logger.error("ChromaDB library not available."); return None
    if not chroma_client: func_logger.error("ChromaDB client instance not provided."); return None
    if not collection_name: func_logger.error("ChromaDB collection name is required."); return None
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$", collection_name) or ".." in collection_name:
        func_logger.error(f"Invalid ChromaDB collection name format: '{collection_name}'.")
        return None
    if embedding_function is None:
        func_logger.warning(f"No embedding function provided for ChromaDB collection '{collection_name}'. Retrieval might fail.")
    try:
        func_logger.debug(f"Accessing ChromaDB collection '{collection_name}'...")
        collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, metadata=metadata_config,)
        if collection: func_logger.info(f"Accessed/created ChromaDB collection '{collection_name}'."); return collection
        else: func_logger.error(f"ChromaDB get_or_create_collection returned None/False for '{collection_name}'."); return None
    except sqlite3.OperationalError as e: func_logger.error(f"SQLite operational error accessing ChromaDB collection '{collection_name}': {e}.", exc_info=True); return None
    except InvalidDimensionException as ide: func_logger.error(f"ChromaDB Dimension Exception for collection '{collection_name}': {ide}. Check embedding function consistency.", exc_info=True); return None
    except Exception as e: func_logger.error(f"Unexpected error accessing/creating ChromaDB collection '{collection_name}': {e}", exc_info=True); return None

# --- 8.2: Async - Get/Create Chroma Collection ---
async def get_or_create_chroma_collection(chroma_client: Any, collection_name: str, embedding_function: Optional[ChromaEmbeddingFunction] = None, metadata_config: Optional[Dict] = None,) -> Optional[Any]:
    return await asyncio.to_thread(_sync_get_or_create_chroma_collection, chroma_client, collection_name, embedding_function, metadata_config)

# --- 8.3: Sync - Add to Chroma Collection ---
def _sync_add_to_chroma_collection(collection: Any, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_add_to_chroma_collection')
    if not collection: func_logger.error("ChromaDB collection object not provided."); return False
    if not hasattr(collection, 'add'): func_logger.error("Provided collection object lacks 'add' method."); return False
    try:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        func_logger.info(f"Successfully added {len(ids)} item(s) to ChromaDB collection '{getattr(collection, 'name', 'unknown')}'.")
        return True
    except InvalidDimensionException as ide: func_logger.error(f"ChromaDB Dimension Error adding to collection '{getattr(collection, 'name', 'unknown')}': {ide}.", exc_info=True); return False
    except Exception as e: func_logger.error(f"Error adding to ChromaDB collection '{getattr(collection, 'name', 'unknown')}': {e}", exc_info=True); return False

# --- 8.4: Async - Add to Chroma Collection ---
async def add_to_chroma_collection(collection: Any, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]) -> bool:
    return await asyncio.to_thread(_sync_add_to_chroma_collection, collection, ids, embeddings, metadatas, documents)

# --- 8.5: Sync - Query Chroma Collection ---
def _sync_query_chroma_collection(collection: Any, query_embeddings: List[List[float]], n_results: int, include: List[str] = ["documents", "distances", "metadatas"]) -> Optional[Dict[str, Any]]:
    func_logger = logging.getLogger(__name__ + '._sync_query_chroma_collection')
    if not collection: func_logger.error("ChromaDB collection object not provided."); return None
    if not hasattr(collection, 'query'): func_logger.error("Provided collection object lacks 'query' method."); return None
    if n_results <= 0: func_logger.warning("n_results <= 0 for ChromaDB query, returning empty."); return {'ids': [], 'embeddings': [], 'documents': [], 'metadatas': [], 'distances': []} # Return empty structure
    try:
        results = collection.query(query_embeddings=query_embeddings, n_results=n_results, include=include)
        func_logger.debug(f"ChromaDB query successful for collection '{getattr(collection, 'name', 'unknown')}'.")
        return results
    except InvalidDimensionException as ide: func_logger.error(f"ChromaDB Dimension Error querying collection '{getattr(collection, 'name', 'unknown')}': {ide}.", exc_info=True); return None
    except Exception as e: func_logger.error(f"Error querying ChromaDB collection '{getattr(collection, 'name', 'unknown')}': {e}", exc_info=True); return None

# --- 8.6: Async - Query Chroma Collection ---
async def query_chroma_collection(collection: Any, query_embeddings: List[List[float]], n_results: int, include: List[str] = ["documents", "distances", "metadatas"]) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(_sync_query_chroma_collection, collection, query_embeddings, n_results, include)

# --- 8.7: Sync - Get Chroma Collection Count ---
def _sync_get_chroma_collection_count(collection: Any) -> int:
    func_logger = logging.getLogger(__name__ + '._sync_get_chroma_collection_count')
    if not collection: func_logger.error("ChromaDB collection object not provided."); return -1
    if not hasattr(collection, 'count'): func_logger.error("Provided collection object lacks 'count' method."); return -1
    try: count = collection.count(); return count
    except Exception as e: func_logger.error(f"Error getting count for ChromaDB collection '{getattr(collection, 'name', 'unknown')}': {e}", exc_info=True); return -1

# --- 8.8: Async - Get Chroma Collection Count ---
async def get_chroma_collection_count(collection: Any) -> int:
    return await asyncio.to_thread(_sync_get_chroma_collection_count, collection)

# === END MODIFIED FILE: i4_llm_agent/database.py ===