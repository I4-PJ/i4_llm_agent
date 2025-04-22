# [[START MODIFIED database.py]]
# i4_llm_agent/database.py

import logging
import sqlite3
import asyncio
import re
import os
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

# ==============================================================================
# === SQLite Initialization                                                  ===
# ==============================================================================

def _sync_initialize_sqlite_tables(cursor: sqlite3.Cursor) -> bool:
    """Initializes all necessary SQLite tables."""
    func_logger = logging.getLogger(__name__ + '._sync_initialize_sqlite_tables')
    if not cursor:
        func_logger.error("SQLite cursor is not available.")
        return False
    try:
        # Initialize T1 Summary Table
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
        # Index for the existence check
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_t1_session_start_end ON {T1_SUMMARY_TABLE_NAME} (session_id, turn_start_index, turn_end_index)"
        )
        func_logger.debug(f"Table '{T1_SUMMARY_TABLE_NAME}' initialized successfully.")

        # Initialize RAG Cache Table
        cursor.execute(f"""CREATE TABLE IF NOT EXISTS {RAG_CACHE_TABLE_NAME} (
                session_id TEXT PRIMARY KEY, cached_context TEXT NOT NULL,
                last_updated_utc REAL NOT NULL, last_updated_iso TEXT
            )""")
        func_logger.debug(f"Table '{RAG_CACHE_TABLE_NAME}' initialized successfully.")

        return True
    except sqlite3.Error as e:
        func_logger.error(f"SQLite error initializing tables: {e}")
        return False
    except Exception as e:
        func_logger.error(f"Unexpected error initializing tables: {e}")
        return False

async def initialize_sqlite_tables(cursor: sqlite3.Cursor) -> bool:
    """Async wrapper to initialize all necessary SQLite tables."""
    return await asyncio.to_thread(_sync_initialize_sqlite_tables, cursor)

# ==============================================================================
# === SQLite Tier 1 Summary Operations                                       ===
# ==============================================================================

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

async def add_tier1_summary(
    cursor: sqlite3.Cursor,
    summary_id: str, session_id: str, user_id: str, summary_text: str, metadata: Dict
) -> bool:
    """Async wrapper to add a Tier 1 summary."""
    return await asyncio.to_thread(_sync_add_tier1_summary, cursor, summary_id, session_id, user_id, summary_text, metadata)

def _sync_get_recent_tier1_summaries(cursor: sqlite3.Cursor, session_id: str, limit: int) -> List[str]:
    func_logger = logging.getLogger(__name__ + '._sync_get_recent_tier1_summaries')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return []
    if limit <= 0: return []
    try:
        cursor.execute(
            f"SELECT summary_text FROM {T1_SUMMARY_TABLE_NAME} WHERE session_id = ? ORDER BY timestamp_utc DESC LIMIT ?",
            (session_id, limit)
        )
        summaries = [row[0] for row in cursor.fetchall()]
        summaries.reverse() # Return in chronological order
        return summaries
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error getting recent T1: {e}"); return []
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error getting recent T1: {e}"); return []

async def get_recent_tier1_summaries(cursor: sqlite3.Cursor, session_id: str, limit: int) -> List[str]:
    """Async wrapper to get recent Tier 1 summaries."""
    return await asyncio.to_thread(_sync_get_recent_tier1_summaries, cursor, session_id, limit)

def _sync_get_tier1_summary_count(cursor: sqlite3.Cursor, session_id: str) -> int:
    func_logger = logging.getLogger(__name__ + '._sync_get_tier1_summary_count')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return -1
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {T1_SUMMARY_TABLE_NAME} WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        return result[0] if result else 0
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error counting T1: {e}"); return -1
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error counting T1: {e}"); return -1

async def get_tier1_summary_count(cursor: sqlite3.Cursor, session_id: str) -> int:
    """Async wrapper to count Tier 1 summaries."""
    return await asyncio.to_thread(_sync_get_tier1_summary_count, cursor, session_id)

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

async def get_oldest_tier1_summary(cursor: sqlite3.Cursor, session_id: str) -> Optional[Tuple[str, str, Dict]]:
    """Async wrapper to get the oldest Tier 1 summary."""
    return await asyncio.to_thread(_sync_get_oldest_tier1_summary, cursor, session_id)

def _sync_delete_tier1_summary(cursor: sqlite3.Cursor, summary_id: str) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_delete_tier1_summary')
    if not cursor: func_logger.error(f"Cursor unavailable for deleting T1 {summary_id}."); return False
    try:
        cursor.execute(f"DELETE FROM {T1_SUMMARY_TABLE_NAME} WHERE id = ?", (summary_id,))
        return cursor.rowcount > 0
    except sqlite3.Error as e: func_logger.error(f"SQLite error deleting T1 {summary_id}: {e}"); return False
    except Exception as e: func_logger.error(f"Unexpected error deleting T1 {summary_id}: {e}"); return False

async def delete_tier1_summary(cursor: sqlite3.Cursor, summary_id: str) -> bool:
    """Async wrapper to delete a Tier 1 summary."""
    return await asyncio.to_thread(_sync_delete_tier1_summary, cursor, summary_id)


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

async def get_max_t1_end_index(cursor: sqlite3.Cursor, session_id: str) -> Optional[int]:
    """
    Async wrapper to retrieve the maximum turn_end_index for a given session_id
    from the Tier 1 summaries table.
    """
    return await asyncio.to_thread(_sync_get_max_t1_end_index, cursor, session_id)


# --- [[[ NEW FUNCTION ]]] ---
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
        # Use EXISTS for efficiency, we only care if at least one row matches
        cursor.execute(
            f"""SELECT EXISTS (
                    SELECT 1 FROM {T1_SUMMARY_TABLE_NAME}
                    WHERE session_id = ? AND turn_start_index = ? AND turn_end_index = ?
                    LIMIT 1
                )""",
            (session_id, start_index, end_index)
        )
        result = cursor.fetchone()
        # fetchone() returns (1,) if exists, (0,) if not
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

async def check_t1_summary_exists(
    cursor: sqlite3.Cursor, session_id: str, start_index: int, end_index: int
) -> bool:
    """
    Async wrapper to check if a T1 summary exists for the exact indices.
    """
    return await asyncio.to_thread(
        _sync_check_t1_summary_exists, cursor, session_id, start_index, end_index
    )
# --- [[[ END NEW FUNCTION ]]] ---


# ==============================================================================
# === SQLite RAG Cache Operations                                            ===
# ==============================================================================
# ... (RAG cache functions remain unchanged) ...
def _sync_add_or_update_rag_cache(cursor: sqlite3.Cursor, session_id: str, context_text: str) -> bool:
    func_logger = logging.getLogger(__name__ + '._sync_add_or_update_rag_cache')
    if not cursor: func_logger.error(f"[{session_id}] Cursor unavailable."); return False
    if not session_id or not isinstance(session_id, str): func_logger.error("Invalid session_id."); return False
    if not isinstance(context_text, str): func_logger.warning(f"[{session_id}] RAG cache context_text not a string. Storing empty."); context_text = ""
    try:
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc); timestamp_utc = now_utc.timestamp(); timestamp_iso = now_utc.isoformat()
    except ImportError:
        timestamp_utc = 0.0; timestamp_iso = "1970-01-01T00:00:00+00:00"; func_logger.error("Could not import datetime for RAG cache timestamp.")
    try:
        cursor.execute(f"""INSERT OR REPLACE INTO {RAG_CACHE_TABLE_NAME} (session_id, cached_context, last_updated_utc, last_updated_iso) VALUES (?, ?, ?, ?)""",
                       (session_id, context_text, timestamp_utc, timestamp_iso))
        return True
    except sqlite3.Error as e: func_logger.error(f"[{session_id}] SQLite error updating RAG cache: {e}"); return False
    except Exception as e: func_logger.error(f"[{session_id}] Unexpected error updating RAG cache: {e}"); return False
async def add_or_update_rag_cache(cursor: sqlite3.Cursor, session_id: str, context_text: str) -> bool:
    return await asyncio.to_thread(_sync_add_or_update_rag_cache, cursor, session_id, context_text)
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
async def get_rag_cache(cursor: sqlite3.Cursor, session_id: str) -> Optional[str]:
    return await asyncio.to_thread(_sync_get_rag_cache, cursor, session_id)

# ==============================================================================
# === ChromaDB Tier 2 Operations                                             ===
# ==============================================================================
# ... (ChromaDB functions remain unchanged) ...
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
async def get_or_create_chroma_collection(chroma_client: Any, collection_name: str, embedding_function: Optional[ChromaEmbeddingFunction] = None, metadata_config: Optional[Dict] = None,) -> Optional[Any]:
    return await asyncio.to_thread(_sync_get_or_create_chroma_collection, chroma_client, collection_name, embedding_function, metadata_config)
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
async def add_to_chroma_collection(collection: Any, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]) -> bool:
    return await asyncio.to_thread(_sync_add_to_chroma_collection, collection, ids, embeddings, metadatas, documents)
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
async def query_chroma_collection(collection: Any, query_embeddings: List[List[float]], n_results: int, include: List[str] = ["documents", "distances", "metadatas"]) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(_sync_query_chroma_collection, collection, query_embeddings, n_results, include)
def _sync_get_chroma_collection_count(collection: Any) -> int:
    func_logger = logging.getLogger(__name__ + '._sync_get_chroma_collection_count')
    if not collection: func_logger.error("ChromaDB collection object not provided."); return -1
    if not hasattr(collection, 'count'): func_logger.error("Provided collection object lacks 'count' method."); return -1
    try: count = collection.count(); return count
    except Exception as e: func_logger.error(f"Error getting count for ChromaDB collection '{getattr(collection, 'name', 'unknown')}': {e}", exc_info=True); return -1
async def get_chroma_collection_count(collection: Any) -> int:
    return await asyncio.to_thread(_sync_get_chroma_collection_count, collection)
# [[END MODIFIED database.py]]