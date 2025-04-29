# === START OF FILE script.txt ===

# === SECTION 1: METADATA HEADER ===
# --- REQUIRED METADATA HEADER ---
"""
title: SESSION_MEMORY PIPE (v0.19.1)
author: Petr Jilek & Assistant Gemini
version: 0.19.1
description: Adds Scene Generation feature (using Event Hint LLM endpoint). Implements enable_scene_generation valve.
requirements: pydantic, chromadb, i4_llm_agent>=0.1.7, tiktoken, httpx, open_webui (internal utils)
"""
# --- END HEADER ---

# === SECTION 2: IMPORTS ===
# /////////////////////////////////////////
# /// 2.1: Standard Library Imports     ///
# /////////////////////////////////////////
import logging
import os
import re
import inspect
import asyncio
import sqlite3
import json
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import (
    Tuple,
    Union,
    List,
    Dict,
    Optional,
    Generator,
    Iterator,
    Any,
    Callable,
    Sequence,
    Coroutine,
    AsyncGenerator,
)

# /////////////////////////////////////////
# /// 2.1.1: HTTPX Import             ///
# /////////////////////////////////////////
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False
    logging.getLogger("SessionMemoryPipe_startup").warning(
        "httpx not installed. Final LLM streaming will fail if enabled."
    )

# /////////////////////////////////////////
# /// 2.2: Core & OWI Imports           ///
# /////////////////////////////////////////


# --- Fallback BaseModel/Field ---
class BaseModel:
    def __init__(self, **kwargs):
        [setattr(self, key, value) for key, value in kwargs.items()]

    def model_post_init(self, __context: Any) -> None:
        pass

    def model_dump(self) -> Dict:
        try:
            return {
                k: v
                for k, v in self.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }
        except Exception:
            return {}


def Field(*args, **kwargs):
    return kwargs.get("default")


try:
    from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField

    BaseModel = PydanticBaseModel
    Field = PydanticField
    logging.getLogger("SessionMemoryPipe_startup").info(
        "Using pydantic BaseModel/Field."
    )
except ImportError:
    logging.getLogger("SessionMemoryPipe_startup").warning(
        "Using fallback BaseModel/Field."
    )
except Exception as e:
    logging.getLogger("SessionMemoryPipe_startup").error(
        f"Pydantic import error (using fallback): {e}", exc_info=True
    )

from fastapi import Request

# /////////////////////////////////////////
# /// 2.3: ChromaDB Import              ///
# /////////////////////////////////////////
try:
    import chromadb

    CHROMADB_AVAILABLE = True
    CHROMADB_IMPORT_ERROR = None
except ImportError as e:
    CHROMADB_AVAILABLE = False
    CHROMADB_IMPORT_ERROR = str(e)
    logging.getLogger("SessionMemoryPipe_startup").error(
        f"CRITICAL: Failed import 'chromadb': {e}. T2 disabled."
    )

# /////////////////////////////////////////
# /// 2.4: OWI RAG Utils Import         ///
# /////////////////////////////////////////
try:
    from open_webui.retrieval.utils import get_embedding_function
    from open_webui.config import (
        RAG_EMBEDDING_CONTENT_PREFIX,
        RAG_EMBEDDING_QUERY_PREFIX,
    )

    OWI_RAG_UTILS_AVAILABLE = True
    OWI_IMPORT_ERROR = None
    OwiEmbeddingFunction = Callable
except ImportError as e:
    OWI_RAG_UTILS_AVAILABLE = False
    OWI_IMPORT_ERROR = str(e)
    get_embedding_function = None
    RAG_EMBEDDING_CONTENT_PREFIX = "passage: "
    RAG_EMBEDDING_QUERY_PREFIX = "query: "
    OwiEmbeddingFunction = Callable
    logging.getLogger("SessionMemoryPipe_startup").warning(
        f"Could not import OWI RAG utils: {e}."
    )

# /////////////////////////////////////////
# /// 2.5: i4_llm_agent Library Import  ///
# /////////////////////////////////////////
# Requires i4_llm_agent version >= 0.1.7
try:
    from i4_llm_agent import (
        SessionManager,
        SessionPipeOrchestrator,
        initialize_sqlite_tables,
        CHROMADB_AVAILABLE as I4_AGENT_CHROMADB_FLAG,
        DIALOGUE_ROLES as I4_AGENT_DIALOGUE_ROLES,
        # --- START IMPORT: Import specific default templates ---
        # NOTE: These are now the *only* source for these prompts
        DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
        DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
        DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT,
        DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
        DEFAULT_EVENT_HINT_TEMPLATE_TEXT,
        DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT,
        DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT,  # <<< ADDED
        # --- END IMPORT ---
    )

    I4_LLM_AGENT_AVAILABLE = True
    IMPORT_ERROR_DETAILS = None
    if not I4_AGENT_CHROMADB_FLAG and CHROMADB_AVAILABLE:
        logging.getLogger("SessionMemoryPipe_startup").warning(
            "ChromaDB available locally, but i4_llm_agent reports it missing."
        )
    elif I4_AGENT_CHROMADB_FLAG and not CHROMADB_AVAILABLE:
        logging.getLogger("SessionMemoryPipe_startup").warning(
            "i4_llm_agent reports ChromaDB available, but local import failed."
        )
except ImportError as e:
    I4_LLM_AGENT_AVAILABLE = False
    IMPORT_ERROR_DETAILS = str(e)
    logging.getLogger("SessionMemoryPipe_startup").critical(
        f"CRITICAL: Failed import 'i4_llm_agent' (v0.1.7+ required): {e}."
    )

    class SessionManager:
        pass

    class SessionPipeOrchestrator:
        pass

    async def initialize_sqlite_tables(*args, **kwargs):
        return False

    I4_AGENT_DIALOGUE_ROLES = ["user", "assistant"]
    # Dummy defaults if library not found
    DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT = "[Default Cache Prompt Load Failed]"
    DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT = (
        "[Default Select Prompt Load Failed]"
    )
    DEFAULT_INVENTORY_UPDATE_TEMPLATE_TEXT = "[Default Inventory Prompt Load Failed]"
    DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE = "[Default Stateless Prompt Load Failed]"
    DEFAULT_EVENT_HINT_TEMPLATE_TEXT = "[Default Event Hint Prompt Load Failed]"
    DEFAULT_WORLD_STATE_PARSE_TEMPLATE_TEXT = (
        "[Default World State Parse Prompt Load Failed]"
    )
    DEFAULT_SCENE_ASSESSMENT_TEMPLATE_TEXT = (
        "[Default Scene Assessment Prompt Load Failed]"  # <<< ADDED
    )


# /////////////////////////////////////////
# /// 2.6: Tiktoken Import              ///
# /////////////////////////////////////////
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    logging.getLogger("SessionMemoryPipe_startup").critical(
        "CRITICAL: tiktoken not installed. Token counting WILL FAIL."
    )


# === SECTION 3: CONSTANTS & DEFAULTS ===
DEFAULT_LOG_DIR = r"C:\\Utils\\OpenWebUI"
DEFAULT_LOG_FILE_NAME = "session_memory_v19_1_pipe_log.log"  # Version updated
DEFAULT_LOG_LEVEL = "DEBUG"
ENV_VAR_LOG_FILE_PATH = "SM_LOG_FILE_PATH"
ENV_VAR_LOG_LEVEL = "SM_LOG_LEVEL"

# --- Summarizer Config (Prompt defaults to constant below) ---
ENV_VAR_SUMMARIZER_API_URL = "SM_SUMMARIZER_API_URL"
ENV_VAR_SUMMARIZER_API_KEY = "SM_SUMMARIZER_API_KEY"
ENV_VAR_SUMMARIZER_TEMPERATURE = "SM_SUMMARIZER_TEMPERATURE"
ENV_VAR_SUMMARIZER_SYSTEM_PROMPT = "SM_SUMMARIZER_SYSTEM_PROMPT"  # Kept for valve
DEFAULT_SUMMARIZER_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
DEFAULT_SUMMARIZER_API_KEY = ""
DEFAULT_SUMMARIZER_TEMPERATURE = 0.5

SUMMARIZER_DIALOGUE_CHUNK_PLACEHOLDER = "{dialogue_chunk}"

# --- <<< MODIFIED: ADAPTED DETAILED ROLEPLAY PROMPT FOR T1 SUMMARIZER >>> ---
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

**Scene Location and Context (from Chunk):**
(description based *only* on dialogue chunk)

**Emotional State Changes (per character, from Chunk):**
- (Character Name): emotional shifts *expressed in chunk*.

**Relationship Developments (from Chunk):**
- (short descriptions *from chunk*)

**Practical Developments (from Chunk):**
- (details about survival, fatigue, injuries, supplies *mentioned in chunk*)

**World-State Changes (from Chunk):**
- (plot changes, movement of threats, discoveries *stated in chunk*)

**Critical Dialogue Fragments (from Chunk):**
- (List 1–3 key quotes *from this chunk* that define emotional turning points)

**Important Continuity Anchors (from Chunk):**
- (Facts, feelings, or decisions *from this chunk* that must persist.)

---

[[NOTES]]
- Focus **exclusively** on the provided DIALOGUE CHUNK.
- Base the summary *only* on the text within the chunk.
- Prioritize emotional realism and narrative continuity over brevity based on the chunk's content.

**Memory Summary Output (Based SOLELY on the Dialogue Chunk):**
"""
# --- <<< END ADAPTED SUMMARIZER PROMPT >>> ---

# --- Core Memory Config ---
DEFAULT_TOKENIZER_ENCODING = "cl100k_base"
ENV_VAR_TOKENIZER_ENCODING = "SM_TOKENIZER_ENCODING"
DEFAULT_T0_ACTIVE_HISTORY_TOKEN_LIMIT = 4000
ENV_VAR_T0_ACTIVE_HISTORY_TOKEN_LIMIT = "SM_T0_ACTIVE_HISTORY_TOKEN_LIMIT"
DEFAULT_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET = 2000
ENV_VAR_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET = "SM_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET"
DEFAULT_MAX_STORED_SUMMARY_BLOCKS = 10
ENV_VAR_MAX_STORED_SUMMARY_BLOCKS = "SM_MAX_STORED_SUMMARY_BLOCKS"

# --- Paths ---
DEFAULT_CHROMADB_PATH = os.path.join(DEFAULT_LOG_DIR, "session_summary_t2_db")
ENV_VAR_CHROMADB_PATH = "SM_CHROMADB_PATH"
DEFAULT_SQLITE_DB_PATH = os.path.join(
    DEFAULT_LOG_DIR, "session_memory_tier1_cache_inventory.db"
)
ENV_VAR_SQLITE_DB_PATH = "SM_SQLITE_DB_PATH"

# --- RAG / T2 Config ---
DEFAULT_SUMMARY_COLLECTION_PREFIX = "sm_t2_"
ENV_VAR_SUMMARY_COLLECTION_PREFIX = "SM_SUMMARY_COLLECTION_PREFIX"
ENV_VAR_RAGQ_LLM_API_URL = "SM_RAGQ_LLM_API_URL"
ENV_VAR_RAGQ_LLM_API_KEY = "SM_RAGQ_LLM_API_KEY"
ENV_VAR_RAGQ_LLM_TEMPERATURE = "SM_RAGQ_LLM_TEMPERATURE"
ENV_VAR_RAGQ_LLM_PROMPT = "SM_RAGQ_LLM_PROMPT"  # Kept for valve
DEFAULT_RAGQ_LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
DEFAULT_RAGQ_LLM_API_KEY = ""
DEFAULT_RAGQ_LLM_TEMPERATURE = 0.3
DEFAULT_RAGQ_LLM_PROMPT = """Based on the latest user message and recent dialogue context, generate a concise search query focusing on the key entities, topics, or questions raised.

Latest Message: {latest_message}

Dialogue Context:
{dialogue_context}

Search Query:"""  # Script default
DEFAULT_RAG_SUMMARY_RESULTS_COUNT = 3
ENV_VAR_RAG_SUMMARY_RESULTS_COUNT = "SM_RAG_SUMMARY_RESULTS_COUNT"

# --- Refinement / RAG Cache Features (Prompts use library defaults) ---
ENV_VAR_ENABLE_RAG_CACHE = "SM_ENABLE_RAG_CACHE"
ENV_VAR_ENABLE_STATELESS_REFINEMENT = "SM_ENABLE_REFINEMENT"
ENV_VAR_REFINER_API_URL = "SM_REFINER_API_URL"
ENV_VAR_REFINER_API_KEY = "SM_REFINER_API_KEY"
ENV_VAR_REFINER_TEMPERATURE = "SM_REFINER_TEMPERATURE"
ENV_VAR_REFINER_HISTORY_COUNT = "SM_REFINER_HISTORY_COUNT"
ENV_VAR_STATELESS_REFINER_SKIP_THRESHOLD = "SM_REFINER_SKIP_THRESHOLD"
ENV_VAR_CACHE_UPDATE_SKIP_OWI_THRESHOLD = "SM_CACHE_UPDATE_SKIP_OWI_THRESHOLD"
ENV_VAR_CACHE_UPDATE_SIMILARITY_THRESHOLD = "SM_CACHE_UPDATE_SIMILARITY_THRESHOLD"
DEFAULT_ENABLE_RAG_CACHE = False
DEFAULT_ENABLE_STATELESS_REFINEMENT = False
DEFAULT_REFINER_API_URL = DEFAULT_RAGQ_LLM_API_URL
DEFAULT_REFINER_API_KEY = ""
DEFAULT_REFINER_TEMPERATURE = 0.3
DEFAULT_REFINER_HISTORY_COUNT = 6
DEFAULT_STATELESS_REFINER_SKIP_THRESHOLD = 500
DEFAULT_CACHE_UPDATE_SKIP_OWI_THRESHOLD = 50
DEFAULT_CACHE_UPDATE_SIMILARITY_THRESHOLD = 0.9

# --- Final LLM Pass-Through Config ---
ENV_VAR_FINAL_LLM_API_URL = "SM_FINAL_LLM_API_URL"
ENV_VAR_FINAL_LLM_API_KEY = "SM_FINAL_LLM_API_KEY"
ENV_VAR_FINAL_LLM_TEMPERATURE = "SM_FINAL_LLM_TEMPERATURE"
ENV_VAR_FINAL_LLM_TIMEOUT = "SM_FINAL_LLM_TIMEOUT"
DEFAULT_FINAL_LLM_API_URL = ""
DEFAULT_FINAL_LLM_API_KEY = ""
DEFAULT_FINAL_LLM_TEMPERATURE = 0.7
DEFAULT_FINAL_LLM_TIMEOUT = 120

# --- Inventory Management Feature (Prompt uses library default) ---
DEFAULT_ENABLE_INVENTORY_MANAGEMENT = False
ENV_VAR_ENABLE_INVENTORY_MANAGEMENT = "SM_ENABLE_INVENTORY"
ENV_VAR_INV_LLM_API_URL = "SM_INV_LLM_API_URL"
ENV_VAR_INV_LLM_API_KEY = "SM_INV_LLM_API_KEY"
ENV_VAR_INV_LLM_TEMPERATURE = "SM_INV_LLM_TEMPERATURE"
DEFAULT_INV_LLM_API_URL = DEFAULT_REFINER_API_URL
DEFAULT_INV_LLM_API_KEY = ""
DEFAULT_INV_LLM_TEMPERATURE = DEFAULT_REFINER_TEMPERATURE

# --- Event Hint Feature (Prompt uses library default) ---
DEFAULT_ENABLE_EVENT_HINTS = False
ENV_VAR_ENABLE_EVENT_HINTS = "SM_ENABLE_EVENT_HINTS"
ENV_VAR_EVENT_HINT_LLM_API_URL = "SM_EVENT_HINT_LLM_API_URL"
ENV_VAR_EVENT_HINT_LLM_API_KEY = "SM_EVENT_HINT_LLM_API_KEY"
ENV_VAR_EVENT_HINT_LLM_TEMPERATURE = "SM_EVENT_HINT_LLM_TEMPERATURE"
ENV_VAR_EVENT_HINT_HISTORY_COUNT = "SM_EVENT_HINT_HISTORY_COUNT"
DEFAULT_EVENT_HINT_LLM_API_URL = DEFAULT_RAGQ_LLM_API_URL
DEFAULT_EVENT_HINT_LLM_API_KEY = ""
DEFAULT_EVENT_HINT_LLM_TEMPERATURE = 0.7
DEFAULT_EVENT_HINT_HISTORY_COUNT = 6

# --- Scene Generation Feature (Uses Event Hint LLM endpoint) --- # <<< NEW SECTION
DEFAULT_ENABLE_SCENE_GENERATION = False  # <<< NEW CONSTANT
ENV_VAR_ENABLE_SCENE_GENERATION = "SM_ENABLE_SCENE_GENERATION"  # <<< NEW ENV VAR

# --- General & Debug ---
DEFAULT_INCLUDE_ACK_TURNS = True
ENV_VAR_INCLUDE_ACK_TURNS = "SM_INCLUDE_ACK_TURNS"
DEFAULT_EMIT_STATUS_UPDATES = True
ENV_VAR_EMIT_STATUS_UPDATES = "SM_EMIT_STATUS_UPDATES"
DEFAULT_DEBUG_LOG_FINAL_PAYLOAD = False
ENV_VAR_DEBUG_LOG_FINAL_PAYLOAD = "SM_DEBUG_LOG_PAYLOAD"
DEFAULT_DEBUG_LOG_RAW_INPUT = False
ENV_VAR_DEBUG_LOG_RAW_INPUT = "SM_DEBUG_LOG_RAW_INPUT"

# --- Logger Setup ---
logger = logging.getLogger("SessionMemoryPipeV19_1Logger")  # Version updated
logging.basicConfig(level=logging.INFO)


# === SECTION 4: EMBEDDING WRAPPER ===
class ChromaDBCompatibleEmbedder:
    def __init__(
        self,
        owi_embedding_function: OwiEmbeddingFunction,
        user_context: Optional[Dict] = None,
    ):
        if not callable(owi_embedding_function):
            raise TypeError("owi_embedding_function must be callable.")
        self._owi_embed_func = owi_embedding_function
        self._user = user_context

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        embeddings = [[] for _ in input]
        valid_result = True
        try:
            embeddings = self._owi_embed_func(
                input, prefix=RAG_EMBEDDING_CONTENT_PREFIX, user=self._user
            )
        except TypeError:
            try:
                embeddings = self._owi_embed_func(input)
            except Exception as retry_e:
                logger.error(
                    f"Wrapper: Error embedding retry: {retry_e}", exc_info=True
                )
                embeddings = [[] for _ in input]
        except Exception as e:
            logger.error(f"Wrapper: General error embedding: {e}", exc_info=True)
            embeddings = [[] for _ in input]
        if not isinstance(embeddings, list) or not all(
            isinstance(e, list) for e in embeddings
        ):
            logger.error(f"Wrapper: Invalid output type: {type(embeddings)}.")
            valid_result = False
        elif len(embeddings) != len(input):
            logger.error(
                f"Wrapper: Mismatched count Input:{len(input)}, Output:{len(embeddings)}."
            )
            valid_result = False
        if not valid_result:
            embeddings = [[] for _ in input]
        return embeddings


# === SECTION 5: PIPE CLASS DEFINITION (Includes Scene Generation Valve) ===
class Pipe:
    version = "0.19.1"  # Version updated

    # === SECTION 5.1: VALVES DEFINITION (Includes Scene Generation Valve) ===
    class Valves(BaseModel):
        # --- Logging & Paths ---
        log_file_path: str = Field(
            default=os.getenv(
                ENV_VAR_LOG_FILE_PATH,
                os.path.join(DEFAULT_LOG_DIR, DEFAULT_LOG_FILE_NAME),
            )
        )
        log_level: str = Field(default=os.getenv(ENV_VAR_LOG_LEVEL, DEFAULT_LOG_LEVEL))
        sqlite_db_path: str = Field(
            default=os.getenv(ENV_VAR_SQLITE_DB_PATH, DEFAULT_SQLITE_DB_PATH)
        )
        chromadb_path: str = Field(
            default=os.getenv(ENV_VAR_CHROMADB_PATH, DEFAULT_CHROMADB_PATH)
        )
        # --- Core Memory Config ---
        tokenizer_encoding_name: str = Field(
            default=os.getenv(ENV_VAR_TOKENIZER_ENCODING, DEFAULT_TOKENIZER_ENCODING)
        )
        t1_summarization_chunk_token_target: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET,
                    DEFAULT_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET,
                )
            )
        )
        max_stored_summary_blocks: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_MAX_STORED_SUMMARY_BLOCKS, DEFAULT_MAX_STORED_SUMMARY_BLOCKS
                )
            )
        )
        t0_active_history_token_limit: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_T0_ACTIVE_HISTORY_TOKEN_LIMIT,
                    DEFAULT_T0_ACTIVE_HISTORY_TOKEN_LIMIT,
                )
            )
        )
        # --- Summarizer LLM (Prompt uses script default) ---
        summarizer_api_url: str = Field(
            default=os.getenv(ENV_VAR_SUMMARIZER_API_URL, DEFAULT_SUMMARIZER_API_URL)
        )
        summarizer_api_key: str = Field(
            default=os.getenv(ENV_VAR_SUMMARIZER_API_KEY, DEFAULT_SUMMARIZER_API_KEY)
        )
        summarizer_temperature: float = Field(
            default=float(
                os.getenv(
                    ENV_VAR_SUMMARIZER_TEMPERATURE, DEFAULT_SUMMARIZER_TEMPERATURE
                )
            )
        )
        summarizer_system_prompt: str = Field(
            default=os.getenv(
                ENV_VAR_SUMMARIZER_SYSTEM_PROMPT, DEFAULT_SUMMARIZER_SYSTEM_PROMPT
            )
        )
        # --- RAG Query LLM (Prompt uses script default) ---
        ragq_llm_api_url: str = Field(
            default=os.getenv(ENV_VAR_RAGQ_LLM_API_URL, DEFAULT_RAGQ_LLM_API_URL)
        )
        ragq_llm_api_key: str = Field(
            default=os.getenv(ENV_VAR_RAGQ_LLM_API_KEY, DEFAULT_RAGQ_LLM_API_KEY)
        )
        ragq_llm_temperature: float = Field(
            default=float(
                os.getenv(ENV_VAR_RAGQ_LLM_TEMPERATURE, DEFAULT_RAGQ_LLM_TEMPERATURE)
            )
        )
        ragq_llm_prompt: str = Field(
            default=os.getenv(ENV_VAR_RAGQ_LLM_PROMPT, DEFAULT_RAGQ_LLM_PROMPT)
        )
        # --- RAG & T2 Chroma Config ---
        summary_collection_prefix: str = Field(
            default=os.getenv(
                ENV_VAR_SUMMARY_COLLECTION_PREFIX, DEFAULT_SUMMARY_COLLECTION_PREFIX
            )
        )
        rag_summary_results_count: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_RAG_SUMMARY_RESULTS_COUNT, DEFAULT_RAG_SUMMARY_RESULTS_COUNT
                )
            )
        )
        # --- Final LLM Pass-Through Config ---
        final_llm_api_url: str = Field(
            default=os.getenv(ENV_VAR_FINAL_LLM_API_URL, DEFAULT_FINAL_LLM_API_URL)
        )
        final_llm_api_key: str = Field(
            default=os.getenv(ENV_VAR_FINAL_LLM_API_KEY, DEFAULT_FINAL_LLM_API_KEY)
        )
        final_llm_temperature: float = Field(
            default=float(
                os.getenv(ENV_VAR_FINAL_LLM_TEMPERATURE, DEFAULT_FINAL_LLM_TEMPERATURE)
            )
        )
        final_llm_timeout: int = Field(
            default=int(os.getenv(ENV_VAR_FINAL_LLM_TIMEOUT, DEFAULT_FINAL_LLM_TIMEOUT))
        )
        # --- Refinement / RAG Cache Features (Prompts use library defaults) ---
        enable_rag_cache: bool = Field(
            default=str(
                os.getenv(ENV_VAR_ENABLE_RAG_CACHE, str(DEFAULT_ENABLE_RAG_CACHE))
            ).lower()
            == "true"
        )
        enable_stateless_refinement: bool = Field(
            default=str(
                os.getenv(
                    ENV_VAR_ENABLE_STATELESS_REFINEMENT,
                    str(DEFAULT_ENABLE_STATELESS_REFINEMENT),
                )
            ).lower()
            == "true"
        )
        refiner_llm_api_url: str = Field(
            default=os.getenv(ENV_VAR_REFINER_API_URL, DEFAULT_REFINER_API_URL)
        )
        refiner_llm_api_key: str = Field(
            default=os.getenv(ENV_VAR_REFINER_API_KEY, DEFAULT_REFINER_API_KEY)
        )
        refiner_llm_temperature: float = Field(
            default=float(
                os.getenv(ENV_VAR_REFINER_TEMPERATURE, DEFAULT_REFINER_TEMPERATURE)
            )
        )
        refiner_history_count: int = Field(
            default=int(
                os.getenv(ENV_VAR_REFINER_HISTORY_COUNT, DEFAULT_REFINER_HISTORY_COUNT)
            )
        )
        CACHE_UPDATE_SKIP_OWI_THRESHOLD: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_CACHE_UPDATE_SKIP_OWI_THRESHOLD,
                    DEFAULT_CACHE_UPDATE_SKIP_OWI_THRESHOLD,
                )
            )
        )
        CACHE_UPDATE_SIMILARITY_THRESHOLD: float = Field(
            default=float(
                os.getenv(
                    ENV_VAR_CACHE_UPDATE_SIMILARITY_THRESHOLD,
                    DEFAULT_CACHE_UPDATE_SIMILARITY_THRESHOLD,
                )
            )
        )
        stateless_refiner_skip_threshold: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_STATELESS_REFINER_SKIP_THRESHOLD,
                    DEFAULT_STATELESS_REFINER_SKIP_THRESHOLD,
                )
            )
        )
        # --- Inventory Management Feature (Prompt uses library default) ---
        enable_inventory_management: bool = Field(
            default=str(
                os.getenv(
                    ENV_VAR_ENABLE_INVENTORY_MANAGEMENT,
                    str(DEFAULT_ENABLE_INVENTORY_MANAGEMENT),
                )
            ).lower()
            == "true",
            description="Enable/disable the character inventory management feature.",
        )
        inv_llm_api_url: str = Field(
            default=os.getenv(ENV_VAR_INV_LLM_API_URL, DEFAULT_INV_LLM_API_URL)
        )
        inv_llm_api_key: str = Field(
            default=os.getenv(ENV_VAR_INV_LLM_API_KEY, DEFAULT_INV_LLM_API_KEY)
        )
        inv_llm_temperature: float = Field(
            default=float(
                os.getenv(ENV_VAR_INV_LLM_TEMPERATURE, DEFAULT_INV_LLM_TEMPERATURE)
            )
        )
        # --- Event Hint Feature (Prompt uses library default) ---
        enable_event_hints: bool = Field(
            default=str(
                os.getenv(ENV_VAR_ENABLE_EVENT_HINTS, str(DEFAULT_ENABLE_EVENT_HINTS))
            ).lower()
            == "true",
            description="Enable/disable dynamic event hint generation.",
        )
        event_hint_llm_api_url: str = Field(
            default=os.getenv(
                ENV_VAR_EVENT_HINT_LLM_API_URL, DEFAULT_EVENT_HINT_LLM_API_URL
            )
        )
        event_hint_llm_api_key: str = Field(
            default=os.getenv(
                ENV_VAR_EVENT_HINT_LLM_API_KEY, DEFAULT_EVENT_HINT_LLM_API_KEY
            )
        )
        event_hint_llm_temperature: float = Field(
            default=float(
                os.getenv(
                    ENV_VAR_EVENT_HINT_LLM_TEMPERATURE,
                    DEFAULT_EVENT_HINT_LLM_TEMPERATURE,
                )
            )
        )
        event_hint_history_count: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_EVENT_HINT_HISTORY_COUNT, DEFAULT_EVENT_HINT_HISTORY_COUNT
                )
            )
        )
        # --- Scene Generation Feature (Uses Event Hint LLM endpoint) --- # <<< NEW SECTION
        enable_scene_generation: bool = Field(  # <<< NEW VALVE
            default=str(
                os.getenv(
                    ENV_VAR_ENABLE_SCENE_GENERATION,
                    str(DEFAULT_ENABLE_SCENE_GENERATION),
                )
            ).lower()
            == "true",
            description="Enable/disable dynamic background scene assessment and generation (Uses Event Hint LLM endpoint).",
        )
        # --- General & Debug ---
        include_ack_turns: bool = Field(
            default=str(
                os.getenv(ENV_VAR_INCLUDE_ACK_TURNS, str(DEFAULT_INCLUDE_ACK_TURNS))
            ).lower()
            == "true"
        )
        emit_status_updates: bool = Field(
            default=str(
                os.getenv(ENV_VAR_EMIT_STATUS_UPDATES, str(DEFAULT_EMIT_STATUS_UPDATES))
            ).lower()
            == "true"
        )
        debug_log_final_payload: bool = Field(
            default=str(
                os.getenv(
                    ENV_VAR_DEBUG_LOG_FINAL_PAYLOAD,
                    str(DEFAULT_DEBUG_LOG_FINAL_PAYLOAD),
                )
            ).lower()
            == "true"
        )
        debug_log_raw_input: bool = Field(
            default=str(
                os.getenv(ENV_VAR_DEBUG_LOG_RAW_INPUT, str(DEFAULT_DEBUG_LOG_RAW_INPUT))
            ).lower()
            == "true"
        )

        # --- Post Init Validation ---
        def model_post_init(self, __context: Any) -> None:
            # (Validation logic remains the same for remaining valves)
            if self.t0_active_history_token_limit <= 0:
                self.t0_active_history_token_limit = (
                    DEFAULT_T0_ACTIVE_HISTORY_TOKEN_LIMIT
                )
                logger.warning("Reset t0_limit.")
            if self.t1_summarization_chunk_token_target <= 0:
                self.t1_summarization_chunk_token_target = (
                    DEFAULT_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET
                )
                logger.warning("Reset t1_target.")
            if self.max_stored_summary_blocks < 0:
                self.max_stored_summary_blocks = DEFAULT_MAX_STORED_SUMMARY_BLOCKS
                logger.warning("Reset max_blocks.")
            if self.rag_summary_results_count < 0:
                self.rag_summary_results_count = DEFAULT_RAG_SUMMARY_RESULTS_COUNT
                logger.warning("Reset rag_count.")
            if not (0.0 <= self.summarizer_temperature <= 2.0):
                self.summarizer_temperature = DEFAULT_SUMMARIZER_TEMPERATURE
                logger.warning("Reset summarizer_temp.")
            if not (0.0 <= self.ragq_llm_temperature <= 2.0):
                self.ragq_llm_temperature = DEFAULT_RAGQ_LLM_TEMPERATURE
                logger.warning("Reset ragq_temp.")
            if not (0.0 <= self.final_llm_temperature <= 2.0):
                self.final_llm_temperature = DEFAULT_FINAL_LLM_TEMPERATURE
                logger.warning("Reset final_llm_temp.")
            if self.final_llm_timeout <= 0:
                self.final_llm_timeout = DEFAULT_FINAL_LLM_TIMEOUT
                logger.warning("Reset final_llm_timeout.")
            if not (0.0 <= self.refiner_llm_temperature <= 2.0):
                self.refiner_llm_temperature = DEFAULT_REFINER_TEMPERATURE
                logger.warning("Reset refiner_llm_temp.")
            if self.refiner_history_count < 0:
                self.refiner_history_count = DEFAULT_REFINER_HISTORY_COUNT
                logger.warning("Reset refiner_history_count.")
            if self.stateless_refiner_skip_threshold < 0:
                self.stateless_refiner_skip_threshold = (
                    DEFAULT_STATELESS_REFINER_SKIP_THRESHOLD
                )
                logger.warning("Reset stateless_refiner_skip_threshold.")
            if self.CACHE_UPDATE_SKIP_OWI_THRESHOLD < 0:
                self.CACHE_UPDATE_SKIP_OWI_THRESHOLD = (
                    DEFAULT_CACHE_UPDATE_SKIP_OWI_THRESHOLD
                )
                logger.warning("Reset CACHE_UPDATE_SKIP_OWI_THRESHOLD to default.")
            if not (0.0 <= self.CACHE_UPDATE_SIMILARITY_THRESHOLD <= 1.0):
                self.CACHE_UPDATE_SIMILARITY_THRESHOLD = (
                    DEFAULT_CACHE_UPDATE_SIMILARITY_THRESHOLD
                )
                logger.warning("Reset CACHE_UPDATE_SIMILARITY_THRESHOLD to default.")
            if not (0.0 <= self.inv_llm_temperature <= 2.0):
                self.inv_llm_temperature = DEFAULT_INV_LLM_TEMPERATURE
                logger.warning("Reset inv_llm_temperature to default.")
            if not (0.0 <= self.event_hint_llm_temperature <= 2.0):
                self.event_hint_llm_temperature = DEFAULT_EVENT_HINT_LLM_TEMPERATURE
                logger.warning("Reset event_hint_llm_temperature to default.")
            if self.event_hint_history_count < 0:
                self.event_hint_history_count = DEFAULT_EVENT_HINT_HISTORY_COUNT
                logger.warning("Reset event_hint_history_count to default.")
            # --- No validation needed for scene_llm_temperature anymore ---
            # --- Check kept prompt defaults ---
            if not self.summarizer_system_prompt or not isinstance(
                self.summarizer_system_prompt, str
            ):
                self.summarizer_system_prompt = DEFAULT_SUMMARIZER_SYSTEM_PROMPT
                logger.warning("Reset summarizer_system_prompt to script default.")
            if not self.ragq_llm_prompt or not isinstance(self.ragq_llm_prompt, str):
                self.ragq_llm_prompt = DEFAULT_RAGQ_LLM_PROMPT
                logger.warning("Reset ragq_llm_prompt to script default.")
            logger.info("Valves model_post_init validation complete.")

    # --- User Valves ---
    class UserValves(BaseModel):
        long_term_goal: str = Field(
            default="",
            description="A user-defined long-term goal or instruction for the session.",
        )
        process_owi_rag: bool = Field(
            default=True,
            description="Enable/disable processing of the RAG context provided by OpenWebUI.",
        )
        text_block_to_remove: str = Field(
            default="",
            description="Specify an exact block of text to remove from the system prompt.",
        )

    # === SECTION 5.2: INITIALIZATION METHOD ===
    def __init__(self):
        self.type = "pipe"
        self.name = (
            f"SESSION_MEMORY PIPE (v{self.version} - Scene Gen)"  # Version updated
        )
        self.logger = logger
        self.logger.info(f"Initializing '{self.name}'...")
        if not I4_LLM_AGENT_AVAILABLE:
            error_msg = f"i4_llm_agent library (v0.1.7+ required) failed import: {IMPORT_ERROR_DETAILS}. Pipe cannot function."
            self.logger.critical(error_msg)
            raise ImportError(error_msg)
        try:
            self.valves = self.Valves()
            # Modified logging to exclude removed prompt valves
            init_log_valves = {
                k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                for k, v in self.valves.model_dump().items()
                if "api_key" not in k
            }
            # Remove prompt keys explicitly for logging if they weren't removed by the filter
            init_log_valves.pop("summarizer_system_prompt", None)
            init_log_valves.pop("ragq_llm_prompt", None)
            # Add feature flags
            init_log_valves["INIT_enable_inventory"] = getattr(
                self.valves, "enable_inventory_management", "LOAD_FAILED"
            )
            init_log_valves["INIT_enable_event_hints"] = getattr(
                self.valves, "enable_event_hints", "LOAD_FAILED"
            )
            init_log_valves["INIT_enable_scene_generation"] = getattr(
                self.valves, "enable_scene_generation", "LOAD_FAILED"
            )  # <<< ADDED
            init_log_valves["INIT_debug_log_final_payload"] = getattr(
                self.valves, "debug_log_final_payload", "LOAD_FAILED"
            )
            self.logger.info(
                f"Pipe.__init__: self.valves loaded (Prompts use defaults): {init_log_valves}"
            )
            # Warnings for missing keys remain the same
            if not self.valves.summarizer_api_key:
                self.logger.warning("Global Summarizer API Key MISSING.")
            if not self.valves.ragq_llm_api_key:
                self.logger.warning("Global RAG Query LLM API Key MISSING.")
            if (
                self.valves.enable_rag_cache or self.valves.enable_stateless_refinement
            ) and not self.valves.refiner_llm_api_key:
                self.logger.warning(
                    "A refinement feature is ENABLED globally but Refiner API Key MISSING!"
                )
            if (
                self.valves.enable_inventory_management
                and not self.valves.inv_llm_api_key
            ):
                self.logger.warning(
                    "Inventory Management ENABLED but Inventory LLM API Key MISSING!"
                )
            if (
                self.valves.enable_event_hints
                and not self.valves.event_hint_llm_api_key
            ):
                self.logger.warning(
                    "Event Hints ENABLED but Event Hint LLM API Key MISSING!"
                )
            # --- No warning needed for scene_llm_api_key ---
        except Exception as e:
            self.logger.error(f"CRITICAL Global Valve init error: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize pipe Global Valves") from e
        try:
            self.configure_logger()
            self.logger.info(
                f"Logger configured. Level: {logging.getLevelName(self.logger.getEffectiveLevel())}, Path: {self.valves.log_file_path or 'Console'}"
            )
        except Exception as e:
            print(f"CRITICAL Logger config error: {e}")
            self.logger.critical(f"Logger config failed: {e}", exc_info=True)
        # --- DB and Manager Initialization (Unchanged) ---
        self._sqlite_conn = None
        self._sqlite_cursor = None
        try:
            sqlite_db_path = self.valves.sqlite_db_path
            self.logger.info(f"Init SQLite connection: {sqlite_db_path}")
            os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
            self._sqlite_conn = sqlite3.connect(
                sqlite_db_path, check_same_thread=False, isolation_level=None
            )
            self._sqlite_conn.execute("PRAGMA journal_mode=WAL;")
            self._sqlite_cursor = self._sqlite_conn.cursor()
            self.logger.info(
                f"SQLite connection and cursor established for '{sqlite_db_path}'."
            )
        except Exception as e:
            self.logger.error(f"SQLite init error: {e}", exc_info=True)
            self._sqlite_conn.close() if self._sqlite_conn else None
            self._sqlite_conn, self._sqlite_cursor = None, None
        self._chroma_client = None
        if CHROMADB_AVAILABLE:
            try:
                chromadb_path = self.valves.chromadb_path
                self.logger.info(f"Init ChromaDB client: {chromadb_path}")
                os.makedirs(chromadb_path, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=chromadb_path)
                self.logger.info(
                    f"ChromaDB client initialized for path '{chromadb_path}'."
                )
            except Exception as e:
                self._chroma_client = None
                self.logger.error(f"ChromaDB client init error: {e}", exc_info=True)
        else:
            self.logger.warning(
                "ChromaDB library not available. T2 memory features disabled."
            )
        try:
            self.session_manager = SessionManager()
            self.logger.info("SessionManager initialized.")
        except Exception as e:
            self.logger.critical(
                f"Failed to initialize SessionManager: {e}", exc_info=True
            )
            raise RuntimeError("Failed to initialize SessionManager") from e
        try:
            # Pass self.valves which now lacks prompt template fields
            self.orchestrator = SessionPipeOrchestrator(
                config=self.valves,
                session_manager=self.session_manager,
                sqlite_cursor=self._sqlite_cursor,
                chroma_client=self._chroma_client,
                logger_instance=self.logger,
            )
            self.logger.info("SessionPipeOrchestrator initialized.")
        except Exception as e:
            self.logger.critical(
                f"Failed to initialize SessionPipeOrchestrator: {e}", exc_info=True
            )
            raise RuntimeError("Failed to initialize SessionPipeOrchestrator") from e
        self._current_event_emitter = None
        self._owi_embedding_func_cache = None
        self.logger.info(f"'{self.name}' initialization complete.")

    # === SECTION 5.3: LOGGER CONFIGURATION ===
    def configure_logger(self):
        # (No changes needed here)
        log_level_str = self.valves.log_level.upper()
        log_path = self.valves.log_file_path
        try:
            numeric_level = getattr(logging, log_level_str)
        except (AttributeError, AssertionError):
            print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to DEBUG.")
            numeric_level = logging.DEBUG
        for handler in list(self.logger.handlers):
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception as e:
                print(f"Error removing log handler: {e}")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        if log_path:
            log_dir = os.path.dirname(log_path)
            try:
                os.makedirs(log_dir, exist_ok=True)
                fh = RotatingFileHandler(
                    log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
                )
                fh.setLevel(numeric_level)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
            except PermissionError:
                self.logger.error(
                    f"Permission denied for log file: {log_path}. Console only."
                )
            except Exception as e:
                self.logger.error(
                    f"Error setting up file logging handler: {e}", exc_info=True
                )
        else:
            self.logger.info("No log file path configured. Console only.")
        self.logger.setLevel(numeric_level)
        self.logger.propagate = False

    # === SECTION 5.4: LIFECYCLE METHODS ===
    async def on_startup(self):
        # (No changes needed here)
        self.logger.info(f"on_startup: '{self.name}' (v{self.version}) starting...")
        init_success_main = False
        if self._sqlite_cursor:
            try:
                self.logger.info(
                    "Attempting main SQLite table initialization (T1, Cache, Inv, WorldState, SceneState)..."
                )  # <<< Updated log message
                init_success_main = await asyncio.to_thread(
                    initialize_sqlite_tables, self._sqlite_cursor
                )
                (
                    self.logger.info(
                        "Main SQLite table initialization reported success."
                    )
                    if init_success_main
                    else self.logger.error(
                        "Main SQLite table initialization reported failure."
                    )
                )
            except Exception as e:
                self.logger.error(
                    f"Error during main SQLite table initialization: {e}", exc_info=True
                )
                init_success_main = False
            # Explicit checks remain useful for debugging specific table issues
            self.logger.info("Explicitly attempting Inventory table initialization...")
            try:
                from i4_llm_agent.database import _sync_initialize_inventory_table

                init_success_inv = await asyncio.to_thread(
                    _sync_initialize_inventory_table, self._sqlite_cursor
                )
                (
                    self.logger.info(
                        "Explicit Inventory table initialization reported success."
                    )
                    if init_success_inv
                    else self.logger.warning(
                        "Explicit Inventory table initialization reported failure (may already exist or error occurred)."
                    )
                )
            except ImportError:
                self.logger.error(
                    "Failed to import _sync_initialize_inventory_table for explicit check."
                )
            except Exception as e_inv:
                self.logger.error(
                    f"Error during explicit Inventory table initialization: {e_inv}",
                    exc_info=True,
                )
            self.logger.info(
                "Explicitly attempting World State table initialization..."
            )
            try:
                from i4_llm_agent.database import _sync_initialize_world_state_table

                init_success_ws = await asyncio.to_thread(
                    _sync_initialize_world_state_table, self._sqlite_cursor
                )
                (
                    self.logger.info(
                        "Explicit World State table initialization reported success."
                    )
                    if init_success_ws
                    else self.logger.warning(
                        "Explicit World State table initialization reported failure (may already exist or error occurred)."
                    )
                )
            except ImportError:
                self.logger.error(
                    "Failed to import _sync_initialize_world_state_table for explicit check."
                )
            except Exception as e_ws:
                self.logger.error(
                    f"Error during explicit World State table initialization: {e_ws}",
                    exc_info=True,
                )
            self.logger.info(
                "Explicitly attempting Scene State table initialization..."
            )  # <<< ADDED CHECK
            try:  # <<< ADDED CHECK
                from i4_llm_agent.database import (
                    _sync_initialize_scene_state_table,
                )  # <<< ADDED CHECK

                init_success_scene = await asyncio.to_thread(
                    _sync_initialize_scene_state_table, self._sqlite_cursor
                )  # <<< ADDED CHECK
                (
                    self.logger.info(
                        "Explicit Scene State table initialization reported success."
                    )
                    if init_success_scene
                    else self.logger.warning(
                        "Explicit Scene State table initialization reported failure (may already exist or error occurred)."
                    )
                )  # <<< ADDED CHECK
            except ImportError:
                self.logger.error(
                    "Failed to import _sync_initialize_scene_state_table for explicit check."
                )  # <<< ADDED CHECK
            except Exception as e_scene:
                self.logger.error(
                    f"Error during explicit Scene State table initialization: {e_scene}",
                    exc_info=True,
                )  # <<< ADDED CHECK

        else:
            self.logger.warning(
                "SQLite cursor not available on startup. Cannot initialize tables."
            )
        if self._chroma_client:
            try:
                self.logger.info("Checking ChromaDB connection...")
                version = await asyncio.to_thread(self._chroma_client.heartbeat)
                self.logger.info(
                    f"ChromaDB connection verified (heartbeat: {version}ns)."
                )
            except Exception as e:
                self.logger.error(
                    f"ChromaDB connection check failed on startup: {e}", exc_info=True
                )
        else:
            self.logger.warning("ChromaDB client not available on startup.")
        self.logger.info(f"'{self.name}' startup complete.")

    async def on_shutdown(self):
        # (No changes needed here)
        self.logger.info(f"on_shutdown: '{self.name}' shutting down...")
        if self._sqlite_conn:
            try:
                self.logger.info("Closing SQLite...")
                self._sqlite_conn.close()
                self.logger.info("SQLite closed.")
            except Exception as e:
                self.logger.error(f"Error closing SQLite: {e}", exc_info=True)
        if self._chroma_client:
            self.logger.info("Resetting ChromaDB client ref.")
            self._chroma_client = None
        self.logger.info("Closing logger handlers.")
        for handler in list(self.logger.handlers):
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception as e:
                print(f"ERROR closing log handler: {handler}: {e}")
        self.logger.info(f"'{self.name}' shutdown complete.")

    async def on_valves_updated(self):
        # (Simplified logging due to removed prompt valves)
        self.logger.info(
            f"on_valves_updated: Reloading global settings for '{self.name}'..."
        )
        old_log_path = getattr(self.valves, "log_file_path", None)
        old_log_level = getattr(self.valves, "log_level", None)
        old_sqlite_path = getattr(self.valves, "sqlite_db_path", None)
        old_chromadb_path = getattr(self.valves, "chromadb_path", None)
        try:
            self.valves = self.Valves()
            # Modified logging to exclude removed prompt valves
            update_log_valves = {
                k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                for k, v in self.valves.model_dump().items()
                if "api_key" not in k
            }
            update_log_valves.pop("summarizer_system_prompt", None)
            update_log_valves.pop("ragq_llm_prompt", None)
            # Add feature flags
            update_log_valves["UPDATE_enable_inventory"] = getattr(
                self.valves, "enable_inventory_management", "LOAD_FAILED"
            )
            update_log_valves["UPDATE_enable_event_hints"] = getattr(
                self.valves, "enable_event_hints", "LOAD_FAILED"
            )
            update_log_valves["UPDATE_enable_scene_generation"] = getattr(
                self.valves, "enable_scene_generation", "LOAD_FAILED"
            )  # <<< ADDED
            update_log_valves["UPDATE_debug_log_final_payload"] = getattr(
                self.valves, "debug_log_final_payload", "LOAD_FAILED"
            )
            self.logger.info(
                f"Pipe.on_valves_updated: self.valves RE-loaded (Prompts use defaults): {update_log_valves}"
            )
            if hasattr(self, "orchestrator"):
                # Pass the updated config (without prompt templates) to orchestrator
                self.orchestrator.config = self.valves
                self.logger.info("Orchestrator config updated.")
            else:
                self.logger.warning(
                    "Orchestrator object not found during valves update."
                )
        except Exception as e:
            self.logger.error(
                f"Error re-initializing global valves: {e}. Pipe may use old settings.",
                exc_info=True,
            )
        # --- Checks for path/log changes remain ---
        new_log_path = getattr(self.valves, "log_file_path", None)
        new_log_level = getattr(self.valves, "log_level", None)
        if old_log_path != new_log_path or old_log_level != new_log_level:
            self.logger.info("Log settings changed. Reconfiguring logger...")
            self.configure_logger()
            self.logger.info("Logger reconfigured successfully.")
        new_sqlite_path = getattr(self.valves, "sqlite_db_path", None)
        if old_sqlite_path != new_sqlite_path:
            self.logger.warning(f"SQLite DB path changed. Restart might be required.")
        new_chromadb_path = getattr(self.valves, "chromadb_path", None)
        if old_chromadb_path != new_chromadb_path:
            self.logger.warning(f"ChromaDB path changed. Restart might be required.")
        self.logger.info("Clearing OWI embedding function cache.")
        self._owi_embedding_func_cache = None
        self.logger.info("on_valves_updated: Reload finished.")

    # === SECTION 5.7: HELPER - OWI EMBEDDING FUNCTION RETRIEVAL ===
    def _get_owi_embedding_function(
        self, __request__: Request, __user__: Optional[Dict]
    ) -> Optional[OwiEmbeddingFunction]:
        # (No changes needed here)
        if (
            hasattr(self, "_owi_embedding_func_cache")
            and self._owi_embedding_func_cache
        ):
            return self._owi_embedding_func_cache
        if not (OWI_RAG_UTILS_AVAILABLE and get_embedding_function):
            self.logger.error("Embeddings: OWI utils unavailable.")
            return None
        if (
            not __request__
            or not hasattr(__request__, "app")
            or not hasattr(__request__.app, "state")
        ):
            self.logger.error("Embeddings: Request context/app state missing.")
            return None
        config = getattr(__request__.app.state, "config", None)
        if not config:
            self.logger.error("Embeddings: OWI config missing.")
            return None
        embedding_func = None
        try:
            engine = getattr(config, "RAG_EMBEDDING_ENGINE", "")
            model = getattr(config, "RAG_EMBEDDING_MODEL", None)
            batch_size = getattr(config, "RAG_EMBEDDING_BATCH_SIZE", 1)
            openai_url = getattr(config, "RAG_OPENAI_API_BASE_URL", None)
            openai_key = getattr(config, "RAG_OPENAI_API_KEY", None)
            ollama_url = getattr(config, "RAG_OLLAMA_BASE_URL", None)
            ef_object = getattr(__request__.app.state, "ef", None)
            if engine == "" and not ef_object:
                self.logger.error("Embeddings: Local engine but no 'ef' object found.")
                return None
            use_url, use_key = None, None
            if not ef_object:
                if engine == "openai":
                    use_url, use_key = openai_url, openai_key
                elif engine == "ollama":
                    use_url, use_key = ollama_url, None
            self.logger.info(
                f"Calling OWI get_embedding_function (Engine:'{engine or 'Local (ef)'}', Model:'{model or 'Default'}', EF_Provided={bool(ef_object)})"
            )
            embedding_func = get_embedding_function(
                embedding_engine=engine,
                embedding_model=model,
                embedding_function=ef_object,
                url=use_url,
                key=use_key,
                embedding_batch_size=batch_size,
            )
            if embedding_func:
                self.logger.info("Embeddings: Successfully retrieved function.")
                self._owi_embedding_func_cache = embedding_func
            else:
                self.logger.error("Embeddings: OWI util returned None.")
        except Exception as e:
            self.logger.error(f"Embeddings: Error during retrieval: {e}", exc_info=True)
        return embedding_func

    # === SECTION 5.8 & 5.9: HELPER - Debug Logging (Unchanged) ===
    def _get_debug_log_path(self, suffix: str) -> Optional[str]:
        # (No changes needed here)
        func_logger = self.logger
        func_logger.debug(f"_get_debug_log_path called with suffix: '{suffix}'")
        try:
            base_log_path = self.valves.log_file_path
            if not base_log_path:
                func_logger.error(
                    "Cannot get debug path: Main log_file_path valve is empty."
                )
                return None
            func_logger.debug(f"Base log path from valves: '{base_log_path}'")
            log_dir = os.path.dirname(base_log_path)
            func_logger.debug(f"Target log directory: '{log_dir}'")
            try:
                func_logger.debug(
                    f"Attempting os.makedirs for: '{log_dir}' (exist_ok=True)"
                )
                os.makedirs(log_dir, exist_ok=True)
                func_logger.debug(f"os.makedirs command finished for: '{log_dir}'")
            except PermissionError as pe:
                func_logger.error(
                    f"PERMISSION ERROR creating log directory '{log_dir}': {pe}"
                )
                return None
            except Exception as e_mkdir:
                func_logger.error(
                    f"Error creating log directory '{log_dir}': {e_mkdir}",
                    exc_info=True,
                )
                return None
            base_name, _ = os.path.splitext(os.path.basename(base_log_path))
            debug_filename = f"{base_name}{suffix}.log"
            final_path = os.path.join(log_dir, debug_filename)
            func_logger.info(f"Constructed debug log path: '{final_path}'")
            return final_path
        except Exception as e:
            func_logger.error(f"Failed get debug path '{suffix}': {e}", exc_info=True)
            return None

    def _log_debug_raw_input(self, session_id: str, body: Dict):
        # (No changes needed here)
        if not getattr(self.valves, "debug_log_raw_input", False):
            return
        debug_log_path = self._get_debug_log_path(".DEBUG_INPUT")
        if not debug_log_path:
            self.logger.error(
                f"[{session_id}] Cannot log raw input: No path determined."
            )
            return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_entry = {
                "ts": ts,
                "pipe": self.version,
                "sid": session_id,
                "body": body,
            }
            self.logger.debug(
                f"[{session_id}] Attempting to write RAW INPUT debug log to: {debug_log_path}"
            )
            with open(debug_log_path, "a", encoding="utf-8") as f:
                json.dump(log_entry, f)
                f.write("\n")
            self.logger.debug(f"[{session_id}] Successfully wrote RAW INPUT debug log.")
        except Exception as e:
            self.logger.error(
                f"[{session_id}] Failed write debug raw input log: {e}", exc_info=True
            )

    # === SECTION 5.11: MAIN PIPE METHOD (Unchanged) ===
    async def pipe(
        self,
        body: dict,
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        __chat_id__: Optional[str] = None,
        __files__: Optional[List] = None,
        __event_call__=None,
        __task__=None,
        __task_body__: Optional[dict] = None,
    ) -> Union[dict, AsyncGenerator[str, None], str]:
        # (Logic mostly unchanged, uses updated orchestrator config)
        pipe_start_time_iso = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Pipe.pipe v{self.version}: Entered at {pipe_start_time_iso}")
        self._current_event_emitter = __event_emitter__
        session_id = "uninitialized_session"
        user_id = "default_user"

        async def emit_status(description: str, done: bool = False):
            # (Emit status logic unchanged)
            log_session_id_prefix = (
                f"[{session_id}]"
                if session_id != "uninitialized_session"
                else "[startup/error]"
            )
            if (
                getattr(self.valves, "emit_status_updates", True)
                and self._current_event_emitter
                and callable(self._current_event_emitter)
            ):
                try:
                    await self._current_event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": str(description),
                                "done": bool(done),
                            },
                        }
                    )
                except Exception as e_emit:
                    self.logger.warning(
                        f"{log_session_id_prefix} Pipe failed emit status '{description}': {e_emit}"
                    )
            else:
                self.logger.debug(
                    f"{log_session_id_prefix} Pipe status update (not emitted): '{description}' (Done: {done})"
                )

        try:
            # --- Input Validation and Setup ---
            await emit_status("Status: Pipe validating input...")
            if not isinstance(body, dict):
                await emit_status("ERROR: Invalid input type.", done=True)
                return {"error": "Invalid input body type.", "status_code": 400}
            if not __request__:
                self.logger.warning("__request__ context missing.")

            # --- User Info Extraction ---
            raw_user_valves_data = None
            if __user__ and isinstance(__user__, dict):
                user_id = __user__.get("id", "missing_id")
                user_info_log = {k: v for k, v in __user__.items() if k != "valves"}
                self.logger.debug(
                    f"Pipe.pipe: Received __user__ info (excluding valves): {user_info_log}"
                )
                raw_user_valves_data = __user__.get("valves")
                self.logger.debug(
                    f"Pipe.pipe: Received __user__['valves'] raw data: {raw_user_valves_data}"
                )
            else:
                user_id = "default_user"
                self.logger.warning(
                    f"User info/ID missing in __user__. Using '{user_id}'. Received: {__user__}"
                )

            # --- Session ID ---
            chat_id = __chat_id__
            if not chat_id or not isinstance(chat_id, str) or len(chat_id.strip()) == 0:
                await emit_status(
                    "ERROR: Cannot isolate session (missing chat_id).", True
                )
                return {
                    "error": "Pipe requires a valid __chat_id__.",
                    "status_code": 500,
                }
            safe_chat_id_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", chat_id)
            session_id = f"user_{user_id}_chat_{safe_chat_id_part}"
            self.logger.info(f"Pipe Session ID: {session_id}")

            # --- Orchestrator Config Update ---
            if hasattr(self, "orchestrator") and hasattr(self, "valves"):
                # Pass the potentially updated self.valves (without prompt templates)
                self.orchestrator.config = self.valves
                self.orchestrator.pipe_logger = self.logger
                self.orchestrator.pipe_debug_path_getter = self._get_debug_log_path
                self.logger.debug(
                    f"[{session_id}] Pipe: Force-updated orchestrator config, logger, and path getter refs."
                )
            elif not hasattr(self, "orchestrator"):
                self.logger.error(f"[{session_id}] Pipe: Orchestrator missing!")
                await emit_status("ERROR: Orchestrator Component Missing.", True)
                return {
                    "error": "Internal Pipe Error: Orchestrator component failed.",
                    "status_code": 500,
                }

            # --- Debug Logging Raw Input ---
            self._log_debug_raw_input(session_id, body)

            # --- Session State ---
            session_state = self.session_manager.get_or_create_session(session_id)

            # --- User Valves Parsing ---
            valves_data_from_user = (
                __user__.get("valves") if isinstance(__user__, dict) else None
            )
            user_valves_obj = None
            data_type = type(valves_data_from_user)
            data_value_str = str(valves_data_from_user)
            self.logger.debug(
                f"[{session_id}] PRE-CHECK: Received valves_data_from_user Type={data_type}, Value='{data_value_str[:150]}...'"
            )
            if isinstance(valves_data_from_user, self.UserValves):
                self.logger.info(
                    f"[{session_id}] Received UserValves object directly. Using it."
                )
                user_valves_obj = valves_data_from_user
                parsed_text_block = getattr(
                    user_valves_obj, "text_block_to_remove", "ATTR_MISSING"
                )
                parsed_process_rag = getattr(
                    user_valves_obj, "process_owi_rag", "ATTR_MISSING"
                )
                parsed_long_term = getattr(
                    user_valves_obj, "long_term_goal", "ATTR_MISSING"
                )
                log_text_block = parsed_text_block[:100] + (
                    "..." if len(parsed_text_block) > 100 else ""
                )
                log_long_term = parsed_long_term[:100] + (
                    "..." if len(parsed_long_term) > 100 else ""
                )
                self.logger.debug(
                    f"[{session_id}] Values from received UserValves object: text_block_to_remove='{log_text_block}', process_owi_rag={parsed_process_rag}, long_term_goal='{log_long_term}'"
                )
                if (
                    user_valves_obj.long_term_goal == ""
                    and user_valves_obj.process_owi_rag is True
                    and user_valves_obj.text_block_to_remove == ""
                ):
                    self.logger.warning(
                        f"[{session_id}] Received UserValves object matches defaults. Original string input might have failed OWI pre-parsing."
                    )
            elif isinstance(valves_data_from_user, dict):
                self.logger.info(
                    f"[{session_id}] Received valves data as dict. Initializing UserValves model."
                )
                try:
                    user_valves_obj = self.UserValves(**valves_data_from_user)
                except Exception as e_init_dict:
                    self.logger.error(
                        f"[{session_id}] Failed to initialize UserValves from dict {valves_data_from_user}: {e_init_dict}. Using defaults.",
                        exc_info=True,
                    )
                    user_valves_obj = self.UserValves()
            else:
                log_reason = (
                    "not provided"
                    if valves_data_from_user is None
                    else f"unexpected Type={data_type}"
                )
                self.logger.warning(
                    f"[{session_id}] User valves data {log_reason}. Using default UserValves object."
                )
                user_valves_obj = self.UserValves()
            if user_valves_obj is None:
                self.logger.error(
                    f"[{session_id}] user_valves_obj was None after checks. Forcing defaults."
                )
                user_valves_obj = self.UserValves()
            self.session_manager.set_user_valves(session_id, user_valves_obj)
            self.logger.debug(
                f"[{session_id}] Stored final UserValves object in session state."
            )

            # --- History & Regen Check ---
            await emit_status("Status: Pipe handling history...")
            incoming_messages = body.get("messages", [])
            previous_input = self.session_manager.get_previous_input_messages(
                session_id
            )
            is_regeneration_heuristic = (
                previous_input is not None
                and incoming_messages == previous_input
                and len(incoming_messages) > 0
            )
            log_msg = f"[{session_id}] Regeneration heuristic: {'DETECTED' if is_regeneration_heuristic else 'Not detected'}."
            (
                self.logger.info(log_msg)
                if is_regeneration_heuristic
                else self.logger.debug(log_msg)
            )
            self.session_manager.set_previous_input_messages(
                session_id, incoming_messages.copy()
            )

            # --- Embedding Setup ---
            await emit_status("Status: Pipe resolving embeddings...")
            owi_embed_func = None
            chroma_embed_wrapper = None
            if self._chroma_client and __request__:
                owi_embed_func = self._get_owi_embedding_function(__request__, __user__)
                if owi_embed_func:
                    try:
                        chroma_embed_wrapper = ChromaDBCompatibleEmbedder(
                            owi_embed_func, __user__
                        )
                        self.logger.info(
                            f"[{session_id}] ChromaDB embedder wrapper created."
                        )
                    except Exception as wrapper_e:
                        self.logger.error(
                            f"[{session_id}] Embedding wrapper creation failed: {wrapper_e}.",
                            exc_info=True,
                        )
                else:
                    self.logger.error(
                        f"[{session_id}] Failed retrieve OWI embedding func. T2 RAG may be impaired."
                    )
            elif not self._chroma_client:
                self.logger.debug(
                    f"[{session_id}] Skipping embedding setup: Chroma client unavailable."
                )
            elif not __request__:
                self.logger.debug(
                    f"[{session_id}] Skipping embedding setup: OWI Request context missing."
                )

            # --- Call Orchestrator ---
            await emit_status("Status: Pipe delegating to orchestrator...")
            self.logger.info(f"[{session_id}] Calling orchestrator.process_turn...")
            # Orchestrator will use its internally referenced config (updated above)
            orchestrator_result = await self.orchestrator.process_turn(
                session_id=session_id,
                user_id=user_id,
                body=body,
                user_valves=user_valves_obj,
                event_emitter=self._current_event_emitter,
                embedding_func=owi_embed_func,
                chroma_embed_wrapper=chroma_embed_wrapper,
                is_regeneration_heuristic=is_regeneration_heuristic,
            )
            self.logger.info(
                f"[{session_id}] Orchestrator returned result type: {type(orchestrator_result).__name__}"
            )

            # --- Final Cleanup & Return ---
            pipe_end_time_iso = datetime.now(timezone.utc).isoformat()
            self.logger.info(
                f"Pipe.pipe v{self.version} [{session_id}]: Finished at {pipe_end_time_iso}"
            )
            return orchestrator_result

        except asyncio.CancelledError:
            self.logger.info(
                f"Pipe.pipe [{session_id}]: Processing cancelled by external signal."
            )
            await emit_status("Status: Processing stopped by user.", done=True)
            raise
        except Exception as e_pipe_global:
            self.logger.critical(
                f"Pipe.pipe [{session_id if 'session_id' in locals() else 'unknown'}]: UNHANDLED PIPE SETUP/WRAPPER EXCEPTION: {e_pipe_global}",
                exc_info=True,
            )
            await emit_status(
                f"ERROR: Pipe Failed ({type(e_pipe_global).__name__})", done=True
            )
            return f"Apologies, a critical error occurred in the Session Memory Pipe: {type(e_pipe_global).__name__}."


# === SECTION 6: END OF SCRIPT ===
