# === SECTION 1: METADATA HEADER ===
# --- REQUIRED METADATA HEADER ---
"""
title: SESSION_MEMORY PIPE (v0.18.0 - Two-Step RAG Cache)
author: Petr jilek & Assistant Gemini 2.5
version: 0.18.0
description: Uses i4_llm_agent library (v0.1.3+). Implements session isolation using __chat_id__. Includes optional Two-Step RAG Cache feature: 1) Update persistent cache by merging OWI RAG. 2) Select turn-relevant context from updated cache (+OWI RAG) for final prompt. Integrates T0 history, T1/T2 memory, RAG query gen, payload construction, ACK valve, status emit, debug logs.
requirements: pydantic, chromadb, i4_llm_agent>=0.1.3, tiktoken, open_webui (internal utils)
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
import uuid
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
)

# /////////////////////////////////////////
# /// 2.2: Core & OWI Imports           ///
# /////////////////////////////////////////
try:
    from pydantic import BaseModel, Field
except ImportError:

    class BaseModel:
        pass

    def Field(*args, **kwargs):
        return kwargs.get("default")

    logging.warning("pydantic not found. Valve validation might be limited.")

from fastapi import Request

# /////////////////////////////////////////
# /// 2.3: ChromaDB Import              ///
# /////////////////////////////////////////
try:
    import chromadb
    from chromadb.api.models.Collection import Collection
    from chromadb.errors import InvalidDimensionException

    ChromaEmbeddingFunction = Callable[[Sequence[str]], List[List[float]]]
    CHROMADB_AVAILABLE = True
    CHROMADB_IMPORT_ERROR = None
except ImportError as e:
    CHROMADB_AVAILABLE = False
    CHROMADB_IMPORT_ERROR = str(e)
    Collection = None
    ChromaEmbeddingFunction = Callable
    InvalidDimensionException = Exception
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
# Requires i4_llm_agent version >= 0.1.3
try:
    # Import necessary components directly from the top-level package
    from i4_llm_agent import (
        # API Client
        call_google_llm_api,
        # History Utils
        format_history_for_llm,
        get_recent_turns,
        DIALOGUE_ROLES,
        # T1 Memory Management
        manage_tier1_summarization,
        # Prompting Utilities
        construct_final_llm_payload,
        clean_context_tags,
        generate_rag_query,
        combine_background_context,  # <<< Correctly imported here
        process_system_prompt,
        # Stateless Refinement
        refine_external_context,
        format_stateless_refiner_prompt,
        DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE,
        # Two-Step RAG Cache
        initialize_rag_cache_table,
        update_rag_cache,
        select_final_context,
        DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT,
        DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT,
        # Token Counting Util
        count_tokens,
    )

    # Set flag indicating library availability
    I4_LLM_AGENT_AVAILABLE = True

    # Assign Aliases for use within the pipe script
    LLM_CALL_FUNC = call_google_llm_api
    HISTORY_FORMAT_FUNC = format_history_for_llm
    HISTORY_GET_RECENT_FUNC = get_recent_turns
    MEMORY_MANAGE_FUNC = manage_tier1_summarization
    PROMPT_PAYLOAD_CONSTRUCT_FUNC = construct_final_llm_payload
    PROMPT_CLEAN_FUNC = clean_context_tags
    PROMPT_RAGQ_GEN_FUNC = generate_rag_query
    PROMPT_CONTEXT_COMBINE_FUNC = combine_background_context  # Alias uses imported name
    PROMPT_PROCESS_SYSTEM_PROMPT_FUNC = process_system_prompt
    UTIL_COUNT_TOKENS_FUNC = count_tokens
    DIALOGUE_ROLES_LIST = DIALOGUE_ROLES
    # Stateless Refinement Aliases
    STATELESS_REFINE_FUNC = refine_external_context
    STATELESS_REFINE_PROMPT_FORMAT_FUNC = (
        format_stateless_refiner_prompt  # Alias if needed internally
    )
    DEFAULT_STATELESS_PROMPT = DEFAULT_STATELESS_REFINER_PROMPT_TEMPLATE
    # RAG Cache Aliases
    INIT_RAG_CACHE_TABLE_FUNC = initialize_rag_cache_table
    CACHE_UPDATE_FUNC = update_rag_cache
    FINAL_CONTEXT_SELECT_FUNC = select_final_context
    # Default RAG Cache Prompt Aliases
    DEFAULT_CACHE_UPDATE_PROMPT = DEFAULT_CACHE_UPDATE_TEMPLATE_TEXT
    DEFAULT_FINAL_SELECT_PROMPT = DEFAULT_FINAL_CONTEXT_SELECTION_TEMPLATE_TEXT

    # Clear import error details on success
    IMPORT_ERROR_DETAILS = None

except ImportError as e:
    # Handle import failure
    I4_LLM_AGENT_AVAILABLE = False
    IMPORT_ERROR_DETAILS = str(e)
    logging.getLogger("SessionMemoryPipe_startup").critical(
        f"CRITICAL: Failed import 'i4_llm_agent' (v0.1.3+): {e}."
    )
    # Assign None to all aliases to prevent NameErrors later
    LLM_CALL_FUNC = None
    HISTORY_FORMAT_FUNC = None
    HISTORY_GET_RECENT_FUNC = None
    MEMORY_MANAGE_FUNC = None
    PROMPT_PAYLOAD_CONSTRUCT_FUNC = None
    PROMPT_CLEAN_FUNC = None
    PROMPT_RAGQ_GEN_FUNC = None
    PROMPT_CONTEXT_COMBINE_FUNC = None
    PROMPT_PROCESS_SYSTEM_PROMPT_FUNC = None
    UTIL_COUNT_TOKENS_FUNC = None
    DIALOGUE_ROLES_LIST = ["user", "assistant"]  # Fallback
    STATELESS_REFINE_FUNC = None
    STATELESS_REFINE_PROMPT_FORMAT_FUNC = None
    INIT_RAG_CACHE_TABLE_FUNC = None
    CACHE_UPDATE_FUNC = None
    FINAL_CONTEXT_SELECT_FUNC = None
    DEFAULT_CACHE_UPDATE_PROMPT = "[Error: Prompt Unavailable]"
    DEFAULT_FINAL_SELECT_PROMPT = "[Error: Prompt Unavailable]"
    DEFAULT_STATELESS_PROMPT = "[Error: Prompt Unavailable]"

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
DEFAULT_LOG_FILE_NAME = "session_memory_v18_0_pipe_log.log"  # Version updated
DEFAULT_LOG_LEVEL = "DEBUG"
ENV_VAR_LOG_FILE_PATH = "SM_LOG_FILE_PATH"
ENV_VAR_LOG_LEVEL = "SM_LOG_LEVEL"
ENV_VAR_SUMMARIZER_API_URL = "SM_SUMMARIZER_API_URL"
ENV_VAR_SUMMARIZER_API_KEY = "SM_SUMMARIZER_API_KEY"
ENV_VAR_SUMMARIZER_TEMPERATURE = "SM_SUMMARIZER_TEMPERATURE"
ENV_VAR_SUMMARIZER_SYSTEM_PROMPT = "SM_SUMMARIZER_SYSTEM_PROMPT"
DEFAULT_SUMMARIZER_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
DEFAULT_SUMMARIZER_API_KEY = ""
DEFAULT_SUMMARIZER_TEMPERATURE = 0.5
DEFAULT_SUMMARIZER_SYSTEM_PROMPT = """**Role:** Conversation Summarizer
**Task:** Read the following CHAT HISTORY between a User and an Assistant (possibly roleplaying). Create a concise, neutral, third-person summary covering the key topics, events, decisions, and emotional shifts.
**Objective:** Capture the essential information, **including key named entities (people, places, items)**, needed for the Assistant to recall the context of this conversation segment later. Use specific names mentioned. Focus on factual events and explicitly stated information/feelings. Avoid interpretation unless quoting a character's stated feeling.
**Format:** Narrative summary.

**CHAT HISTORY:**
---
{chat_history_string}
---

**Concise Summary (including key names/places/items):**"""
DEFAULT_TOKENIZER_ENCODING = "cl100k_base"
ENV_VAR_TOKENIZER_ENCODING = "SM_TOKENIZER_ENCODING"
DEFAULT_T0_ACTIVE_HISTORY_TOKEN_LIMIT = 4000
ENV_VAR_T0_ACTIVE_HISTORY_TOKEN_LIMIT = "SM_T0_ACTIVE_HISTORY_TOKEN_LIMIT"
DEFAULT_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET = 2000
ENV_VAR_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET = "SM_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET"
DEFAULT_MAX_STORED_SUMMARY_BLOCKS = 10
ENV_VAR_MAX_STORED_SUMMARY_BLOCKS = "SM_MAX_STORED_SUMMARY_BLOCKS"
DEFAULT_CHROMADB_PATH = os.path.join(DEFAULT_LOG_DIR, "session_summary_t2_db")
ENV_VAR_CHROMADB_PATH = "SM_CHROMADB_PATH"
DEFAULT_SQLITE_DB_PATH = os.path.join(
    DEFAULT_LOG_DIR, "session_memory_tier1_and_cache.db"
)
ENV_VAR_SQLITE_DB_PATH = "SM_SQLITE_DB_PATH"  # Combined DB file
DEFAULT_SUMMARY_COLLECTION_PREFIX = "sm_t2_"
ENV_VAR_SUMMARY_COLLECTION_PREFIX = "SM_SUMMARY_COLLECTION_PREFIX"
ENV_VAR_RAGQ_LLM_API_URL = "SM_RAGQ_LLM_API_URL"
ENV_VAR_RAGQ_LLM_API_KEY = "SM_RAGQ_LLM_API_KEY"
ENV_VAR_RAGQ_LLM_TEMPERATURE = "SM_RAGQ_LLM_TEMPERATURE"
ENV_VAR_RAGQ_LLM_PROMPT = "SM_RAGQ_LLM_PROMPT"
DEFAULT_RAGQ_LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
DEFAULT_RAGQ_LLM_API_KEY = ""
DEFAULT_RAGQ_LLM_TEMPERATURE = 0.3
DEFAULT_RAGQ_LLM_PROMPT = """Based on the latest user message and recent dialogue history, generate a concise search query suitable for retrieving relevant past conversation summaries or background lore documents. Focus on key entities, concepts, questions, or actions mentioned.

Recent Dialogue History:
{dialogue_context}

Latest User Message:
{latest_message}

Search Query:"""
DEFAULT_RAG_SUMMARY_RESULTS_COUNT = 3
ENV_VAR_RAG_SUMMARY_RESULTS_COUNT = "SM_RAG_SUMMARY_RESULTS_COUNT"

# --- Refinement & RAG Cache Defaults & Env Vars ---
ENV_VAR_ENABLE_RAG_CACHE = "SM_ENABLE_RAG_CACHE"  # NEW
ENV_VAR_ENABLE_STATELESS_REFINEMENT = (
    "SM_ENABLE_REFINEMENT"  # Keep old name for stateless
)
ENV_VAR_REFINER_API_URL = (
    "SM_REFINER_API_URL"  # Shared by Cache Update & Final Select & Stateless
)
ENV_VAR_REFINER_API_KEY = "SM_REFINER_API_KEY"  # Shared
ENV_VAR_REFINER_TEMPERATURE = (
    "SM_REFINER_TEMPERATURE"  # Shared (can be overridden if needed later)
)
ENV_VAR_CACHE_UPDATE_PROMPT_TEMPLATE = (
    "SM_CACHE_UPDATE_PROMPT_TEMPLATE"  # NEW Step 1 Prompt
)
ENV_VAR_FINAL_SELECT_PROMPT_TEMPLATE = (
    "SM_FINAL_SELECT_PROMPT_TEMPLATE"  # NEW Step 2 Prompt
)
ENV_VAR_STATELESS_REFINER_PROMPT_TEMPLATE = (
    "SM_REFINER_PROMPT_TEMPLATE"  # Old name for stateless prompt
)
ENV_VAR_REFINER_HISTORY_COUNT = "SM_REFINER_HISTORY_COUNT"  # Shared
ENV_VAR_STATELESS_REFINER_SKIP_THRESHOLD = (
    "SM_REFINER_SKIP_THRESHOLD"  # Only for stateless
)

DEFAULT_ENABLE_RAG_CACHE = False  # Default to disabled
DEFAULT_ENABLE_STATELESS_REFINEMENT = False  # Default stateless to disabled
DEFAULT_REFINER_API_URL = DEFAULT_RAGQ_LLM_API_URL
DEFAULT_REFINER_API_KEY = ""
DEFAULT_REFINER_TEMPERATURE = 0.3
# Default prompts are now imported from library
DEFAULT_REFINER_HISTORY_COUNT = 6
DEFAULT_STATELESS_REFINER_SKIP_THRESHOLD = 500

# --- Final LLM Call Defaults & Env Vars ---
ENV_VAR_FINAL_LLM_API_URL = "SM_FINAL_LLM_API_URL"
ENV_VAR_FINAL_LLM_API_KEY = "SM_FINAL_LLM_API_KEY"
ENV_VAR_FINAL_LLM_TEMPERATURE = "SM_FINAL_LLM_TEMPERATURE"
ENV_VAR_FINAL_LLM_TIMEOUT = "SM_FINAL_LLM_TIMEOUT"
DEFAULT_FINAL_LLM_API_URL = ""
DEFAULT_FINAL_LLM_API_KEY = ""
DEFAULT_FINAL_LLM_TEMPERATURE = 0.7
DEFAULT_FINAL_LLM_TIMEOUT = 120

# --- Other Defaults & Env Vars ---
DEFAULT_INCLUDE_ACK_TURNS = True
ENV_VAR_INCLUDE_ACK_TURNS = "SM_INCLUDE_ACK_TURNS"
DEFAULT_EMIT_STATUS_UPDATES = True
ENV_VAR_EMIT_STATUS_UPDATES = "SM_EMIT_STATUS_UPDATES"
DEFAULT_DEBUG_LOG_FINAL_PAYLOAD = False
ENV_VAR_DEBUG_LOG_FINAL_PAYLOAD = "SM_DEBUG_LOG_PAYLOAD"
DEFAULT_DEBUG_LOG_RAW_INPUT = False
ENV_VAR_DEBUG_LOG_RAW_INPUT = "SM_DEBUG_LOG_RAW_INPUT"

# --- Logger Setup ---
logger = logging.getLogger("SessionMemoryPipeV18_0Logger")  # Updated version
logging.basicConfig(level=logging.INFO)


# === SECTION 4: EMBEDDING WRAPPER ===
class ChromaDBCompatibleEmbedder:
    # (Implementation is unchanged)
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


# === SECTION 5: PIPE CLASS DEFINITION ===
class Pipe:
    version = "0.18.0"  # Version updated

    # === SECTION 5.1: VALVES DEFINITION ===
    class Valves(BaseModel):
        # Existing Valves (Log, DB Paths, Tokenizer, Summarizer, T1/T0 Limits, RAGQ, Final LLM)
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
        summary_collection_prefix: str = Field(
            default=os.getenv(
                ENV_VAR_SUMMARY_COLLECTION_PREFIX, DEFAULT_SUMMARY_COLLECTION_PREFIX
            )
        )
        tokenizer_encoding_name: str = Field(
            default=os.getenv(ENV_VAR_TOKENIZER_ENCODING, DEFAULT_TOKENIZER_ENCODING)
        )
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
        rag_summary_results_count: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_RAG_SUMMARY_RESULTS_COUNT, DEFAULT_RAG_SUMMARY_RESULTS_COUNT
                )
            )
        )
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

        # NEW: RAG Cache Enable Valve
        enable_rag_cache: bool = Field(
            default=str(
                os.getenv(ENV_VAR_ENABLE_RAG_CACHE, str(DEFAULT_ENABLE_RAG_CACHE))
            ).lower()
            == "true"
        )

        # Existing Stateless Refinement Enable Valve
        enable_stateless_refinement: bool = Field(
            default=str(
                os.getenv(
                    ENV_VAR_ENABLE_STATELESS_REFINEMENT,
                    str(DEFAULT_ENABLE_STATELESS_REFINEMENT),
                )
            ).lower()
            == "true"
        )

        # Shared Refiner Config Valves
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

        # NEW: RAG Cache Specific Prompt Template Valves
        # Uses imported default text if env var not set
        cache_update_prompt_template: str = Field(
            default=os.getenv(
                ENV_VAR_CACHE_UPDATE_PROMPT_TEMPLATE, DEFAULT_CACHE_UPDATE_PROMPT
            )
        )
        final_context_selection_prompt_template: str = Field(
            default=os.getenv(
                ENV_VAR_FINAL_SELECT_PROMPT_TEMPLATE, DEFAULT_FINAL_SELECT_PROMPT
            )
        )

        # Stateless Refiner Specific Prompt Template Valve
        # Defaults to None -> uses library default text
        stateless_refiner_prompt_template: Optional[str] = Field(
            default=os.getenv(ENV_VAR_STATELESS_REFINER_PROMPT_TEMPLATE, None)
        )
        # Stateless Refiner Specific Skip Threshold Valve
        stateless_refiner_skip_threshold: int = Field(
            default=int(
                os.getenv(
                    ENV_VAR_STATELESS_REFINER_SKIP_THRESHOLD,
                    DEFAULT_STATELESS_REFINER_SKIP_THRESHOLD,
                )
            )
        )

        # Other Valves (Keep as before)
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

        # Post Init Validation
        def model_post_init(self, __context: Any) -> None:
            # Existing checks
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
            # Refiner temp applies to both now
            if not (0.0 <= self.refiner_llm_temperature <= 2.0):
                self.refiner_llm_temperature = DEFAULT_REFINER_TEMPERATURE
                logger.warning("Reset refiner_llm_temp.")
            # History count applies to both
            if self.refiner_history_count < 0:
                self.refiner_history_count = DEFAULT_REFINER_HISTORY_COUNT
                logger.warning("Reset refiner_history_count.")
            # Stateless specific checks
            if self.stateless_refiner_skip_threshold < 0:
                self.stateless_refiner_skip_threshold = (
                    DEFAULT_STATELESS_REFINER_SKIP_THRESHOLD
                )
                logger.warning("Reset stateless_refiner_skip_threshold.")
            if (
                self.stateless_refiner_prompt_template is not None
                and not self.stateless_refiner_prompt_template.strip()
            ):
                self.stateless_refiner_prompt_template = None
                logger.warning(
                    "Reset stateless_refiner_prompt_template to None (was empty)."
                )
            # NEW Checks for RAG Cache templates
            if (
                not self.cache_update_prompt_template
                or not self.cache_update_prompt_template.strip()
            ):
                self.cache_update_prompt_template = DEFAULT_CACHE_UPDATE_PROMPT
                logger.warning(
                    "Reset cache_update_prompt_template to library default (was empty/missing)."
                )
            if (
                not self.final_context_selection_prompt_template
                or not self.final_context_selection_prompt_template.strip()
            ):
                self.final_context_selection_prompt_template = (
                    DEFAULT_FINAL_SELECT_PROMPT
                )
                logger.warning(
                    "Reset final_context_selection_prompt_template to library default (was empty/missing)."
                )

    # === SECTION 5.2: INITIALIZATION METHOD ===
    def __init__(self):
        self.type = "pipe"
        self.name = f"SESSION_MEMORY PIPE (v{self.version}) - Two-Step RAG Cache"  # Updated name
        self.logger = logger
        self.logger.info(f"Initializing '{self.name}'...")
        if not I4_LLM_AGENT_AVAILABLE:
            raise ImportError(f"i4_llm_agent failed import: {IMPORT_ERROR_DETAILS}")
        # Log availability of functions
        if (
            not CACHE_UPDATE_FUNC
            or not FINAL_CONTEXT_SELECT_FUNC
            or not INIT_RAG_CACHE_TABLE_FUNC
        ):
            self.logger.warning("RAG Cache functions unavailable. Feature disabled.")
        if not STATELESS_REFINE_FUNC:
            self.logger.warning("Stateless Refinement function unavailable.")
        if not CHROMADB_AVAILABLE:
            self.logger.error(
                f"ChromaDB unavailable. T2 disabled. Err: {CHROMADB_IMPORT_ERROR}"
            )
        if not OWI_RAG_UTILS_AVAILABLE:
            self.logger.warning(
                f"OWI RAG utils unavailable. Embeddings may fail. Err: {OWI_IMPORT_ERROR}"
            )
        if not TIKTOKEN_AVAILABLE:
            self.logger.critical("tiktoken unavailable. Token counting WILL FAIL.")

        # Load Valves
        try:
            self.valves = self.Valves()
            # Simplified logging of valve settings
            log_valves = {
                k: v
                for k, v in self.valves.model_dump().items()
                if "api_key" not in k and "prompt" not in k
            }  # Exclude prompts too now
            log_valves["cache_update_prompt_set"] = bool(
                self.valves.cache_update_prompt_template != DEFAULT_CACHE_UPDATE_PROMPT
            )
            log_valves["final_select_prompt_set"] = bool(
                self.valves.final_context_selection_prompt_template
                != DEFAULT_FINAL_SELECT_PROMPT
            )
            log_valves["stateless_prompt_set"] = bool(
                self.valves.stateless_refiner_prompt_template is not None
            )
            self.logger.info(f"Valves loaded: {log_valves}")
            # Check API keys
            if not self.valves.summarizer_api_key:
                self.logger.warning("Summarizer API Key MISSING.")
            if not self.valves.ragq_llm_api_key:
                self.logger.warning("RAG Query LLM API Key MISSING.")
            self.logger.info(
                f"RAG Cache Feature ENABLED: {self.valves.enable_rag_cache}"
            )
            self.logger.info(
                f"Stateless Refinement Feature ENABLED: {self.valves.enable_stateless_refinement}"
            )
            if (
                self.valves.enable_rag_cache or self.valves.enable_stateless_refinement
            ) and not self.valves.refiner_llm_api_key:
                self.logger.warning(
                    "A refinement feature is ENABLED but Refiner API Key MISSING!"
                )
            if not self.valves.final_llm_api_url or not self.valves.final_llm_api_key:
                self.logger.info("Final LLM Call disabled.")
            else:
                self.logger.info("Final LLM Call enabled.")
        except Exception as e:
            self.logger.error(f"CRITICAL Valve init error: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize pipe Valves") from e

        # Configure Logger
        try:
            self.configure_logger()
            self.logger.info(
                f"Logger configured. Level: {logging.getLevelName(self.logger.getEffectiveLevel())}, Path: {self.valves.log_file_path or 'Console'}"
            )
        except Exception as e:
            print(f"CRITICAL Logger config error: {e}")
            self.logger.critical(f"Logger config failed: {e}", exc_info=True)

        # Initialize Tokenizer
        self._tokenizer = None
        if TIKTOKEN_AVAILABLE and UTIL_COUNT_TOKENS_FUNC:
            try:
                self.logger.info(
                    f"Init tokenizer: '{self.valves.tokenizer_encoding_name}'..."
                )
                self._tokenizer = tiktoken.get_encoding(
                    self.valves.tokenizer_encoding_name
                )
                self.logger.info("Tokenizer initialized.")
            except Exception as e:
                self.logger.error(
                    f"Tokenizer init failed: {e}. Token counting disabled.",
                    exc_info=True,
                )
        else:
            self.logger.error("Tokenizer disabled: tiktoken or counter unavailable.")

        # Initialize SQLite
        self._sqlite_conn = None
        self._sqlite_cursor = None
        try:
            sqlite_db_path = self.valves.sqlite_db_path
            self.logger.info(f"Init SQLite: {sqlite_db_path}")
            os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
            self._sqlite_conn = sqlite3.connect(
                sqlite_db_path, check_same_thread=False, isolation_level=None
            )
            self._sqlite_conn.execute("PRAGMA journal_mode=WAL;")
            self._sqlite_cursor = self._sqlite_conn.cursor()
            # Initialize T1 Summary Table
            self._sqlite_cursor.execute(
                """CREATE TABLE IF NOT EXISTS tier1_text_summaries (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, user_id TEXT, summary_text TEXT NOT NULL,
                    timestamp_utc REAL NOT NULL, timestamp_iso TEXT, turn_start_index INTEGER, turn_end_index INTEGER,
                    char_length INTEGER, config_t0_token_limit INTEGER, config_t1_chunk_target INTEGER,
                    calculated_prompt_tokens INTEGER, t0_end_index_at_summary INTEGER
                )"""
            )
            self._sqlite_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_ts ON tier1_text_summaries (session_id, timestamp_utc)"
            )
            self.logger.info("SQLite T1 Summary table initialized.")
            # NEW: Initialize RAG Cache Table
            if INIT_RAG_CACHE_TABLE_FUNC:
                from i4_llm_agent.cache import (
                    _sync_initialize_rag_cache_table,
                )  # Import sync helper

                init_success = _sync_initialize_rag_cache_table(self._sqlite_cursor)
                if init_success:
                    self.logger.info("SQLite RAG Cache table initialized.")
                else:
                    self.logger.error("Failed to initialize SQLite RAG Cache table!")
            else:
                self.logger.error(
                    "Cannot initialize RAG Cache table: function unavailable."
                )
            self.logger.info(f"SQLite DB initialized at '{sqlite_db_path}'.")
        except Exception as e:
            self.logger.error(f"SQLite init error: {e}", exc_info=True)
            if self._sqlite_conn:
                self._sqlite_conn.close()
            self._sqlite_conn, self._sqlite_cursor = None, None

        # Initialize ChromaDB
        self._chroma_client = None
        if CHROMADB_AVAILABLE:
            try:
                chromadb_path = self.valves.chromadb_path
                self.logger.info(f"Init ChromaDB T2: {chromadb_path}")
                os.makedirs(chromadb_path, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=chromadb_path)
                self._chroma_client.heartbeat()
                self.logger.info(
                    f"ChromaDB T2 client initialized at '{chromadb_path}'."
                )
            except Exception as e:
                self._chroma_client = None
                self.logger.error(f"ChromaDB T2 init error: {e}", exc_info=True)
        else:
            self.logger.warning("ChromaDB unavailable. T2 disabled.")

        self.sessions: Dict[str, Dict] = {}
        self.logger.info("In-memory session manager initialized.")
        self._owi_embedding_func_cache: Optional[OwiEmbeddingFunction] = None
        self._current_event_emitter = None
        self.logger.info(f"'{self.name}' initialization complete.")

    # === SECTION 5.3: LOGGER CONFIGURATION ===
    def configure_logger(self):
        # (Implementation unchanged)
        log_level_str = self.valves.log_level.upper()
        log_path = self.valves.log_file_path
        try:
            numeric_level = getattr(logging, log_level_str)
            assert isinstance(numeric_level, int)
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
        # (Implementation unchanged)
        self.logger.info(f"on_startup: '{self.name}' (v{self.version}) starting...")
        if self._sqlite_conn and self._sqlite_cursor:
            try:
                self._sqlite_cursor.execute("SELECT 1")
                self.logger.info("SQLite connection verified.")
            except Exception as e:
                self.logger.error(
                    f"SQLite connection check failed on startup: {e}", exc_info=True
                )
        else:
            self.logger.warning("SQLite not available on startup.")

        if self._chroma_client:
            try:
                version = await asyncio.to_thread(self._chroma_client.heartbeat)
                self.logger.info(
                    f"ChromaDB connection verified (heartbeat: {version}ns)."
                )
            except Exception as e:
                self.logger.error(
                    f"ChromaDB connection check failed on startup: {e}", exc_info=True
                )
        else:
            self.logger.warning("ChromaDB not available on startup.")
        self.logger.info(f"'{self.name}' startup complete.")

    async def on_shutdown(self):
        # (Implementation unchanged)
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
                print(
                    f"ERROR closing log handler: {handler}: {e}"
                )  # Use print as logger might be closed
        self.logger.info(f"'{self.name}' shutdown complete.")

    # --- CORRECTED on_valves_updated ---
    async def on_valves_updated(self):
        self.logger.info(f"on_valves_updated: Reloading settings for '{self.name}'...")
        # Store old settings for comparison
        old_log_path = getattr(self.valves, "log_file_path", None)
        old_log_level = getattr(self.valves, "log_level", None)
        old_tokenizer_encoding = getattr(self.valves, "tokenizer_encoding_name", None)
        old_sqlite_path = getattr(self.valves, "sqlite_db_path", None)
        old_chromadb_path = getattr(self.valves, "chromadb_path", None)
        old_rag_cache_enabled = getattr(self.valves, "enable_rag_cache", None)
        old_stateless_refinement_enabled = getattr(
            self.valves, "enable_stateless_refinement", None
        )
        old_final_llm_url = getattr(self.valves, "final_llm_api_url", None)

        # Re-initialize valves
        try:
            self.valves = self.Valves()  # Reload valves
            # Log relevant valve settings
            log_valves = {
                k: v
                for k, v in self.valves.model_dump().items()
                if "api_key" not in k and "prompt" not in k
            }
            log_valves["cache_update_prompt_set"] = bool(
                self.valves.cache_update_prompt_template != DEFAULT_CACHE_UPDATE_PROMPT
            )
            log_valves["final_select_prompt_set"] = bool(
                self.valves.final_context_selection_prompt_template
                != DEFAULT_FINAL_SELECT_PROMPT
            )
            log_valves["stateless_prompt_set"] = bool(
                self.valves.stateless_refiner_prompt_template is not None
            )
            self.logger.info(f"Valves re-initialized: {log_valves}")

            # Log refinement status changes
            if self.valves.enable_rag_cache != old_rag_cache_enabled:
                self.logger.info(
                    f"RAG Cache Feature now ENABLED: {self.valves.enable_rag_cache}"
                )
            if (
                self.valves.enable_stateless_refinement
                != old_stateless_refinement_enabled
            ):
                self.logger.info(
                    f"Stateless Refinement Feature now ENABLED: {self.valves.enable_stateless_refinement}"
                )

            # Check refiner key if any refinement is enabled
            if (
                self.valves.enable_rag_cache or self.valves.enable_stateless_refinement
            ) and not self.valves.refiner_llm_api_key:
                self.logger.warning(
                    "A refinement feature is ENABLED but Refiner Key MISSING after update!"
                )

        except Exception as e:
            self.logger.error(
                f"Error re-initializing valves: {e}. Pipe may use old settings.",
                exc_info=True,
            )
            # Potentially return or raise here if valves are critical

        # --- Reconfigure logger if settings changed ---
        new_log_path = getattr(self.valves, "log_file_path", None)
        new_log_level = getattr(self.valves, "log_level", None)
        if old_log_path != new_log_path or old_log_level != new_log_level:
            self.logger.info("Log settings changed. Reconfiguring logger...")
            # Split the try-except block for configure_logger
            try:
                self.configure_logger()
                self.logger.info(
                    "Logger reconfigured successfully."
                )  # Now on a separate line
            except Exception as e:
                self.logger.error(
                    f"Error during logger reconfiguration: {e}", exc_info=True
                )

        # --- Clear embedding cache ---
        self.logger.info("Clearing OWI embedding function cache.")
        self._owi_embedding_func_cache = None

        # --- Re-init tokenizer if changed ---
        new_tokenizer_encoding = getattr(self.valves, "tokenizer_encoding_name", None)
        if (
            TIKTOKEN_AVAILABLE
            and UTIL_COUNT_TOKENS_FUNC
            and old_tokenizer_encoding != new_tokenizer_encoding
        ):
            self.logger.info(
                f"Tokenizer encoding changed to '{new_tokenizer_encoding}'. Re-initializing..."
            )
            self._tokenizer = None  # Clear old tokenizer first
            try:
                self._tokenizer = tiktoken.get_encoding(new_tokenizer_encoding)
                self.logger.info("Tokenizer re-initialized successfully.")
            except Exception as e:
                self.logger.error(
                    f"Failed to re-initialize tokenizer: {e}. Token counting may fail.",
                    exc_info=True,
                )
                self._tokenizer = None  # Ensure it's None on failure

        # --- Warn about DB path changes ---
        new_sqlite_path = getattr(self.valves, "sqlite_db_path", None)
        if old_sqlite_path != new_sqlite_path:
            self.logger.warning(
                f"SQLite DB path changed from '{old_sqlite_path}' to '{new_sqlite_path}'. Pipe restart might be required for changes to fully apply."
            )
        new_chromadb_path = getattr(self.valves, "chromadb_path", None)
        if old_chromadb_path != new_chromadb_path:
            self.logger.warning(
                f"ChromaDB path changed from '{old_chromadb_path}' to '{new_chromadb_path}'. Pipe restart might be required for changes to fully apply."
            )

        # --- Log final LLM URL change ---
        new_final_llm_url = getattr(self.valves, "final_llm_api_url", None)
        if old_final_llm_url != new_final_llm_url:
            self.logger.info(f"Final LLM API URL changed to: '{new_final_llm_url}'.")

        self.logger.info("on_valves_updated: Reload finished.")

    # === SECTION 5.5: HELPERS - SESSION STATE ===
    def _get_or_create_session(self, session_id: str) -> Dict:
        # (Implementation unchanged)
        if session_id not in self.sessions:
            self.logger.info(f"Creating new session state for ID: {session_id}")
            self.sessions[session_id] = {
                "active_history": [],
                "last_summary_turn_index": -1,
            }
        self.sessions[session_id].setdefault("active_history", [])
        self.sessions[session_id].setdefault("last_summary_turn_index", -1)
        return self.sessions[session_id]

    # === SECTION 5.6: HELPER - TIER 2 CHROMA DB COLLECTION ===
    def _get_or_create_tier2_collection(
        self,
        collection_name: str,
        embedding_function: Optional[ChromaEmbeddingFunction] = None,
        metadata_config: Optional[Dict] = None,
    ) -> Optional[Collection]:
        # (Implementation unchanged)
        if not self._chroma_client:
            self.logger.error("T2: Chroma client unavailable.")
            return None
        if not collection_name:
            self.logger.error("T2: Collection name required.")
            return None
        if not embedding_function:
            self.logger.warning(f"T2: No embedding function for '{collection_name}'.")
        if (
            not re.match(
                r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$", collection_name
            )
            or ".." in collection_name
        ):
            self.logger.error(f"T2: Invalid collection name '{collection_name}'.")
            return None
        try:
            self.logger.debug(f"Accessing T2 collection '{collection_name}'...")
            collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata=metadata_config,
            )
            if collection:
                self.logger.info(
                    f"T2: Accessed/created collection '{collection_name}'."
                )
                return collection
            else:
                self.logger.error(
                    f"T2: get_or_create_collection returned None/False for '{collection_name}'."
                )
                return None
        except sqlite3.OperationalError as e:
            self.logger.error(
                f"T2: SQLite error accessing Chroma '{collection_name}': {e}.",
                exc_info=True,
            )
            return None
        except InvalidDimensionException as ide:
            self.logger.error(
                f"T2: Chroma Dimension Exception for '{collection_name}': {ide}.",
                exc_info=True,
            )
            return None
        except Exception as e:
            self.logger.error(
                f"T2: Unexpected EXCEPTION accessing collection '{collection_name}': {e}",
                exc_info=True,
            )
            return None

    # === SECTION 5.7: HELPER - OWI EMBEDDING FUNCTION RETRIEVAL ===
    def _get_owi_embedding_function(
        self, __request__: Request, __user__: Optional[Dict]
    ) -> Optional[OwiEmbeddingFunction]:
        # (Implementation unchanged)
        if self._owi_embedding_func_cache:
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
                self.logger.error(
                    "Embeddings: Local engine ('') but no 'ef' object found."
                )
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

    # === SECTION 5.8: HELPERS - SQLITE TIER 1 OPERATIONS ===
    # (Implementations unchanged)
    def _sync_add_tier1_summary(
        self,
        summary_id: str,
        session_id: str,
        user_id: str,
        summary_text: str,
        metadata: Dict,
    ) -> bool:
        if not self._sqlite_cursor:
            logger.error("T1 DB: _sync_add_tier1_summary cursor unavailable.")
            return False
        try:
            self._sqlite_cursor.execute(
                """INSERT INTO tier1_text_summaries (id, session_id, user_id, summary_text, timestamp_utc, timestamp_iso, turn_start_index, turn_end_index, char_length, config_t0_token_limit, config_t1_chunk_target, calculated_prompt_tokens, t0_end_index_at_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary_id,
                    session_id,
                    user_id,
                    summary_text,
                    metadata.get("timestamp_utc"),
                    metadata.get("timestamp_iso"),
                    metadata.get("turn_start_index"),
                    metadata.get("turn_end_index"),
                    metadata.get("char_length"),
                    metadata.get("config_t0_token_limit"),
                    metadata.get("config_t1_chunk_target"),
                    metadata.get("calculated_prompt_tokens"),
                    metadata.get("t0_end_index_at_summary"),
                ),
            )
            # self.logger.debug(f"Added T1 summary {summary_id} for session {session_id}.") # Too verbose for default
            return True
        except sqlite3.IntegrityError as e:
            logger.error(
                f"T1 DB: IntegrityError adding {summary_id} (duplicate?): {e}."
            )
            return False
        except sqlite3.Error as e:
            logger.error(f"T1 DB: Error adding {summary_id}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"T1 DB: Unexpected error _sync_add_tier1_summary {summary_id}: {e}",
                exc_info=True,
            )
            return False

    async def _add_tier1_summary(
        self,
        summary_id: str,
        session_id: str,
        user_id: str,
        summary_text: str,
        metadata: Dict,
    ) -> bool:
        return await asyncio.to_thread(
            self._sync_add_tier1_summary,
            summary_id,
            session_id,
            user_id,
            summary_text,
            metadata,
        )

    def _sync_get_recent_tier1_summaries(
        self, session_id: str, limit: int
    ) -> List[str]:
        if not self._sqlite_cursor:
            logger.error("T1 DB: _sync_get_recent_tier1_summaries cursor unavailable.")
            return []
        if limit <= 0:
            return []
        try:
            self._sqlite_cursor.execute(
                "SELECT summary_text FROM tier1_text_summaries WHERE session_id = ? ORDER BY timestamp_utc DESC LIMIT ?",
                (session_id, limit),
            )
            summaries = [row[0] for row in self._sqlite_cursor.fetchall()]
            summaries.reverse()
            return summaries
        except sqlite3.Error as e:
            logger.error(
                f"T1 DB: Error getting recent T1 for {session_id}: {e}", exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"T1 DB: Unexpected error _sync_get_recent_tier1_summaries {session_id}: {e}",
                exc_info=True,
            )
            return []

    async def _get_recent_tier1_summaries(
        self, session_id: str, limit: int
    ) -> List[str]:
        return await asyncio.to_thread(
            self._sync_get_recent_tier1_summaries, session_id, limit
        )

    def _sync_get_tier1_summary_count(self, session_id: str) -> int:
        if not self._sqlite_cursor:
            logger.error("T1 DB: _sync_get_tier1_summary_count cursor unavailable.")
            return -1
        try:
            self._sqlite_cursor.execute(
                "SELECT COUNT(*) FROM tier1_text_summaries WHERE session_id = ?",
                (session_id,),
            )
            result = self._sqlite_cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(
                f"T1 DB: Error counting T1 for {session_id}: {e}", exc_info=True
            )
            return -1
        except Exception as e:
            logger.error(
                f"T1 DB: Unexpected error _sync_get_tier1_summary_count {session_id}: {e}",
                exc_info=True,
            )
            return -1

    async def _get_tier1_summary_count(self, session_id: str) -> int:
        return await asyncio.to_thread(self._sync_get_tier1_summary_count, session_id)

    def _sync_get_oldest_tier1_summary(
        self, session_id: str
    ) -> Optional[Tuple[str, str, Dict]]:
        if not self._sqlite_cursor:
            logger.error("T1 DB: _sync_get_oldest_tier1_summary cursor unavailable.")
            return None
        try:
            self._sqlite_cursor.execute(
                """SELECT id, summary_text, session_id, user_id, timestamp_utc, timestamp_iso, turn_start_index, turn_end_index, char_length, config_t0_token_limit, config_t1_chunk_target, calculated_prompt_tokens, t0_end_index_at_summary FROM tier1_text_summaries WHERE session_id = ? ORDER BY timestamp_utc ASC LIMIT 1""",
                (session_id,),
            )
            row = self._sqlite_cursor.fetchone()
            if row:
                (
                    sid,
                    s_text,
                    sess_id,
                    u_id,
                    ts_utc,
                    ts_iso,
                    s_idx,
                    e_idx,
                    length,
                    t0_lim,
                    t1_targ,
                    p_tok,
                    t0_end,
                ) = row
                metadata = {
                    "session_id": sess_id,
                    "user_id": u_id,
                    "timestamp_utc": ts_utc,
                    "timestamp_iso": ts_iso,
                    "turn_start_index": s_idx,
                    "turn_end_index": e_idx,
                    "char_length": length,
                    "config_t0_token_limit": t0_lim,
                    "config_t1_chunk_target": t1_targ,
                    "calculated_prompt_tokens": p_tok,
                    "t0_end_index_at_summary": t0_end,
                    "doc_type": "llm_summary",
                }
                return sid, s_text, metadata
            else:
                return None
        except sqlite3.Error as e:
            logger.error(
                f"T1 DB: Error getting oldest T1 for {session_id}: {e}", exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"T1 DB: Unexpected error _sync_get_oldest_tier1_summary {session_id}: {e}",
                exc_info=True,
            )
            return None

    async def _get_oldest_tier1_summary(
        self, session_id: str
    ) -> Optional[Tuple[str, str, Dict]]:
        return await asyncio.to_thread(self._sync_get_oldest_tier1_summary, session_id)

    def _sync_delete_tier1_summary(self, summary_id: str) -> bool:
        if not self._sqlite_cursor:
            logger.error("T1 DB: _sync_delete_tier1_summary cursor unavailable.")
            return False
        try:
            self._sqlite_cursor.execute(
                "DELETE FROM tier1_text_summaries WHERE id = ?", (summary_id,)
            )
            deleted_count = self._sqlite_cursor.rowcount
            return deleted_count > 0
        except sqlite3.Error as e:
            logger.error(f"T1 DB: Error deleting {summary_id}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"T1 DB: Unexpected error _sync_delete_tier1_summary {summary_id}: {e}",
                exc_info=True,
            )
            return False

    async def _delete_tier1_summary(self, summary_id: str) -> bool:
        return await asyncio.to_thread(self._sync_delete_tier1_summary, summary_id)

    # === SECTION 5.9: HELPER - ASYNC LLM CALL WRAPPER ===
    async def _async_llm_call_wrapper(
        self,
        api_url: str,
        api_key: str,
        payload: Dict[str, Any],
        temperature: float,
        timeout: int = 90,
        caller_info: str = "LLM",
    ) -> Tuple[bool, Union[str, Dict]]:
        # (Implementation unchanged)
        if not LLM_CALL_FUNC:
            logger.error(f"[{caller_info}] LLM func unavailable.")
            return False, {
                "error_type": "SetupError",
                "message": "LLM func unavailable",
            }
        if not api_url or not api_key:
            error_msg = "Missing API Key" if not api_key else "Missing URL"
            logger.error(f"[{caller_info}] {error_msg}.")
            return False, {"error_type": "ConfigurationError", "message": error_msg}
        try:
            return await asyncio.to_thread(
                LLM_CALL_FUNC,
                api_url=api_url,
                api_key=api_key,
                payload=payload,
                temperature=temperature,
                timeout=timeout,
                caller_info=caller_info,
            )
        except Exception as e:
            logger.error(f"LLM Wrapper Error [{caller_info}]: {e}", exc_info=True)
            return False, {
                "error_type": "AsyncWrapperError",
                "message": f"{type(e).__name__}",
            }

    # === SECTION 5.10: HELPER - DEBUG LOGGING (INPUT & OUTPUT) ===
    # (Implementations unchanged)
    def _get_debug_log_path(self, suffix: str) -> Optional[str]:
        try:
            base_log_path = self.valves.log_file_path or os.path.join(
                DEFAULT_LOG_DIR, DEFAULT_LOG_FILE_NAME
            )
            log_dir = os.path.dirname(base_log_path)
            os.makedirs(log_dir, exist_ok=True)
            base_name, _ = os.path.splitext(os.path.basename(base_log_path))
            debug_filename = f"{base_name}{suffix}.log"
            return os.path.join(log_dir, debug_filename)
        except Exception as e:
            logger.error(f"Failed get debug path '{suffix}': {e}")
            return None

    def _log_debug_raw_input(self, session_id: str, body: Dict):
        if not self.valves.debug_log_raw_input:
            return
        debug_log_path = self._get_debug_log_path(".DEBUG_INPUT")
        if not debug_log_path:
            logger.error(f"[{session_id}] Cannot log raw input: No path.")
            return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_entry = {
                "ts": ts,
                "pipe": self.version,
                "sid": session_id,
                "body": body,
            }
            with open(debug_log_path, "a", encoding="utf-8") as f:
                json.dump(log_entry, f)
                f.write("\\n")  # Compact JSON
        except Exception as e:
            logger.error(
                f"[{session_id}] Failed write debug raw input log: {e}", exc_info=True
            )

    def _log_debug_final_payload(self, session_id: str, output_body: Dict):
        if not self.valves.debug_log_final_payload:
            return
        debug_log_path = self._get_debug_log_path(".DEBUG_PAYLOAD")
        if not debug_log_path:
            logger.error(f"[{session_id}] Cannot log final payload: No path.")
            return
        try:
            ts = datetime.now(timezone.utc).isoformat()
            log_entry = {
                "ts": ts,
                "pipe": self.version,
                "sid": session_id,
                "payload": output_body,
            }
            with open(debug_log_path, "a", encoding="utf-8") as f:
                json.dump(log_entry, f)
                f.write("\\n")  # Compact JSON
        except Exception as e:
            logger.error(
                f"[{session_id}] Failed write debug final payload log: {e}",
                exc_info=True,
            )

    # === SECTION 5.11: MAIN PIPE METHOD ===
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
    ) -> Union[str, Generator, Iterator, Dict]:
        # //////////////////////////////////////////////////////////////////////
        # /// 5.11.0: Initialization & Setup                                ///
        # //////////////////////////////////////////////////////////////////////
        pipe_entry_time = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"pipe: Entered v{self.version} at {pipe_entry_time}")
        self._current_event_emitter = __event_emitter__
        session_id = "uninitialized_session"
        user_id = "default_user"
        session_state: Optional[Dict] = None
        owi_embed_func: Optional[OwiEmbeddingFunction] = None
        tier2_collection: Optional[Collection] = None
        output_body = body.copy() if isinstance(body, dict) else {}
        status_message = "MemProc: Initializing..."
        # Flags & Metrics
        cache_update_performed = False
        final_context_selection_performed = False
        stateless_refinement_performed = False  # Separate flags now
        summarization_performed_successfully = False
        summ_prompt_tokens = -1
        summ_output_tokens = -1
        final_payload_tokens = -1
        t0_dialogue_tokens = -1
        t1_retrieved_count = 0
        t2_retrieved_count = 0
        initial_owi_context_tokens = -1
        refined_context_tokens = -1  # Represents final selected context tokens
        t0_raw_history_slice: List[Dict] = []
        final_llm_payload_contents: Optional[List[Dict]] = None

        async def emit_status(description: str, done: bool = False):
            log_session_id = (
                session_id if session_id != "uninitialized_session" else "startup"
            )
            if (
                self.valves.emit_status_updates
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
                    # self.logger.debug(f"[{log_session_id}] Status emitted: '{description}' (Done: {done})") # Debug level now
                except Exception as e_emit:
                    self.logger.warning(
                        f"[{log_session_id}] Failed emit status: {e_emit}"
                    )
            else:
                self.logger.debug(
                    f"[{log_session_id}] Status update (not emitted): '{description}' (Done: {done})"
                )  # Debug level now

        try:
            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.1: Base Setup, Validation & Session ID                  ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation unchanged)
            self.logger.debug("PIPE_DEBUG: [1] Base Setup")
            await emit_status("MemProc: Validating input...")
            if not isinstance(body, dict):
                await emit_status("ERROR: Invalid input type.", done=True)
                return {"error": "Invalid input body type.", "status_code": 400}
            if not __request__:
                self.logger.warning("__request__ missing.")
            if isinstance(__user__, dict) and "id" in __user__:
                user_id = __user__["id"]
            else:
                self.logger.warning("User info missing. Using 'default_user'.")
            chat_id = __chat_id__
            if not chat_id or not isinstance(chat_id, str) or len(chat_id.strip()) == 0:
                await emit_status(
                    "ERROR: Cannot isolate session (missing chat_id).", True
                )
                return {"error": "Pipe needs valid chat_id.", "status_code": 500}
            safe_chat_id_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", chat_id)
            session_id = f"user_{user_id}_chat_{safe_chat_id_part}"
            self.logger.info(f"Session ID: {session_id}")
            session_state = self._get_or_create_session(session_id)
            if not isinstance(session_state, dict):
                await emit_status(f"ERROR: Internal session state error.", done=True)
                return {"error": f"Internal session state error.", "status_code": 500}
            self._log_debug_raw_input(session_id, body)
            if not I4_LLM_AGENT_AVAILABLE:
                await emit_status("ERROR: Core library missing.", done=True)
                return {"error": "Core library unavailable.", "status_code": 500}
            if not self._tokenizer or not UTIL_COUNT_TOKENS_FUNC:
                self.logger.error(f"[{session_id}] Tokenizer/counter unavailable.")
            if not self._sqlite_conn or not self._sqlite_cursor:
                self.logger.error(f"[{session_id}] SQLite unavailable.")
            incoming_messages = body.get("messages", [])
            if not isinstance(incoming_messages, list):
                incoming_messages = []
            latest_user_message = next(
                (
                    msg
                    for msg in reversed(incoming_messages)
                    if isinstance(msg, dict) and msg.get("role") == "user"
                ),
                None,
            )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.2: Tier 2 Setup (Chroma/Emb)                              ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation unchanged)
            self.logger.debug(f"[{session_id}] PIPE_DEBUG: [2] Tier 2 Setup")
            await emit_status("MemProc: Setting up vector store...")
            chroma_embed_wrapper = None
            tier2_collection = None
            if self._chroma_client and __request__:
                owi_embed_func = self._get_owi_embedding_function(__request__, __user__)
                if owi_embed_func:
                    try:
                        chroma_embed_wrapper = ChromaDBCompatibleEmbedder(
                            owi_embed_func, __user__
                        )
                    except Exception as wrapper_e:
                        self.logger.error(
                            f"[{session_id}] Embedding wrapper creation failed: {wrapper_e}.",
                            exc_info=True,
                        )
                    if chroma_embed_wrapper and self._chroma_client:
                        base_prefix = self.valves.summary_collection_prefix
                        safe_session_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id)[
                            :50
                        ]
                        tier2_collection_name = f"{base_prefix}{safe_session_part}"[:63]
                        tier2_collection = await asyncio.to_thread(
                            self._get_or_create_tier2_collection,
                            tier2_collection_name,
                            chroma_embed_wrapper,
                        )
                        if tier2_collection:
                            self.logger.info(
                                f"[{session_id}] T2 collection '{tier2_collection_name}' ready."
                            )
                        else:
                            self.logger.error(
                                f"[{session_id}] Failed get/create T2 collection '{tier2_collection_name}'."
                            )
                else:
                    self.logger.error(
                        f"[{session_id}] Failed retrieve OWI embedding func. T2 disabled."
                    )
            elif not self._chroma_client:
                self.logger.warning(
                    f"[{session_id}] Skipping T2 setup: Chroma unavailable."
                )
            elif not __request__:
                self.logger.warning(
                    f"[{session_id}] Skipping T2 setup: Request context missing."
                )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.3: Update Active History in Session State                 ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation unchanged)
            self.logger.debug(f"[{session_id}] PIPE_DEBUG: [3] Update Active History")
            await emit_status("MemProc: Updating history...")
            current_active_history = session_state.setdefault("active_history", [])
            current_len = len(current_active_history)
            new_messages_appended_count = 0
            if len(incoming_messages) > current_len:
                num_to_add = len(incoming_messages) - current_len
                # self.logger.debug(f"[{session_id}] Appending {num_to_add} new messages.")
                for i in range(current_len, len(incoming_messages)):
                    msg_to_add = incoming_messages[i]
                    if (
                        isinstance(msg_to_add, dict)
                        and "role" in msg_to_add
                        and "content" in msg_to_add
                    ):
                        current_active_history.append(msg_to_add)
                        new_messages_appended_count += 1
                    else:
                        self.logger.warning(
                            f"[{session_id}] Skipping invalid message at index {i}."
                        )
                if new_messages_appended_count > 0:
                    self.logger.info(
                        f"[{session_id}] Appended {new_messages_appended_count}. New history length: {len(current_active_history)}"
                    )
            elif len(incoming_messages) < current_len:
                self.logger.warning(
                    f"[{session_id}] Incoming history shorter. Overwriting & resetting summary index."
                )
                session_state["active_history"] = incoming_messages.copy()
                current_active_history = session_state["active_history"]
                session_state["last_summary_turn_index"] = -1
            else:
                self.logger.debug(
                    f"[{session_id}] History lengths match ({current_len})."
                )
            latest_user_message = next(
                (
                    msg
                    for msg in reversed(current_active_history)
                    if isinstance(msg, dict) and msg.get("role") == "user"
                ),
                None,
            )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.4: Tier 1 Summarization Check & Execution               ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation unchanged)
            self.logger.debug(
                f"[{session_id}] PIPE_DEBUG: [4] Tier 1 Summarization Check"
            )
            await emit_status("MemProc: Checking summarization...")
            summarization_performed_successfully = False
            new_t1_summary_text = None
            summarization_prompt_tokens = -1
            summarization_output_tokens = -1
            can_summarize = all(
                [
                    MEMORY_MANAGE_FUNC,
                    self._tokenizer,
                    UTIL_COUNT_TOKENS_FUNC,
                    self._sqlite_conn,
                    self._async_llm_call_wrapper,
                    self._add_tier1_summary,
                    self.valves.summarizer_api_url,
                    self.valves.summarizer_api_key,
                    current_active_history,
                ]
            )
            if can_summarize:
                summarizer_llm_config = {
                    "url": self.valves.summarizer_api_url,
                    "key": self.valves.summarizer_api_key,
                    "temp": self.valves.summarizer_temperature,
                    "sys_prompt": self.valves.summarizer_system_prompt,
                }
                try:
                    (
                        summarization_performed,
                        generated_summary,
                        _,
                        prompt_tokens,
                        t0_end_idx,
                    ) = await MEMORY_MANAGE_FUNC(
                        session_state=session_state,
                        active_history=current_active_history,
                        t0_token_limit=self.valves.t0_active_history_token_limit,
                        t1_chunk_size_target=self.valves.t1_summarization_chunk_token_target,
                        tokenizer=self._tokenizer,
                        llm_call_func=self._async_llm_call_wrapper,
                        llm_config=summarizer_llm_config,
                        add_t1_summary_func=self._add_tier1_summary,
                        session_id=session_id,
                        user_id=user_id,
                        dialogue_only_roles=DIALOGUE_ROLES_LIST,
                    )
                    if summarization_performed:
                        summarization_performed_successfully = True
                        new_t1_summary_text = generated_summary
                        summarization_prompt_tokens = prompt_tokens
                        if (
                            new_t1_summary_text
                            and UTIL_COUNT_TOKENS_FUNC
                            and self._tokenizer
                        ):
                            try:
                                summarization_output_tokens = UTIL_COUNT_TOKENS_FUNC(
                                    new_t1_summary_text, self._tokenizer
                                )
                            except Exception as e_tok_out:
                                summarization_output_tokens = -1
                        self.logger.info(
                            f"[{session_id}] T1 summary generated/saved. NewIdx: {session_state.get('last_summary_turn_index', 'N/A')}. SumIN: {summarization_prompt_tokens}. SumOUT: {summarization_output_tokens}. TrigIdx: {t0_end_idx}."
                        )
                        await emit_status("MemProc: Summary generated.", done=False)
                    else:
                        self.logger.debug(f"[{session_id}] T1 criteria not met.")
                except Exception as e_manage:
                    self.logger.error(
                        f"[{session_id}] EXCEPTION during T1 manage call: {e_manage}",
                        exc_info=True,
                    )
                    summarization_performed_successfully = False
            else:
                self.logger.warning(
                    f"[{session_id}] Skipping T1 check: Missing prereqs."
                )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.5: Tier 1 -> T2 Transition Check                          ///
            # //////////////////////////////////////////////////////////////////////
            self.logger.debug(
                f"[{session_id}] PIPE_DEBUG: [5] T1 -> T2 Transition Check"
            )
            await emit_status("MemProc: Checking long-term memory capacity...")

            # Check prerequisites for transition
            can_transition = all(
                [
                    summarization_performed_successfully,
                    tier2_collection,
                    chroma_embed_wrapper,
                    self._sqlite_conn is not None,
                    self.valves.max_stored_summary_blocks > 0,
                ]
            )

            if can_transition:
                max_t1_blocks = self.valves.max_stored_summary_blocks
                current_tier1_count = await self._get_tier1_summary_count(session_id)

                if current_tier1_count == -1:
                    self.logger.error(
                        f"[{session_id}] Failed get T1 count. Skipping T1->T2 check."
                    )
                elif current_tier1_count > max_t1_blocks:
                    self.logger.info(
                        f"[{session_id}] T1 limit ({max_t1_blocks}) exceeded ({current_tier1_count}). Transitioning T1->T2..."
                    )
                    await emit_status("MemProc: Archiving oldest summary...")

                    oldest_summary_data = await self._get_oldest_tier1_summary(
                        session_id
                    )

                    if oldest_summary_data:
                        oldest_id, oldest_text, oldest_metadata = oldest_summary_data
                        self.logger.debug(
                            f"[{session_id}] Retrieved oldest T1 ({oldest_id}) for transition."
                        )

                        embedding_vector = None
                        embedding_successful = False  # Flag to track success

                        try:
                            # Embed the oldest summary text
                            embedding_list = await asyncio.to_thread(
                                chroma_embed_wrapper, [oldest_text]
                            )

                            # --- Simplified Validation Logic ---
                            is_valid_structure = False
                            if (
                                isinstance(embedding_list, list)
                                and len(embedding_list) == 1
                            ):
                                first_item = embedding_list[0]
                                if isinstance(first_item, list) and len(first_item) > 0:
                                    # Structure is valid [[float, float, ...]]
                                    is_valid_structure = True
                                    # Assign on a separate line
                                    embedding_vector = first_item

                            if is_valid_structure and embedding_vector:
                                embedding_successful = True
                                self.logger.debug(
                                    f"[{session_id}] Embedded T1 {oldest_id} OK (Dim: {len(embedding_vector)})."
                                )
                            else:
                                self.logger.error(
                                    f"[{session_id}] Embedding T1 {oldest_id} returned invalid structure: {embedding_list}"
                                )
                                embedding_successful = False  # Ensure flag is False

                        except Exception as embed_e:
                            self.logger.error(
                                f"[{session_id}] EXCEPTION embedding T1->T2 {oldest_id}: {embed_e}",
                                exc_info=True,
                            )
                            embedding_successful = (
                                False  # Ensure flag is False on exception
                            )

                        # Proceed only if embedding was successful
                        if embedding_successful and embedding_vector:
                            chroma_metadata = oldest_metadata.copy()
                            chroma_metadata["transitioned_from_t1"] = True
                            chroma_metadata["original_t1_id"] = oldest_id
                            sanitized_chroma_metadata = {
                                k: (
                                    v
                                    if isinstance(v, (str, int, float, bool))
                                    else str(v)
                                )
                                for k, v in chroma_metadata.items()
                                if v is not None
                            }
                            tier2_id = f"t2_{oldest_id}"
                            self.logger.info(
                                f"[{session_id}] Adding summary {tier2_id} to T2 collection '{tier2_collection.name}'..."
                            )
                            try:
                                await asyncio.to_thread(
                                    tier2_collection.add,
                                    ids=[tier2_id],
                                    embeddings=[embedding_vector],
                                    metadatas=[sanitized_chroma_metadata],
                                    documents=[oldest_text],
                                )
                                self.logger.info(
                                    f"[{session_id}] Added {tier2_id} to T2. Deleting original T1..."
                                )
                                deleted_from_t1 = await self._delete_tier1_summary(
                                    oldest_id
                                )
                                if deleted_from_t1:
                                    await emit_status(
                                        "MemProc: Summary archive complete.", done=False
                                    )
                                else:
                                    self.logger.warning(
                                        f"[{session_id}] Added {tier2_id} to T2, but FAILED delete T1 {oldest_id}."
                                    )
                                    await emit_status(
                                        "WARN: Failed remove T1 summary.", done=False
                                    )
                            except InvalidDimensionException as ide_add:
                                self.logger.error(
                                    f"[{session_id}] Chroma DIMENSION ERROR adding {tier2_id}: {ide_add}.",
                                    exc_info=True,
                                )
                                await emit_status(
                                    "ERROR: Archive failed (dimension).", done=False
                                )
                            except Exception as add_t2_e:
                                self.logger.error(
                                    f"[{session_id}] Error adding {tier2_id} to T2: {add_t2_e}",
                                    exc_info=True,
                                )
                                await emit_status("ERROR: Failed archive.", done=False)
                        else:
                            self.logger.error(
                                f"[{session_id}] Skipping T2 add/T1 delete for {oldest_id}: embedding failed or vector invalid."
                            )
                            # Optionally emit status here if needed

                    else:
                        self.logger.warning(
                            f"[{session_id}] T1 count ({current_tier1_count}) exceeded limit ({max_t1_blocks}), but couldn't retrieve oldest summary."
                        )
                else:
                    # T1 count is within the limit
                    self.logger.debug(
                        f"[{session_id}] T1 count ({current_tier1_count}) within limit ({max_t1_blocks}). No transition needed."
                    )
            else:
                # Log why transition was skipped if prerequisites weren't met
                self.logger.debug(
                    f"[{session_id}] Skipping T1->T2 transition check: Prerequisites not met."
                )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.6: Tier 2 RAG Lookup                                      ///
            # //////////////////////////////////////////////////////////////////////
            self.logger.debug(f"[{session_id}] PIPE_DEBUG: [6] Tier 2 RAG Lookup")
            await emit_status("MemProc: Searching long-term memory...")
            retrieved_rag_summaries: List[str] = []
            t2_retrieved_count = 0

            # Check prerequisites (unchanged)
            can_rag = all(
                [
                    tier2_collection,
                    latest_user_message,
                    owi_embed_func,
                    PROMPT_RAGQ_GEN_FUNC,
                    self._async_llm_call_wrapper,
                    self.valves.ragq_llm_api_url,
                    self.valves.ragq_llm_api_key,
                    self.valves.ragq_llm_prompt,
                    self.valves.rag_summary_results_count > 0,
                ]
            )

            if not can_rag:
                self.logger.info(
                    f"[{session_id}] Skipping T2 RAG: Prerequisites not met."
                )
            else:
                # Check if collection is empty (unchanged)
                try:
                    t2_doc_count = await asyncio.to_thread(tier2_collection.count)
                    can_rag = t2_doc_count > 0
                except Exception as e_count:
                    self.logger.error(
                        f"[{session_id}] Error getting T2 count for '{tier2_collection.name}': {e_count}. Skipping RAG.",
                        exc_info=True,
                    )
                    can_rag = False
                if not can_rag:
                    self.logger.info(
                        f"[{session_id}] Skipping T2 RAG: Collection '{tier2_collection.name}' is empty."
                    )

            # Proceed with RAG if checks passed
            if can_rag:
                rag_query = None
                query_embedding = None  # Initialize embedding variable
                query_embedding_successful = False  # Flag for success

                try:
                    # --- Generate RAG Query (unchanged) ---
                    await emit_status("MemProc: Generating search query...")
                    self.logger.debug(f"[{session_id}] Generating RAG query for T2...")
                    context_messages_for_ragq = HISTORY_GET_RECENT_FUNC(
                        current_active_history,
                        count=6,
                        exclude_last=True,
                        roles=DIALOGUE_ROLES_LIST,
                    )
                    dialogue_context_str = (
                        HISTORY_FORMAT_FUNC(context_messages_for_ragq)
                        if context_messages_for_ragq and HISTORY_FORMAT_FUNC
                        else "[No recent history available]"
                    )
                    latest_message_str = (
                        latest_user_message.get("content", "")
                        if latest_user_message
                        else "[No user message]"
                    )
                    ragq_llm_config = {
                        "url": self.valves.ragq_llm_api_url,
                        "key": self.valves.ragq_llm_api_key,
                        "temp": self.valves.ragq_llm_temperature,
                        "prompt": self.valves.ragq_llm_prompt,
                    }
                    rag_query_result = await PROMPT_RAGQ_GEN_FUNC(
                        latest_message_str=latest_message_str,
                        dialogue_context_str=dialogue_context_str,
                        llm_call_func=self._async_llm_call_wrapper,
                        llm_config=ragq_llm_config,
                        caller_info=f"SMP_RAGQ_{session_id}",
                    )
                    if (
                        rag_query_result
                        and isinstance(rag_query_result, str)
                        and not rag_query_result.startswith("[Error:")
                        and rag_query_result.strip()
                    ):
                        rag_query = rag_query_result.strip()
                        self.logger.info(
                            f"[{session_id}] Generated RAG query: '{rag_query}'"
                        )
                    else:
                        self.logger.error(
                            f"[{session_id}] RAG Query Generation failed: {rag_query_result}. Skipping RAG lookup."
                        )
                        rag_query = None  # Ensure it's None on failure

                    # --- Embed RAG Query (if generated successfully) ---
                    if rag_query:
                        await emit_status("MemProc: Embedding search query...")
                        self.logger.debug(
                            f"[{session_id}] Embedding generated RAG query: '{rag_query}'"
                        )
                        try:
                            query_embedding_list = await asyncio.to_thread(
                                owi_embed_func,
                                [rag_query],
                                prefix=RAG_EMBEDDING_QUERY_PREFIX,
                                user=__user__,
                            )

                            # --- Simplified Validation Logic ---
                            is_valid_structure = False
                            if (
                                isinstance(query_embedding_list, list)
                                and len(query_embedding_list) == 1
                            ):
                                first_item = query_embedding_list[0]
                                if isinstance(first_item, list) and len(first_item) > 0:
                                    is_valid_structure = True
                                    # Assign on separate line
                                    query_embedding = first_item

                            if is_valid_structure and query_embedding:
                                query_embedding_successful = True
                                self.logger.debug(
                                    f"[{session_id}] Successfully embedded RAG query. Dimension: {len(query_embedding)}"
                                )
                            else:
                                self.logger.error(
                                    f"[{session_id}] RAG query embedding returned invalid structure: {query_embedding_list}. Skipping RAG query."
                                )
                                query_embedding_successful = False

                        except Exception as embed_e:
                            self.logger.error(
                                f"[{session_id}] EXCEPTION during RAG query embedding: {embed_e}",
                                exc_info=True,
                            )
                            query_embedding_successful = False

                    # --- Query ChromaDB (if embedding was successful) ---
                    if query_embedding_successful and query_embedding:
                        n_results = self.valves.rag_summary_results_count
                        self.logger.debug(
                            f"[{session_id}] Querying T2 collection '{tier2_collection.name}' with embedded query for {n_results} results..."
                        )
                        await emit_status(
                            f"MemProc: Searching vector store (top {n_results})..."
                        )
                        try:
                            rag_results_dict = await asyncio.to_thread(
                                tier2_collection.query,
                                query_embeddings=[query_embedding],
                                n_results=n_results,
                                include=["documents", "distances", "metadatas"],
                            )
                            # Process results (unchanged logic)
                            if (
                                rag_results_dict
                                and isinstance(rag_results_dict.get("documents"), list)
                                and rag_results_dict["documents"]
                                and isinstance(rag_results_dict["documents"][0], list)
                            ):
                                retrieved_docs = rag_results_dict["documents"][0]
                                if retrieved_docs:
                                    retrieved_rag_summaries = retrieved_docs
                                    t2_retrieved_count = len(retrieved_docs)
                                    distances = rag_results_dict.get(
                                        "distances", [[None]]
                                    )[0]
                                    ids = rag_results_dict.get("ids", [["N/A"]])[0]
                                    dist_str = [
                                        f"{d:.4f}" for d in distances if d is not None
                                    ]
                                    self.logger.info(
                                        f"[{session_id}] Retrieved {t2_retrieved_count} docs from T2 RAG. IDs: {ids}, Distances: {dist_str}"
                                    )
                                else:
                                    self.logger.info(
                                        f"[{session_id}] T2 RAG query executed successfully but returned no documents."
                                    )
                            else:
                                self.logger.info(
                                    f"[{session_id}] T2 RAG query returned no matches or unexpected result structure: {rag_results_dict}"
                                )
                        except InvalidDimensionException as ide_query:
                            self.logger.error(
                                f"[{session_id}] T2 ChromaDB DIMENSION ERROR during query: {ide_query}. Ensure query embedding dimension matches collection.",
                                exc_info=True,
                            )
                        except Exception as e_query:
                            self.logger.error(
                                f"[{session_id}] T2 ChromaDB query EXCEPTION: {e_query}",
                                exc_info=True,
                            )
                    # --- End of ChromaDB Query block ---
                    elif rag_query:  # Log if embedding failed but query existed
                        self.logger.error(
                            f"[{session_id}] Skipping T2 ChromaDB query because query embedding failed."
                        )

                except Exception as e_rag_outer:
                    self.logger.error(
                        f"[{session_id}] Unexpected error during outer T2 RAG processing block: {e_rag_outer}",
                        exc_info=True,
                    )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.7: Prepare Final Payload Inputs & Context Refinement      ///
            # //////////////////////////////////////////////////////////////////////
            self.logger.debug(
                f"[{session_id}] PIPE_DEBUG: [7] Prepare Context & Refinement"
            )
            await emit_status("MemProc: Preparing context...")

            # --- 7a: Retrieve T1 Summaries ---
            recent_t1_summaries = []
            t1_retrieved_count = 0
            if self._sqlite_conn and self.valves.max_stored_summary_blocks > 0:
                try:
                    recent_t1_summaries = await self._get_recent_tier1_summaries(
                        session_id, self.valves.max_stored_summary_blocks
                    )
                    t1_retrieved_count = len(recent_t1_summaries)
                except Exception as e_get_t1:
                    self.logger.error(
                        f"[{session_id}] Error retrieving T1: {e_get_t1}", exc_info=True
                    )
            if t1_retrieved_count > 0:
                self.logger.info(
                    f"[{session_id}] Retrieved {t1_retrieved_count} T1 summaries."
                )

            # --- 7b: Process System Prompt & Extract Initial OWI Context ---
            base_system_prompt_text = "You are helpful."
            extracted_owi_context = None
            initial_owi_context_tokens = -1
            current_output_messages = output_body.get("messages", [])
            if PROMPT_PROCESS_SYSTEM_PROMPT_FUNC:
                try:
                    base_system_prompt_text, extracted_owi_context = (
                        PROMPT_PROCESS_SYSTEM_PROMPT_FUNC(current_output_messages)
                    )
                    # self.logger.debug(f"[{session_id}] Processed system prompt. Base: {len(base_system_prompt_text)}, OWI: {len(extracted_owi_context or '')}")
                    if (
                        extracted_owi_context
                        and UTIL_COUNT_TOKENS_FUNC
                        and self._tokenizer
                    ):
                        try:
                            initial_owi_context_tokens = UTIL_COUNT_TOKENS_FUNC(
                                extracted_owi_context, self._tokenizer
                            )
                            self.logger.debug(
                                f"[{session_id}] OWI_IN tokens: {initial_owi_context_tokens}"
                            )
                        except Exception as e_tok_owi_in:
                            initial_owi_context_tokens = -1
                    elif not extracted_owi_context:
                        self.logger.debug(f"[{session_id}] No OWI <context> tag found.")
                    if not base_system_prompt_text:
                        base_system_prompt_text = "You are helpful."
                        self.logger.warning(
                            f"[{session_id}] System prompt empty after clean. Using default."
                        )
                except Exception as e_proc_sys:
                    self.logger.error(
                        f"[{session_id}] Error process_system_prompt: {e_proc_sys}.",
                        exc_info=True,
                    )
                    base_system_prompt_text = "You are helpful."
                    extracted_owi_context = None
            else:
                self.logger.error(f"[{session_id}] process_system_prompt unavailable.")

            # --- 7c: Context Refinement (RAG Cache OR Stateless OR None) ---
            context_for_prompt = extracted_owi_context  # Start with extracted context
            refined_context_tokens = -1  # Reset refined token count
            cache_update_performed = False
            final_context_selection_performed = False
            stateless_refinement_performed = False  # Reset flags
            latest_user_query_str = (
                latest_user_message.get("content", "") if latest_user_message else ""
            )

            # --- Two-Step RAG Cache Refinement ---
            if (
                self.valves.enable_rag_cache
                and CACHE_UPDATE_FUNC
                and FINAL_CONTEXT_SELECT_FUNC
                and self._sqlite_cursor
            ):
                self.logger.info(
                    f"[{session_id}] RAG Cache Feature ENABLED. Running two-step refinement..."
                )
                cache_update_llm_config = {
                    "url": self.valves.refiner_llm_api_url,
                    "key": self.valves.refiner_llm_api_key,
                    "temp": self.valves.refiner_llm_temperature,  # Using shared temp for now
                    "prompt_template": self.valves.cache_update_prompt_template,
                }
                final_select_llm_config = {
                    "url": self.valves.refiner_llm_api_url,
                    "key": self.valves.refiner_llm_api_key,
                    "temp": self.valves.refiner_llm_temperature,  # Using shared temp for now
                    "prompt_template": self.valves.final_context_selection_prompt_template,
                }

                # Check necessary configs
                if not (
                    cache_update_llm_config["url"]
                    and cache_update_llm_config["key"]
                    and final_select_llm_config["url"]
                    and final_select_llm_config["key"]
                    and cache_update_llm_config["prompt_template"]
                    and final_select_llm_config["prompt_template"]
                ):
                    self.logger.error(
                        f"[{session_id}] Skipping RAG Cache: Refiner URL/Key/Prompts missing."
                    )
                    await emit_status(
                        "ERROR: RAG Cache Refiner config incomplete.", done=False
                    )
                    # Fallback: context_for_prompt remains original extracted_owi_context
                elif not latest_user_query_str:
                    self.logger.warning(
                        f"[{session_id}] Skipping RAG Cache: Latest user query empty."
                    )
                    await emit_status(
                        "WARN: Skipping RAG Cache (no query).", done=False
                    )
                    # Fallback: context_for_prompt remains original extracted_owi_context
                else:
                    # --- Execute Step 1: Update Cache ---
                    await emit_status("MemProc: Updating background cache...")
                    updated_cache_text = await CACHE_UPDATE_FUNC(
                        session_id=session_id,
                        current_owi_context=extracted_owi_context,
                        history_messages=current_active_history,
                        latest_user_query=latest_user_query_str,
                        llm_call_func=self._async_llm_call_wrapper,
                        sqlite_cursor=self._sqlite_cursor,
                        cache_update_llm_config=cache_update_llm_config,
                        history_count=self.valves.refiner_history_count,  # Use shared history count
                        dialogue_only_roles=DIALOGUE_ROLES_LIST,
                        caller_info=f"SMP_CacheUpdate_{session_id}",
                    )
                    cache_update_performed = True  # Mark step 1 as attempted/run

                    # --- Execute Step 2: Select Final Context ---
                    await emit_status("MemProc: Selecting relevant context...")
                    final_selected_context = await FINAL_CONTEXT_SELECT_FUNC(
                        updated_cache_text=updated_cache_text,  # Use result from step 1
                        current_owi_context=extracted_owi_context,  # Also pass current OWI
                        history_messages=current_active_history,
                        latest_user_query=latest_user_query_str,
                        llm_call_func=self._async_llm_call_wrapper,
                        context_selection_llm_config=final_select_llm_config,
                        history_count=self.valves.refiner_history_count,  # Use shared history count
                        dialogue_only_roles=DIALOGUE_ROLES_LIST,
                        caller_info=f"SMP_CtxSelect_{session_id}",
                    )
                    final_context_selection_performed = (
                        True  # Mark step 2 as attempted/run
                    )

                    # Use the output of Step 2 as the context for the final prompt
                    context_for_prompt = final_selected_context
                    self.logger.info(
                        f"[{session_id}] Two-Step RAG Cache refinement complete. Using selected context (length: {len(context_for_prompt)})."
                    )

                    # Calculate tokens for the final selected context
                    if UTIL_COUNT_TOKENS_FUNC and self._tokenizer:
                        try:
                            refined_context_tokens = UTIL_COUNT_TOKENS_FUNC(
                                context_for_prompt, self._tokenizer
                            )
                            self.logger.debug(
                                f"[{session_id}] Refined context tokens (FinalSelectOUT): {refined_context_tokens}"
                            )
                        except Exception as e_tok_cache:
                            refined_context_tokens = -1
                            self.logger.error(
                                f"[{session_id}] Error calculating final selected tokens: {e_tok_cache}"
                            )
                    await emit_status(
                        "MemProc: Context selection complete.", done=False
                    )

            # --- ELSE IF: Stateless Refinement (Only if RAG Cache is disabled) ---
            elif self.valves.enable_stateless_refinement and STATELESS_REFINE_FUNC:
                self.logger.info(
                    f"[{session_id}] Stateless Refinement ENABLED (RAG Cache disabled). Proceeding..."
                )
                await emit_status("MemProc: Refining OWI context (stateless)...")
                if not extracted_owi_context:
                    self.logger.debug(
                        f"[{session_id}] Skipping stateless: No OWI context."
                    )
                elif not latest_user_query_str:
                    self.logger.warning(
                        f"[{session_id}] Skipping stateless: Query empty."
                    )
                else:
                    stateless_refiner_config = {
                        "url": self.valves.refiner_llm_api_url,
                        "key": self.valves.refiner_llm_api_key,
                        "temp": self.valves.refiner_llm_temperature,
                        "prompt_template": self.valves.stateless_refiner_prompt_template,  # Use specific valve
                    }
                    if (
                        not stateless_refiner_config["url"]
                        or not stateless_refiner_config["key"]
                    ):
                        self.logger.error(
                            f"[{session_id}] Skipping stateless: Refiner URL/Key missing."
                        )
                        await emit_status(
                            "ERROR: Stateless Refiner config incomplete.", done=False
                        )
                    else:
                        try:
                            refined_stateless_context = await STATELESS_REFINE_FUNC(
                                external_context=extracted_owi_context,
                                history_messages=current_active_history,
                                latest_user_query=latest_user_query_str,
                                llm_call_func=self._async_llm_call_wrapper,
                                refiner_llm_config=stateless_refiner_config,
                                skip_threshold=self.valves.stateless_refiner_skip_threshold,
                                history_count=self.valves.refiner_history_count,
                                dialogue_only_roles=DIALOGUE_ROLES_LIST,
                                caller_info=f"SMP_StatelessRef_{session_id}",
                            )
                            if refined_stateless_context != extracted_owi_context:
                                context_for_prompt = (
                                    refined_stateless_context  # Update context
                                )
                                stateless_refinement_performed = (
                                    True  # Mark step as run
                                )
                                self.logger.info(
                                    f"[{session_id}] Stateless refinement successful. Length: {len(context_for_prompt)}"
                                )
                                # Calculate tokens
                                if UTIL_COUNT_TOKENS_FUNC and self._tokenizer:
                                    try:
                                        refined_context_tokens = UTIL_COUNT_TOKENS_FUNC(
                                            context_for_prompt, self._tokenizer
                                        )
                                        self.logger.debug(
                                            f"[{session_id}] Stateless refined tokens (StatelessOUT): {refined_context_tokens}"
                                        )
                                    except Exception as e_tok_stateless:
                                        refined_context_tokens = -1
                                await emit_status(
                                    "MemProc: OWI context refined (stateless).",
                                    done=False,
                                )
                            else:
                                self.logger.info(
                                    f"[{session_id}] Stateless refinement no change/skipped."
                                )
                                refined_context_tokens = initial_owi_context_tokens
                        except Exception as e_refine_stateless:
                            self.logger.error(
                                f"[{session_id}] EXCEPTION stateless refinement: {e_refine_stateless}",
                                exc_info=True,
                            )
                            await emit_status(
                                "ERROR: Stateless refinement failed.", done=False
                            )

            # --- Log if no refinement performed ---
            if (
                not cache_update_performed
                and not final_context_selection_performed
                and not stateless_refinement_performed
            ):
                self.logger.debug(
                    f"[{session_id}] No refinement performed. Using original OWI context."
                )
                refined_context_tokens = -1  # Indicate no refinement value

            # --- 7d: Select T0 Dialogue History Slice ---
            # (Implementation unchanged)
            t0_raw_history_slice = []
            current_active_history_for_t0 = session_state.get("active_history", [])
            last_summary_idx = session_state.get("last_summary_turn_index", -1)
            start_index_for_t0 = last_summary_idx + 1
            if len(current_active_history_for_t0) > start_index_for_t0:
                history_to_consider = current_active_history_for_t0[start_index_for_t0:]
                end_slice_idx = (
                    -1
                    if (
                        latest_user_message
                        and history_to_consider
                        and history_to_consider[-1] == latest_user_message
                    )
                    else None
                )
                t0_raw_history_slice = [
                    msg
                    for msg in history_to_consider[:end_slice_idx]
                    if msg.get("role") in DIALOGUE_ROLES_LIST
                ]
                self.logger.info(
                    f"[{session_id}] Selected T0 history: {len(t0_raw_history_slice)} msgs (from idx {start_index_for_t0})."
                )
            else:
                self.logger.info(
                    f"[{session_id}] No T0 history after last summary idx ({last_summary_idx})."
                )

            # --- 7e: Combine Context Sources ---
            # (Implementation unchanged, now uses context_for_prompt)
            combined_context_string = "[Error combining context]"
            if PROMPT_CONTEXT_COMBINE_FUNC:
                try:
                    combined_context_string = PROMPT_CONTEXT_COMBINE_FUNC(
                        final_selected_context=context_for_prompt,
                        t1_summaries=recent_t1_summaries,
                        t2_rag_results=retrieved_rag_summaries,
                    )
                except Exception as e_combine:
                    self.logger.error(
                        f"[{session_id}] Error combining context: {e_combine}",
                        exc_info=True,
                    )
            else:
                self.logger.error(
                    f"[{session_id}] Cannot combine context: Function unavailable."
                )
            self.logger.debug(
                f"[{session_id}] Combined background context length: {len(combined_context_string)}."
            )

            # --- 7f: Get Latest User Query Content (already have latest_user_query_str) ---

            # --- 7g: Prepare Final System Prompt ---
            # (Implementation unchanged)
            memory_guidance = "\\n\\n--- Memory Guidance ---\\nUse dialogue history and background info for context."
            enhanced_system_prompt = base_system_prompt_text.strip() + memory_guidance

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.8: Construct Final LLM Payload                            ///
            # //////////////////////////////////////////////////////////////////////
            self.logger.debug(
                f"[{session_id}] PIPE_DEBUG: [8] Construct Final LLM Payload"
            )
            await emit_status("MemProc: Constructing final request...")
            final_llm_payload_contents: Optional[List[Dict]] = None  # Explicit typing
            final_payload_tokens: int = -1  # Explicit typing and init

            # Check if the payload construction function is available
            if PROMPT_PAYLOAD_CONSTRUCT_FUNC:
                try:
                    # Call the function to get the payload dictionary
                    payload_dict = PROMPT_PAYLOAD_CONSTRUCT_FUNC(
                        system_prompt=enhanced_system_prompt,
                        history=t0_raw_history_slice,
                        context=combined_context_string,
                        query=latest_user_query_str,
                        strategy="standard",
                        include_ack_turns=self.valves.include_ack_turns,
                    )

                    # Validate the returned dictionary
                    if isinstance(payload_dict, dict):
                        if "error" in payload_dict:
                            # Log error if indicated by the constructor
                            self.logger.error(
                                f"[{session_id}] Error constructing final payload: {payload_dict['error']}"
                            )
                            # Keep final_llm_payload_contents as None
                        elif "contents" in payload_dict and isinstance(
                            payload_dict.get("contents"), list
                        ):
                            # Success: Assign the contents
                            final_llm_payload_contents = payload_dict["contents"]
                            self.logger.info(
                                f"[{session_id}] Constructed final payload ({len(final_llm_payload_contents)} turns)."
                            )

                            # --- Token Calculation Block ---
                            # Check if we can calculate tokens
                            if (
                                UTIL_COUNT_TOKENS_FUNC
                                and self._tokenizer
                                and final_llm_payload_contents
                            ):
                                try:
                                    total_tok = 0
                                    # Explicitly iterate over the assigned list
                                    loop_target_list = final_llm_payload_contents
                                    for (
                                        turn
                                    ) in (
                                        loop_target_list
                                    ):  # <<< Loop target explicitly defined
                                        if (
                                            isinstance(turn, dict)
                                            and "parts" in turn
                                            and isinstance(turn.get("parts"), list)
                                        ):
                                            turn_parts = turn[
                                                "parts"
                                            ]  # Assign parts to a variable
                                            for part in turn_parts:
                                                if (
                                                    isinstance(part, dict)
                                                    and "text" in part
                                                    and isinstance(
                                                        part.get("text"), str
                                                    )
                                                ):
                                                    part_text = part[
                                                        "text"
                                                    ]  # Assign text to a variable
                                                    # Ensure text is not empty before counting
                                                    if part_text:
                                                        total_tok += (
                                                            UTIL_COUNT_TOKENS_FUNC(
                                                                part_text,
                                                                self._tokenizer,
                                                            )
                                                        )
                                    # Assign final count after loop completes
                                    final_payload_tokens = total_tok
                                    self.logger.debug(
                                        f"[{session_id}] Calculated final payload tokens (FinalIN): {final_payload_tokens}"
                                    )
                                except Exception as e_tok_final:
                                    self.logger.error(
                                        f"[{session_id}] Error calculating final payload tokens: {e_tok_final}",
                                        exc_info=False,
                                    )
                                    final_payload_tokens = -1  # Indicate failure
                            elif not final_llm_payload_contents:
                                # If contents list is empty after construction
                                self.logger.debug(
                                    f"[{session_id}] Final payload contents empty. Tokens = 0."
                                )
                                final_payload_tokens = 0
                            else:
                                # Tokenizer/counter unavailable
                                self.logger.warning(
                                    f"[{session_id}] Skipping final token calculation: Tokenizer unavailable."
                                )
                                final_payload_tokens = -1
                            # --- End Token Calculation Block ---

                        else:
                            # Invalid structure if 'contents' key is missing or not a list
                            self.logger.error(
                                f"[{session_id}] Payload constructor returned invalid structure: {payload_dict}"
                            )
                            # Keep final_llm_payload_contents as None
                    else:
                        # If the constructor didn't return a dictionary
                        self.logger.error(
                            f"[{session_id}] Payload constructor did not return a dictionary: {type(payload_dict)}"
                        )
                        # Keep final_llm_payload_contents as None

                except Exception as e_payload:
                    # Catch errors during the call to PROMPT_PAYLOAD_CONSTRUCT_FUNC itself
                    self.logger.error(
                        f"[{session_id}] EXCEPTION during payload construction call: {e_payload}",
                        exc_info=True,
                    )
                    final_llm_payload_contents = None  # Ensure None on exception
                    final_payload_tokens = -1

            else:
                # If the function itself wasn't imported/available
                self.logger.error(
                    f"[{session_id}] Cannot construct final payload: Function unavailable."
                )
                final_llm_payload_contents = None  # Ensure None if function missing
                final_payload_tokens = -1

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.9: Update Output Body                                     ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation unchanged)
            self.logger.debug(f"[{session_id}] PIPE_DEBUG: [9] Update Output Body")
            if final_llm_payload_contents:
                output_body["messages"] = final_llm_payload_contents
                preserved_keys = [
                    "model",
                    "stream",
                    "options",
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "top_k",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                ]
                keys_preserved = [k for k in preserved_keys if k in body]
                [output_body.update({k: body[k]}) for k in keys_preserved]
                self.logger.info(
                    f"[{session_id}] Output body updated. Preserved: {keys_preserved}."
                )
            else:
                self.logger.error(
                    f"[{session_id}] Final payload failed. Output body not updated."
                )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.10: Calculate T0 Tokens & Assemble Final Status Message ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation updated for new refinement flags)
            self.logger.debug(f"[{session_id}] PIPE_DEBUG: [10] Assemble Status")
            t0_dialogue_tokens = -1
            if t0_raw_history_slice and UTIL_COUNT_TOKENS_FUNC and self._tokenizer:
                try:
                    count = sum(
                        UTIL_COUNT_TOKENS_FUNC(msg["content"], self._tokenizer)
                        for msg in t0_raw_history_slice
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str)
                    )
                    t0_dialogue_tokens = count
                    self.logger.debug(
                        f"[{session_id}] Hist tokens: {t0_dialogue_tokens}"
                    )
                except Exception as e_tok_t0:
                    t0_dialogue_tokens = -1
                    self.logger.error(
                        f"[{session_id}] Error calc T0 tokens: {e_tok_t0}"
                    )
            elif not t0_raw_history_slice:
                t0_dialogue_tokens = 0
            else:
                t0_dialogue_tokens = -1
                self.logger.warning(
                    f"[{session_id}] Skipping T0 token calc: Tokenizer unavailable."
                )

            # Assemble Status with new refinement logic
            refinement_status = "Refined=N"
            if cache_update_performed and final_context_selection_performed:
                refinement_status = "Refined=Cache"
            elif stateless_refinement_performed:
                refinement_status = "Refined=Stateless"
            # Include specific step flags if needed for debug: Refined=Cache(U+,S+)
            status_parts = [
                f"T1={t1_retrieved_count}",
                f"T2={t2_retrieved_count}",
                refinement_status,
            ]
            token_parts = []
            if initial_owi_context_tokens >= 0:
                token_parts.append(f"OWI_IN={initial_owi_context_tokens}")
            if refined_context_tokens >= 0:
                token_parts.append(
                    f"RefOUT={refined_context_tokens}"
                )  # Represents final selected context tokens
            if summarization_prompt_tokens >= 0:
                token_parts.append(f"SumIN={summarization_prompt_tokens}")
            if summarization_output_tokens >= 0:
                token_parts.append(f"SumOUT={summarization_output_tokens}")
            if t0_dialogue_tokens >= 0:
                token_parts.append(f"Hist={t0_dialogue_tokens}")
            if final_payload_tokens >= 0:
                token_parts.append(f"FinalIN={final_payload_tokens}")
            status_message = "MemProc: " + ", ".join(status_parts)
            if token_parts:
                status_message += " | Tokens: " + ", ".join(token_parts)
            await emit_status(status_message, done=False)

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.11: Debug Log Final Payload                               ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation unchanged)
            self.logger.debug(
                f"[{session_id}] PIPE_DEBUG: [11] Debug Log Final Payload"
            )
            self._log_debug_final_payload(session_id, output_body)

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.12: Optional Final LLM Call                               ///
            # //////////////////////////////////////////////////////////////////////
            self.logger.debug(
                f"[{session_id}] PIPE_DEBUG: [12] Optional Final LLM Call"
            )
            final_llm_triggered = bool(
                self.valves.final_llm_api_url and self.valves.final_llm_api_key
            )

            if final_llm_triggered:
                self.logger.info(f"[{session_id}] Final LLM Call via Pipe TRIGGERED.")
                await emit_status("MemProc: Executing final LLM Call...")

                if not final_llm_payload_contents:
                    self.logger.error(
                        f"[{session_id}] Cannot execute Final LLM Call: Payload construction failed."
                    )
                    user_error_message = "Apologies, error preparing final request."
                    await emit_status(
                        "ERROR: Final payload preparation failed.", done=True
                    )
                    pipe_exit_time = datetime.now(timezone.utc).isoformat()
                    self.logger.info(
                        f"pipe [{session_id}]: Finished at {pipe_exit_time} (Final LLM payload error)."
                    )
                    return user_error_message
                else:
                    final_call_payload = {"contents": final_llm_payload_contents}
                    self.logger.info(
                        f"[{session_id}] Calling Final LLM: URL='{self.valves.final_llm_api_url[:40]}...', Temp={self.valves.final_llm_temperature}, Timeout={self.valves.final_llm_timeout}s"
                    )

                    # Call the LLM using the async wrapper
                    success, final_response_or_error = (
                        await self._async_llm_call_wrapper(
                            api_url=self.valves.final_llm_api_url,
                            api_key=self.valves.final_llm_api_key,
                            payload=final_call_payload,
                            temperature=self.valves.final_llm_temperature,
                            timeout=self.valves.final_llm_timeout,
                            caller_info="SMP_FinalLLM",  # Shortened caller info
                        )
                    )

                    # --- Process the result ---
                    if (
                        success
                        and isinstance(final_response_or_error, str)
                        and final_response_or_error.strip()
                    ):
                        # Successful LLM call returning non-empty text
                        final_response_text = (
                            final_response_or_error.strip()
                        )  # Ensure stripped
                        self.logger.info(
                            f"[{session_id}] Final LLM Call successful. Length: {len(final_response_text)}."
                        )
                        final_status = f"{status_message} | FinalLLM=OK"
                        await emit_status(final_status, done=True)
                        pipe_exit_time = datetime.now(timezone.utc).isoformat()
                        self.logger.info(
                            f"pipe [{session_id}]: Finished at {pipe_exit_time} (Final LLM OK)."
                        )
                        return final_response_text  # Return the successful response
                    else:
                        # --- Handle LLM call failure or empty/invalid response ---
                        error_info_str = "Unknown error"  # Default error message

                        # Determine the specific error string based on the response
                        if not success and isinstance(final_response_or_error, dict):
                            # Extract details from error dictionary
                            err_type = final_response_or_error.get(
                                "error_type", "UnknownType"
                            )
                            err_msg = final_response_or_error.get("message", "N/A")
                            error_info_str = f"Type: {err_type}, Msg: {err_msg}"
                        elif not success:
                            # Failure, but not a dictionary - convert to string
                            error_info_str = str(final_response_or_error)
                        elif (
                            success
                            and isinstance(final_response_or_error, str)
                            and not final_response_or_error.strip()
                        ):
                            # Success, but empty string response
                            error_info_str = "LLM returned empty response"
                        elif success:
                            # Success, but unexpected response type
                            error_info_str = f"LLM returned unexpected type: {type(final_response_or_error).__name__}"

                        # Log the detailed error
                        self.logger.error(
                            f"[{session_id}] Final LLM Call via Pipe failed. Error: '{error_info_str}'."
                        )

                        # Prepare user-facing message and status
                        user_error_message = f"Apologies, the final processing step failed. Details: {error_info_str}"
                        error_status_desc = f"ERROR: Final LLM Failed ({error_info_str[:50]}...)"  # Truncate for status

                        # Emit final status and log exit
                        await emit_status(error_status_desc, done=True)
                        pipe_exit_time = datetime.now(timezone.utc).isoformat()
                        self.logger.info(
                            f"pipe [{session_id}]: Finished at {pipe_exit_time} (Final LLM error)."
                        )

                        # Return the user-facing error message
                        return user_error_message
            else:
                # Final LLM call is disabled via valves
                self.logger.debug(
                    f"[{session_id}] Skipping Final LLM Call via Pipe (disabled by valves)."
                )

            # //////////////////////////////////////////////////////////////////////
            # /// 5.11.13: Return Modified Payload Body (Default Action)       ///
            # //////////////////////////////////////////////////////////////////////
            # (Implementation unchanged)
            self.logger.info(
                f"[{session_id}] Final LLM Call disabled. Passing modified payload downstream."
            )
            await emit_status(status_message, done=True)
            pipe_exit_time = datetime.now(timezone.utc).isoformat()
            self.logger.info(
                f"pipe [{session_id}]: Finished at {pipe_exit_time} (returning payload dict)."
            )
            return output_body

        # === Global Error Handling ===
        except Exception as e_pipe_main:
            # (Implementation unchanged)
            self.logger.critical(
                f"pipe [{session_id}]: UNHANDLED PIPE EXCEPTION: {e_pipe_main}",
                exc_info=True,
            )
            error_status_desc = f"ERROR: MemPipe Failed ({type(e_pipe_main).__name__})"
            if (
                self.valves.emit_status_updates
                and self._current_event_emitter
                and callable(self._current_event_emitter)
            ):
                try:
                    await self._current_event_emitter(
                        {
                            "type": "status",
                            "data": {"description": error_status_desc, "done": True},
                        }
                    )
                except Exception as e_emit_err:
                    self.logger.error(
                        f"[{session_id}] Failed emit CRITICAL error status: {e_emit_err}"
                    )
            critical_error_message = f"Apologies, critical error in Session Memory Pipe: {type(e_pipe_main).__name__}."
            self.logger.info(
                f"pipe [{session_id}]: Finished at {datetime.now(timezone.utc).isoformat()} (critical error)."
            )
            return critical_error_message


# === SECTION 6: END OF SCRIPT ===
