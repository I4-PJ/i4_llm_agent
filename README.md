
# Session Memory Pipe for Open WebUI (v0.18.6+)

**Purpose**: Enhances long-form, character-driven roleplay sessions in Open WebUI by providing persistent memory, context management, and flexible LLM integration using the `i4_llm_agent` library. It intelligently summarizes conversations, retrieves relevant memories and lore, refines context, and constructs optimized prompts for the final AI model, while managing context length and API costs.

**Version**: Aligns with Pipe Script v0.18.6+ and `i4_llm_agent` library v0.1.4+

---

## Core Components

This system consists of two main parts:

1.  **`OpenWebUI_MemorySession_SCRIPT` (The Pipe Script):**
    *   The script file loaded into Open WebUI's `pipe` function.
    *   Handles integration with Open WebUI (receiving requests, user info, chat ID, embedding functions).
    *   Manages global configuration via environment variables (`SM_*`) mapped to `Pipe.Valves`.
    *   Sets up database connections (SQLite, ChromaDB) and the tokenizer.
    *   Instantiates and coordinates with the `SessionPipeOrchestrator`.
    *   Handles session identification (`__chat_id__`) and user-specific settings (`__user__['valves']`).
    *   Determines regeneration heuristics.
    *   Manages the final output (either passing a modified payload downstream or returning a direct string response).

2.  **`i4_llm_agent` (The Python Library):**
    *   Contains the core logic for memory management, RAG, prompting, API calls, session state, etc.
    *   Provides the `SessionPipeOrchestrator` class, which encapsulates the main processing flow invoked by `script.txt`.
    *   Includes modules for: `api_client`, `cache`, `database`, `history`, `memory`, `orchestration`, `prompting`, `session`, `utils`.
    *   Designed to be potentially reusable outside the specific OWI pipe context.

---

## 🧠 Memory Management (T0 → T1 → T2)

### 🔹 TIER 0 (Active Dialogue Window)

*   The portion of the **recent dialogue history** (user/assistant turns) considered active for the immediate context.
*   Starts *after* the last message included in a T1 summary.
*   Its size implicitly influences the T1 summarization trigger (`T0_TOKEN_LIMIT` + `T1_CHUNK_TARGET`).

### 🔸 TIER 1 (Short-Term Summarized Memory)

*   **Trigger:** When the unsummarized dialogue exceeds a threshold (`T0_TOKEN_LIMIT` + `T1_CHUNK_TARGET`).
*   **Process:** The oldest chunk of unsummarized dialogue (target size `T1_CHUNK_TARGET`) is summarized by an LLM (e.g., Gemini Flash).
*   **Storage:** Summaries are saved in a **SQLite database** (`tier1_text_summaries` table) with metadata (turn indices, timestamps).
*   **Regen Check:** Avoids creating duplicate summaries for the exact same turn indices during regeneration.
*   **Limit:** Stores up to `SM_MAX_STORED_SUMMARY_BLOCKS`. Oldest is migrated to T2 when the limit is exceeded.

### 🟣 TIER 2 (Long-Term Embedded Memory/Lore)

*   **Source:** T1 summaries migrated due to storage limits. (Can potentially be extended to ingest external lore documents).
*   **Process:** Text is embedded using OWI's configured embedding model.
*   **Storage:** Embeddings and text stored in a session-specific **ChromaDB collection**.
*   **Retrieval:** Queried using semantic search based on an LLM-generated query derived from the current dialogue context (`generate_rag_query`).

---

## ✨ Context Refinement & RAG

This system offers two primary modes for processing external context (like that provided by OWI's RAG):

### 1. RAG Cache (Two-Step)

*   **Enabled via:** `SM_ENABLE_RAG_CACHE=true` (Global Valve)
*   **Requires:** `SM_REFINER_API_URL/KEY`, `SM_CACHE_UPDATE_PROMPT_TEMPLATE`, `SM_FINAL_SELECT_PROMPT_TEMPLATE`.
*   **Step 1 (Update Cache):** Merges previous cache content (SQLite `session_rag_cache` table), current OWI context (if `process_owi_rag` user valve is true), and recent history using an LLM call. Skips LLM call if OWI context is too short (`SM_CACHE_UPDATE_SKIP_OWI_THRESHOLD`) or too similar to the previous cache (`SM_CACHE_UPDATE_SIMILARITY_THRESHOLD`). Updates the SQLite cache table.
*   **Step 2 (Select Final Context):** Takes the updated cache (and optionally current OWI context again) and uses another LLM call to extract only the snippets most relevant to the current user query and history. This selected context is then used in the final prompt.
*   **Goal:** Maintain a persistent, refined cache of background info per session, minimizing redundant processing and selecting only hyper-relevant details for the final prompt.

### 2. Stateless Refinement

*   **Enabled via:** `SM_ENABLE_STATELESS_REFINEMENT=true` (Global Valve, ignored if RAG Cache is enabled).
*   **Requires:** `SM_REFINER_API_URL/KEY`, `SM_STATELESS_REFINER_PROMPT_TEMPLATE`.
*   **Process:** Takes the current OWI context (if `process_owi_rag` user valve is true) and uses a single LLM call to extract/rephrase the parts most relevant to the current query and history. Skips if OWI context is below `SM_STATELESS_REFINER_SKIP_THRESHOLD`.
*   **Goal:** Quickly filter the current OWI context for relevance without maintaining a persistent cache.

### T2 RAG (Independent Lookup)

*   Happens regardless of the refinement mode chosen above.
*   Generates a query using `generate_rag_query` (needs `SM_RAGQ_LLM_API_URL/KEY/PROMPT`).
*   Searches the T2 ChromaDB collection for relevant long-term summaries/lore.
*   Results are included in the `combined_context_string` alongside T1 and refined/raw OWI context.

---

## 🚀 Key Features & Concepts

*   **Modular Design:** Core logic separated into the `i4_llm_agent` library, orchestrated by `OpenWebUI_MemorySession_SCRIPT`.
*   **Tiered Memory:** Efficiently manages short-term and long-term context (T0 -> T1 -> T2).
*   **Configurable Context Refinement:** Choose between stateless filtering or a two-step persistent cache.
*   **Integrated T2 RAG:** Retrieves relevant long-term memories via semantic search.
*   **Session Isolation:** Uses `__chat_id__` and user ID to separate memory and state.
*   **Regeneration Handling:** Prevents duplicate T1 summary generation for identical history blocks.
*   **User-Configurable Session Valves:** Allows per-chat overrides (`long_term_goal`, `process_owi_rag`, `text_block_to_remove`).
*   **Optional Final LLM Call:** Pipe can either pass the constructed payload downstream OR make a final non-streaming call itself and return the response string.
*   **Flexible LLM Configuration:** Separate API settings for Summarizer, RAG Query Gen, Refiner, and Final LLM.
*   **Status Updates:** Provides feedback to the UI via `__event_emitter__`.
*   **Debug Logging:** Optional detailed logging of inputs and final payloads.

---

## How it Works (Processing Flow)

When the pipe receives a request from Open WebUI:

1.  **Initialization & Session Setup:**
    *   `OpenWebUI_MemorySession_SCRIPT` validates input and identifies the session using `__chat_id__` and user ID.
    *   Retrieves or creates an in-memory session state using `SessionManager`.
    *   Parses `UserValves` from `__user__['valves']` (session-specific settings like `long_term_goal`, `process_owi_rag`).
    *   Calculates a `is_regeneration_heuristic` flag based on comparing current and previous input messages.

2.  **Orchestration Begins (`SessionPipeOrchestrator.process_turn`):**
    *   **History Sync:** Ensures the orchestrator's view of active history matches the incoming request.
    *   **Effective Query Determination:** Identifies the actual user query to process (handling regeneration).
    *   **T1 Summarization Check:** Calls `manage_tier1_summarization` to check if enough new dialogue exists to warrant summarization. **Includes a check to skip LLM call if regenerating an identical block.**
    *   **T1 -> T2 Transition Check:** If T1 summarization occurred and T1 storage limits are exceeded, moves the oldest T1 summary to the T2 ChromaDB vector store.
    *   **Context Preparation:**
        *   Retrieves recent T1 summaries (SQLite).
        *   Generates a RAG query based on recent dialogue and searches T2 (ChromaDB) for relevant older summaries/lore.
    *   **Context Refinement:** Based on global valves (`SM_ENABLE_RAG_CACHE`, `SM_ENABLE_STATELESS_REFINEMENT`):
        *   **RAG Cache (Two-Step):** If enabled, updates a session-specific cache (SQLite) by merging previous cache, current OWI context (if `process_owi_rag` user valve is true), and history, then selects the most relevant parts for the current query using LLM calls. Includes similarity/length skip logic for the update step.
        *   **Stateless Refinement:** If enabled (and RAG Cache disabled), uses an LLM call to refine the current OWI context based on the query and history.
        *   **None:** If neither is enabled, uses raw OWI context (if `process_owi_rag` is true) or potentially no external context.
    *   **Combine Background:** Merges the refined/raw context, T1 summaries, and T2 RAG results into a single `combined_context_string`.
    *   **T0 History Selection:** Selects the relevant recent dialogue turns (after the last T1 summary, before the effective query) for the final prompt.
    *   **Final Payload Construction:** Builds the `contents` list for the final LLM using the processed system prompt, T0 history, combined context, user query, and **injecting the `long_term_goal` user valve**.
    *   **Final Status Calculation:** Assembles a status string with key metrics (memory counts, token counts, refinement status).
    *   **Execute/Prepare Output:**
        *   If `SM_FINAL_LLM_API_URL/KEY` are set, makes a **non-streaming** call to the specified final LLM using the constructed payload. Returns the LLM's response string or an error string.
        *   If not triggered, returns the modified payload dictionary (containing the constructed `messages`) for OWI to handle downstream.

3.  **Return Result:** `script.txt` receives the `Dict` or `str` from the orchestrator and returns it to Open WebUI.

---

## ⚙️ Configuration (Valves & Environment Variables)

When setting openrouter API, use path like that https://openrouter.ai/api/v1/chat/completions#google/gemini-2.5-pro   so, API path separated by '#' from model name. Google API (prefered) is set as e.g. https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-03-25:generateContent. To be refactored in future.

Set these environment variables before starting Open WebUI to configure the pipe globally.

| Env Var                                | Default Value (or Description)                      | Purpose                                                                 |
| :------------------------------------- | :-------------------------------------------------- | :---------------------------------------------------------------------- |
| `SM_LOG_FILE_PATH`                     | `...\OpenWebUI\session_memory_v18_6_pipe.log` | Path for the main log file.                                             |
| `SM_LOG_LEVEL`                         | `DEBUG`                                             | Logging level (DEBUG, INFO, WARNING, ERROR).                            |
| `SM_SQLITE_DB_PATH`                    | `...\session_memory_tier1_and_cache.db`             | Path for SQLite DB (T1 Summaries & RAG Cache).                          |
| `SM_CHROMADB_PATH`                     | `...\session_summary_t2_db`                         | Path for ChromaDB persistent storage (T2 Embeddings).                   |
| `SM_SUMMARY_COLLECTION_PREFIX`         | `sm_t2_`                                            | Prefix for ChromaDB collection names (followed by session ID).          |
| `SM_TOKENIZER_ENCODING`                | `cl100k_base`                                       | tiktoken encoding name for token counting.                              |
| `SM_INCLUDE_ACK_TURNS`                 | `true`                                              | Include 'Understood.' turns in constructed prompts.                     |
| `SM_EMIT_STATUS_UPDATES`               | `true`                                              | Send processing status updates to the UI.                               |
| **T1 Summarizer**                      |                                                     |                                                                         |
| `SM_SUMMARIZER_API_URL`                | Gemini Flash Endpoint                               | URL for the T1 summarization LLM.                                       |
| `SM_SUMMARIZER_API_KEY`                | `""` (REQUIRED)                                     | API Key for the T1 summarizer LLM.                                      |
| `SM_SUMMARIZER_TEMPERATURE`            | `0.5`                                               | Temperature for T1 summarization (0.0-2.0).                             |
| `SM_SUMMARIZER_SYSTEM_PROMPT`          | (See Constants)                                     | System prompt for the T1 summarizer.                                    |
| `SM_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET` | `2000`                                              | Target token size for dialogue chunks sent to T1 summarizer.            |
| `SM_MAX_STORED_SUMMARY_BLOCKS`         | `10`                                                | Max T1 summaries in SQLite before migrating oldest to T2.                 |
| `SM_T0_ACTIVE_HISTORY_TOKEN_LIMIT`     | `4000`                                              | Approx. token limit that triggers T1 summarization check.               |
| **T2 RAG Query Generation**            |                                                     |                                                                         |
| `SM_RAGQ_LLM_API_URL`                  | Gemini Flash Endpoint                               | URL for the RAG query generation LLM.                                   |
| `SM_RAGQ_LLM_API_KEY`                  | `""` (REQUIRED)                                     | API Key for the RAG query generation LLM.                               |
| `SM_RAGQ_LLM_TEMPERATURE`              | `0.3`                                               | Temperature for RAG query generation.                                   |
| `SM_RAGQ_LLM_PROMPT`                   | (See Constants)                                     | Prompt template for RAG query generation.                               |
| `SM_RAG_SUMMARY_RESULTS_COUNT`         | `3`                                                 | Number of T2 results to retrieve from ChromaDB.                         |
| **Context Refinement**                 |                                                     |                                                                         |
| `SM_ENABLE_RAG_CACHE`                  | `false`                                             | Enable the Two-Step RAG Cache refinement feature.                       |
| `SM_ENABLE_STATELESS_REFINEMENT`       | `false`                                             | Enable Stateless Refinement (if RAG Cache is false).                    |
| `SM_REFINER_API_URL`                   | Gemini Flash Endpoint                               | URL for the Refinement LLM (used by both RAG Cache & Stateless).        |
| `SM_REFINER_API_KEY`                   | `""` (REQUIRED if refinement enabled)               | API Key for the Refinement LLM.                                         |
| `SM_REFINER_TEMPERATURE`               | `0.3`                                               | Temperature for Refinement LLM calls.                                   |
| `SM_REFINER_HISTORY_COUNT`             | `6`                                                 | Number of recent history turns used in refinement prompts.              |
| `SM_CACHE_UPDATE_PROMPT_TEMPLATE`      | (Library Default)                                   | Env var to override the Step 1 (Cache Update) prompt template.          |
| `SM_FINAL_SELECT_PROMPT_TEMPLATE`      | (Library Default)                                   | Env var to override the Step 2 (Final Select) prompt template.          |
| `SM_CACHE_UPDATE_SKIP_OWI_THRESHOLD`   | `50`                                                | Skip Cache Step 1 if OWI context char length < this.                    |
| `SM_CACHE_UPDATE_SIMILARITY_THRESHOLD` | `0.9`                                               | Skip Cache Step 1 if OWI context similarity to cache > this (0.0-1.0).  |
| `SM_STATELESS_REFINER_PROMPT_TEMPLATE` | (Library Default)                                   | Env var to override the Stateless Refinement prompt template.           |
| `SM_STATELESS_REFINER_SKIP_THRESHOLD`  | `500`                                               | Skip Stateless Refinement if OWI context char length < this.            |
| **Optional Final LLM Call**            |                                                     |                                                                         |
| `SM_FINAL_LLM_API_URL`                 | `""` (DISABLED by default)                          | URL for the final LLM call (if enabled, pipe returns string response).  |
| `SM_FINAL_LLM_API_KEY`                 | `""` (DISABLED by default)                          | API Key for the final LLM call.                                         |
| `SM_FINAL_LLM_TEMPERATURE`             | `0.7`                                               | Temperature for the final LLM call.                                     |
| `SM_FINAL_LLM_TIMEOUT`                 | `120`                                               | Timeout in seconds for the final LLM call.                              |
| **Debugging**                          |                                                     |                                                                         |
| `SM_DEBUG_LOG_RAW_INPUT`               | `false`                                             | Log the full raw input `body` to a `.DEBUG_INPUT.log` file.           |
| `SM_DEBUG_LOG_FINAL_PAYLOAD`           | `false`                                             | Log the final constructed payload to a `.DEBUG_PAYLOAD.log` file.       |

---

## 👤 User Session Valves

These settings can be configured **per chat** within the Open WebUI "Chat Settings" -> "Valves" section for the pipe.

| Parameter               | Type    | Default | Description                                                                                                                               |
| :---------------------- | :------ | :------ | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `long_term_goal`        | `string`| `""`    | Persistent session goal/instruction injected into the system prompt. Remains until changed in chat settings.                                |
| `process_owi_rag`       | `bool`  | `true`  | If `true`, processes context from OWI's RAG. If `false`, ignores OWI context and skips RAG Cache Step 1 (cache update based on OWI input). |
| `text_block_to_remove`  | `string`| `""`    | Exact text block to remove from the system prompt (e.g., conflicting default instructions). Leave empty to disable.                           |

---

## 🚀 Setup & Usage

1.  **Install Library:** Ensure the `i4_llm_agent` library (v0.1.4+) and its dependencies (`tiktoken`, `httpx`, `chromadb`) are installed in your Open WebUI Python environment. You might need a `requirements.txt`:
    ```txt
    # requirements.txt (example)
    tiktoken
    chromadb
    i4_llm_agent>=0.1.4
    httpx
    # Add other dependencies if needed by OWI or your setup
    ```
    Install using `pip install -r requirements.txt`.
2.  **Place Script:** Copy the latest `script.txt` file into your Open WebUI `pipes` directory (e.g., `open-webui/backend/data/pipes/script.txt`). Create the `pipes` directory if it doesn't exist.
3.  **Configure Environment:** Set the `SM_*` environment variables (especially API keys and desired paths) before launching Open WebUI.
4.  **Restart Open WebUI:** Ensure the changes are picked up.
5.  **Select Pipe:** In Open WebUI, select the "SESSION\_MEMORY PIPE" in the chat settings.
6.  **Configure User Valves:** Adjust the session-specific User Valves (`long_term_goal`, etc.) in the Chat Settings -> Valves section as needed.

---

## 💾 Storage

*   **SQLite Database (`SM_SQLITE_DB_PATH`):** Stores Tier 1 summaries and the RAG Cache table (`session_rag_cache`).
*   **ChromaDB (`SM_CHROMADB_PATH`):** Stores Tier 2 embedded summaries/lore vector data persistently.
*   **Log Files (`SM_LOG_FILE_PATH` directory):** Contains the main `.log` file and optional `.DEBUG_INPUT.log` and `.DEBUG_PAYLOAD.log` files.

---

## 📈 Roadmap & Enhancements

*   ✅ **Tiered Memory (T0/T1/T2):** Implemented.
*   ✅ **Context Refinement Options:** RAG Cache (Two-Step) and Stateless Refinement implemented.
*   ✅ **T2 RAG Lookup:** Implemented.
*   ✅ **Session Isolation & User Valves:** Implemented (`long_term_goal`, `process_owi_rag`, `text_block_to_remove`).
*   ✅ **Regeneration Handling:** Duplicate T1 check implemented.
*   ✅ **Non-Streaming Final LLM Option:** Implemented.
*   🔄 **Investigate:** History synchronization robustness, especially around OWI stop/regen actions causing empty history lists.
*   💡 **Planned:** Time-aware memory retrieval, selective character-scoped embeddings, direct lore document ingestion into T2.
*   💡 **Optional:** Integrate local LLMs for summarization/refinement steps.

---

## ✍️ Author & Credits

*   Primarily designed and developed by Petr Jílek & AI Assistant (Gemini Pro).
*   Leverages the `i4_llm_agent` library.
*   Integrates with Open WebUI, ChromaDB, and various LLM APIs.

---
