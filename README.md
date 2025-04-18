
> **Purpose**: The PIPE system is designed to manage long-form, character-driven roleplay sessions by combining memory summarization, RAG-based lore retrieval, and LLM prompt construction while controlling context bloat and optimizing cost.
>
> Open webUI script is in the file OpenWebUI_MemorySession_SCRIPT.py

## Overview

<pre>
+------------------------------------------------------------+
|                    USER MESSAGE INPUT                     |
+------------------------------------------------------------+
                             |
               +-------------+--------------+
               |                            |
+--------------v-------------+  +-----------v-------------+
|     MEMORY PIPELINE        |  |       RAG PIPELINE      |
|   (T0 → T1 → T2 Process)   |  | (Query → Cache → Refine)|
+--------------+-------------+  +-----------+-------------+
               |                            |
   +-----------v------------+    +----------v------------+
   | T0: Active Chat Window |    | Generate Query from   |
   | (last N turns,         |    |   T0 + User Input     |
   |     token-limited)     |    |                       |
   +-----------+------------+    +----------+------------+
               |                            |
   +-----------v------------+    +----------v------------+
   | If T0 > limit:         |    | RAG: Search ChromaDB  |
   | Summarize oldest block |    | (vector store - T2)   |
   | → via external LLM     |    +----------+------------+
   +-----------+------------+               |
               |                            |
   +-----------v------------+    +----------v-------------+
   | Store summary to T1 DB |    | Step 1: Gemini refines |
   | (SQLite, fixed-length) |    |        → cache (SQLite)|
   +-----------+------------+    +----------+-------------+
               |                            |
   +-----------v------------+    +----------v-------------+
   | If T1 is full:         |    | Step 2: Select context |
   | Push oldest summary to |    | relevant to current    |
   | T2 (ChromaDB vector DB)|    | turn from refined cache|
   +-----------+------------+    +----------+-------------+
               |                            |
               +-------------+--------------+
                             |
                +------------v------------+
                |  PROMPT CONSTRUCTION    |
                |  - System Prompt        |
                |  - T0 History           |
                |  - RAG Memory Slice     |
                |  - User Message         |
                +------------+------------+
                             |
                +------------v------------+
                |   FINAL LLM GENERATION  |
                |   (e.g., GPT-4, Claude) |
                +-------------------------+
</pre>


PIPE orchestrates memory and retrieval around a **dual-path memory processing system**:

- **Memory Line (TIER 0 → TIER 1 → TIER 2)**
- **RAG Line (Query ➝ Cache ➝ Compression ➝ Prompt)**

These paths converge at the prompt construction phase, where only the most relevant, compressed context is injected into the main LLM.

---

## 🧠 Memory Processing (T0 → T1 → T2)

### 🔹 TIER 0 (Active Turns)

- Holds the **latest N turns** of chat history (user + assistant)
- Governed by `T0_TOKEN_LIMIT` (e.g., 4000 tokens)

### 🔸 TIER 1 (Summarized Memory)

- When T0 exceeds its token limit:

  - A complete user + assistant cycle block is summarized by **external LLM (e.g. Gemini)**
  - The summarized block is saved in a **SQLite T1 DB**
  - The original chat block is trimmed from T0

- T1 DB can store multiple blocks (max defined by `MAX_T1_BLOCKS`)

- When full: **oldest T1 block is migrated into T2**

### 🟣 TIER 2 (Embedded Memory)

- T1 blocks pushed into T2 are vectorized and stored in **ChromaDB**
- Serves as the long-term narrative memory (lore, prior arcs, etc.)

---

## 🔁 RAG Line Processing (Parallel)

### Step 1: **Generate Query**

- Uses recent T0 turns + current user message to create semantic RAG query

### Step 2: **Retrieve Candidates** (T2)

- Searches ChromaDB using generated query

### Step 3: **Refine via Cache Update**

- Uses Gemini or other LLM to summarize/filter raw RAG results
- Saves refined block into a **cache table in SQLite**

### Step 4: **Final Context Selection**

- Uses current dialogue and intent to extract only the most relevant context from cache
- Final \~100–300 tokens are injected into LLM

---

## 📤 Prompt Construction Flow

1. Collect recent **T0 history**
2. Retrieve selected memory from **RAG cache (refined)**
3. Combine with **system prompt** and **user input**
4. Send to **final LLM** for generation (GPT-4, Claude, Gemini Pro, etc.)

---

## 🎯 Key Features

- **Parallel memory + RAG pipelines**
- **Two-step cache refinement** for accuracy and brevity
- **Tiered memory**: summarized, persistent, embedded
- **Session isolation** with scoped chat/user IDs
- **Context budget enforcement** to reduce bloat

---

## ⚙️ Customizable Valves (ENV Options)

| Env Var                            | Description                                 |
| ---------------------------------- | ------------------------------------------- |
| `SM_ENABLE_RAG_CACHE=true`         | Enables two-step cache refiner              |
| `SM_REFINER_API_KEY`               | Gemini or Claude key for refinement         |
| `SM_FINAL_LLM_API_URL`             | Target LLM endpoint for generation          |
| `SM_T0_ACTIVE_HISTORY_TOKEN_LIMIT` | Token limit for active window               |
| `SM_CACHE_UPDATE_PROMPT_TEMPLATE`  | Custom prompt for Step 1 cache summarizer   |
| `SM_FINAL_SELECT_PROMPT_TEMPLATE`  | Custom prompt for Step 2 relevance selector |

---

## 💸 Cost Optimization Strategy

| Stage                    | LLM Involved                | Cost Notes                                   |
| ------------------------ | --------------------------- | -------------------------------------------- |
| **T0 → T1 Summarizer**   | Gemini or Claude            | ✅ Triggered only when T0 exceeds limit       |
| **RAG Cache Update**     | Gemini                      | ✅ Cached per scene/query hash                |
| **Final Context Select** | Gemini                      | ✅ Small token footprint, reused during scene |
| **Final LLM Call**       | GPT-4 / Claude / Gemini Pro | 🔥 Premium model, highly focused input       |

---

## 📂 File System & Storage

- **ChromaDB**: Persistent memory (T2)
- **SQLite**: T1 summaries + refined RAG cache
- **Logs**: `.DEBUG_INPUT`, `.DEBUG_PAYLOAD`, `.log` files for inspection

---

## 🧪 Roadmap & Enhancements

- ✅ Scene-aware memory flushing from T0
- ✅ Smart chunk alignment based on speaker turns
- 🔄 Planned: timecode-tagged memory retrieval, selective character-scoped embeddings
- 🚧 Optional: integrate local summarizers for low-cost preprocessing

---

## ✍️ Author & Credits

- Designed by Petr Jílek and Agent Gemini 2.5 Pro
- Built using `i4_llm_agent`, `ChromaDB`, `Open WebUI`, and Gemini APIs

---

## 📎 Version

**PIPE v0.18.0**\
Supports: Dual-path Memory and RAG Pipelines, Two-Step Cache Refinement, Stateless Summarization

