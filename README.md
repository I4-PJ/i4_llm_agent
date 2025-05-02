# i4_llm_agent - Advanced Memory Management for LLM Roleplay

![Version](https://img.shields.io/badge/version-0.20.1-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

`i4_llm_agent` is a sophisticated memory management system designed for roleplay sessions in OpenWebUI. It implements a multi-tiered memory architecture that allows LLMs to maintain coherent, persistent context beyond standard context window limitations. The system manages various aspects of roleplay state including dialogue history, scene descriptions, character inventories, world state, and environmental elements.

## Features

### Tiered Memory Architecture

- **T0 Memory**: Immediate dialogue history maintained in context window
- **T1 Memory**: Summarized chunks of dialogue stored in SQLite
- **Aged Memory**: Condensed older T1 summaries for efficient long-term retention (New in v0.20.1)
- **T2 Memory**: Vector-stored memories in ChromaDB for semantic retrieval

### Memory Aging System

The Memory Aging feature condenses batches of older T1 summaries into single "aged summaries" when a threshold is reached:

- Configurable trigger threshold for when aging should occur
- Adjustable batch size for condensing summaries
- Preserves narrative continuity while optimizing storage
- Maintains sequential memory flow with clear chronology

### World and Scene Management

- Tracks day, time of day, weather, and season
- Maintains scene descriptions and keywords
- Provides dynamic weather progression with intelligent transitions
- Generates environmental event hints for immersion
- Two-stage state assessment for coherent world progression

### Character Inventory Tracking

- Monitors item acquisitions, removals, and modifications
- Supports command-based and natural language inventory changes
- Recognizes narrative-based inventory modifications
- Stores persistent inventory state per character

### Context Processing

- Refines background context for relevance to current query
- Selects appropriate memory summaries based on context
- Structures content in XML format for clear delineation
- Prepends usage guidelines to ensure proper context interpretation
- RAG-based retrieval for semantically relevant memories

## Technical Components

### Core Modules

- **SessionPipeOrchestrator**: Main coordinator managing the processing flow
- **SessionManager**: In-memory session state handler
- **Pipe**: Interface with OpenWebUI
- **Database Integration**: SQLite for structured data, ChromaDB for vector storage
- **LLM Integration**: Supports multiple LLM APIs through adapter patterns

### Directory Structure

```
i4_llm_agent/
├── __init__.py               # Package exports and version info
├── api_client.py             # LLM API integration (LiteLLM support)
├── cache.py                  # RAG cache functionality
├── context_processor.py      # Context selection and formatting
├── database.py               # SQLite and ChromaDB operations
├── event_hints.py            # Environmental detail generation
├── history.py                # Dialogue history management
├── inventory.py              # Character inventory tracking
├── memory.py                 # T1 summarization and management
├── orchestration.py          # Main processing coordination
├── prompting.py              # Prompt templates and formatting
├── session.py                # Session state management
├── state_assessment.py       # World and scene state tracking
└── utils.py                  # Utility functions (token counting, etc.)
```

## Installation

This package is currently used as an internal component for OpenWebUI. To install:

1. Clone the repository or download the source code
2. Navigate to the project root directory
3. Install the package in development mode:

```bash
pip install .
```

This will install the package and its dependencies in your Python environment.

For development work, you can install in editable mode:

```bash
pip install -e .
```

## Dependencies

- Python 3.8+
- OpenWebUI environment
- LLM API access (configurable)
- Required packages:
  - pydantic
  - chromadb
  - tiktoken
  - httpx

## Configuration

### Environment Variables

Key configuration parameters can be set via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SM_LOG_FILE_PATH` | Path to log file | `C:\Utils\OpenWebUI\session_memory_v20_1_pipe_log.log` |
| `SM_LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) | `DEBUG` |
| `SM_TOKENIZER_ENCODING` | Tokenizer encoding name | `cl100k_base` |
| `SM_T0_ACTIVE_HISTORY_TOKEN_LIMIT` | Token limit for T0 active window | `4000` |
| `SM_T1_SUMMARIZATION_CHUNK_TOKEN_TARGET` | Target token size for T1 chunks | `2000` |
| `SM_MAX_STORED_SUMMARY_BLOCKS` | Maximum T1 summaries before T2 push | `20` |
| `SM_AGING_TRIGGER_THRESHOLD` | Threshold to trigger memory aging | `15` |
| `SM_AGING_BATCH_SIZE` | Number of T1 summaries to condense | `5` |
| `SM_CHROMADB_PATH` | Path to ChromaDB | `C:\Utils\OpenWebUI\session_summary_t2_db` |
| `SM_SQLITE_DB_PATH` | Path to SQLite database | `C:\Utils\OpenWebUI\session_memory_tier1_cache_inventory.db` |

Additional configuration options are available for LLM API endpoints, temperatures, RAG parameters, and feature toggles.

### User Valves

User-specific settings that can be configured per session:

- `long_term_goal`: A persistent objective guiding the session
- `process_owi_rag`: Enable/disable processing of RAG context
- `text_block_to_remove`: Specific text to remove from system prompt
- `period_setting`: Historical period or setting (e.g., "Victorian Era")

## Usage as OpenWebUI Pipe

To use as an OpenWebUI pipe:

1. Install the package in your OpenWebUI environment
2. Configure the pipe in OpenWebUI settings
3. Enable the pipe for your chat sessions

Example pipe configuration:

```json
{
  "type": "pipe",
  "name": "SESSION_MEMORY PIPE (v0.20.1 - Memory Aging)",
  "valves": {
    "log_level": "INFO",
    "aging_trigger_threshold": 15,
    "aging_batch_size": 5,
    "enable_inventory_management": true,
    "enable_event_hints": true
  }
}
```

## Memory Aging Process

The memory aging system follows this process:

1. Monitor T1 summary count against threshold
2. When threshold is reached, select oldest batch of T1 summaries
3. Condense batch into a single, coherent "aged summary"
4. Store the new aged summary and delete original T1 summaries
5. Make aged summaries available for future context

This creates a progressively compressed memory hierarchy that mimics human memory abstraction over time.

## Status and Debugging

The system provides detailed status updates about memory operations:

- T1 summarization metrics
- Aging operations tracking
- T2 transitions
- Context selection information
- Token usage across different memory tiers

Debug logs can be enabled for detailed analysis of memory operations.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenWebUI community for the integration framework
- ChromaDB for vector storage capabilities
- Contributors to the LLM agent architecture

## Roadmap

- [ ] Improved inventory item merging
- [ ] Enhanced memory decay simulation
- [ ] Multi-character relationship tracking
- [ ] Expanded period settings library
- [ ] Performance optimizations for large session histories

---

*Version 0.20.1 - Memory Aging, Reverted DB Name*
