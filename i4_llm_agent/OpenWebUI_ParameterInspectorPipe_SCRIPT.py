# === SECTION 1: METADATA HEADER ===
# --- REQUIRED METADATA HEADER ---
"""
title: Parameter Inspector Pipe (v1.0.1 - Callable Fix)
author: Assistant
version: 1.0.1
description: Corrected v1.0.0 - Added missing 'Callable' import from typing. Inspects arguments passed to the pipe function based on its signature.
requirements: fastapi, pydantic, json # Basic requirements
"""
# --- END HEADER ---

# === SECTION 2: IMPORTS ===
import logging
import re
import json  # To pretty-print dicts
from typing import List, Dict, Optional, Union, Any, Callable  # <<< ADDED Callable HERE
from pydantic import BaseModel, Field
from fastapi import Request  # Import Request for type hinting


# === SECTION 3: PIPE CLASS DEFINITION ===
class Pipe:
    # Basic pipe attributes
    type: str = "pipe"
    name: str = "Parameter Inspector"
    version: str = "1.0.1"  # Incremented version

    # Minimal Valves
    class Valves(BaseModel):
        pass  # No configurable options needed

    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)  # Capture all logs
        self.valves = self.Valves()
        self.logger.info(f"Initialized {self.name} v{self.version}")
        # Ensure basic console handler if none exists
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # --- Main Pipe Method ---
    async def pipe(
        self,
        # --- Arguments we expect to receive ---
        body: dict,  # Always received
        __request__: Optional[Request] = None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __session_id__: Optional[str] = None,  # Check this too
        __event_emitter__: Optional[Callable] = None,  # <<< TYPE HINT NOW VALID
        # --- Catch any other arguments OWI might send ---
        **kwargs: Any,
    ) -> str:  # Return string to chat UI
        """
        Inspects declared arguments and reports which ones were received.
        """
        self.logger.info(f"{self.name}: Entered pipe method.")
        findings = []

        # --- Log details about each expected argument ---

        # 1. body
        findings.append(f"Argument 'body': Received (Type: {type(body).__name__})")
        body_keys = list(body.keys())
        findings.append(f"  Body Keys: {body_keys}")
        self.logger.debug(f"Argument 'body' received. Keys: {body_keys}")

        # 2. __request__
        if __request__ is not None:
            findings.append(
                f"Argument '__request__': Received (Type: {type(__request__).__name__})"
            )
            url_path = "N/A"
            try:
                url_path = str(__request__.url.path)
            except Exception:
                pass  # Ignore errors getting path here
            findings.append(f"  Request Path (from __request__): {url_path}")
            self.logger.debug(f"Argument '__request__' received. Path: {url_path}")
        else:
            findings.append("Argument '__request__': NOT received (was None).")
            self.logger.debug("Argument '__request__' was None.")

        # 3. __user__
        if __user__ is not None:
            findings.append(
                f"Argument '__user__': Received (Type: {type(__user__).__name__})"
            )
            user_id = __user__.get("id", "N/A")
            findings.append(f"  User ID (from __user__): {user_id}")
            self.logger.debug(f"Argument '__user__' received. ID: {user_id}")
        else:
            findings.append("Argument '__user__': NOT received (was None).")
            self.logger.debug("Argument '__user__' was None.")

        # 4. __metadata__
        if __metadata__ is not None:
            findings.append(
                f"Argument '__metadata__': Received (Type: {type(__metadata__).__name__})"
            )
            metadata_keys = list(__metadata__.keys())
            findings.append(f"  Metadata Keys (from __metadata__): {metadata_keys}")
            self.logger.debug(
                f"Argument '__metadata__' received. Keys: {metadata_keys}"
            )
            try:
                # Attempt to serialize with default=str for non-standard types
                metadata_preview = json.dumps(__metadata__, indent=2, default=str)
                self.logger.debug(f"__metadata__ Content Preview:\n{metadata_preview}")
            except Exception as e:
                self.logger.error(f"Could not serialize __metadata__: {e}")
                self.logger.debug(f"Raw __metadata__ repr: {repr(__metadata__)}")
        else:
            findings.append("Argument '__metadata__': NOT received (was None).")
            self.logger.debug("Argument '__metadata__' was None.")

        # 5. __chat_id__
        if __chat_id__ is not None:
            findings.append(
                f"Argument '__chat_id__': Received (Type: {type(__chat_id__).__name__}, Value: '{__chat_id__}')"
            )
            self.logger.debug(
                f"Argument '__chat_id__' received with value: {__chat_id__}"
            )
        else:
            findings.append("Argument '__chat_id__': NOT received (was None).")
            self.logger.debug("Argument '__chat_id__' was None.")

        # 6. __session_id__
        if __session_id__ is not None:
            findings.append(
                f"Argument '__session_id__': Received (Type: {type(__session_id__).__name__}, Value: '{__session_id__}')"
            )
            self.logger.debug(
                f"Argument '__session_id__' received with value: {__session_id__}"
            )
        else:
            findings.append("Argument '__session_id__': NOT received (was None).")
            self.logger.debug("Argument '__session_id__' was None.")

        # 7. __event_emitter__
        if __event_emitter__ is not None:
            is_callable = callable(__event_emitter__)
            findings.append(
                f"Argument '__event_emitter__': Received (Type: {type(__event_emitter__).__name__}, Callable: {is_callable})"
            )
            self.logger.debug(
                f"Argument '__event_emitter__' received. Callable: {is_callable}"
            )
        else:
            findings.append("Argument '__event_emitter__': NOT received (was None).")
            self.logger.debug("Argument '__event_emitter__' was None.")

        # 8. kwargs (Other unexpected arguments)
        if kwargs:
            findings.append(
                f"Other Keyword Args Received (**kwargs): {list(kwargs.keys())}"
            )
            self.logger.debug(f"Received kwargs: {kwargs}")
        else:
            findings.append("No other keyword arguments received in **kwargs.")
            self.logger.debug("No other keyword arguments received in **kwargs.")

        # --- Format and Return Findings ---
        summary_message = "Pipe Parameter Inspection Results (v1.0.1):\n-----------------------------------------\n"
        summary_message += "\n".join(findings)
        summary_message += "\n-----------------------------------------"
        self.logger.info("Inspection complete. Returning summary.")

        return summary_message


# === SECTION 4: END OF SCRIPT ===

