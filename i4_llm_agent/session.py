# i4_llm_agent/session.py
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__) # i4_llm_agent.session

class SessionManager:
    """
    Manages in-memory session state for the Session Memory Pipe.

    Each session's state includes:
    - active_history: The full message history as last received.
    - last_summary_turn_index: The index in active_history of the last message
                               included in a Tier 1 summary.
    - previous_input_messages: Stores the `body.messages` from the previous pipe
                               invocation for regeneration detection.
    - user_valves: Stores the validated UserValves object for the session.
    - rag_cache: Stores the latest RAG cache content for the session (optional).
    """

    def __init__(self):
        """Initializes the session manager with an empty sessions dictionary."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("SessionManager initialized.")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the state for a given session ID.

        Args:
            session_id: The unique identifier for the session.

        Returns:
            The session state dictionary if found, otherwise None.
        """
        return self.sessions.get(session_id)

    def create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Creates a new session state with default values.

        Args:
            session_id: The unique identifier for the new session.

        Returns:
            The newly created session state dictionary.
        """
        if session_id in self.sessions:
            logger.warning(f"Session '{session_id}' already exists. Returning existing session.")
            return self.sessions[session_id]

        logger.info(f"Creating new session state for ID: {session_id}")
        new_session_state: Dict[str, Any] = {
            "active_history": [],
            "last_summary_turn_index": -1,
            "previous_input_messages": None,
            "user_valves": None, # Will be populated by the orchestrator/pipe
            "rag_cache": None, # Will be populated if RAG Cache is enabled
        }
        self.sessions[session_id] = new_session_state
        return new_session_state

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieves an existing session state or creates a new one if it doesn't exist.
        Ensures default keys are present.

        Args:
            session_id: The unique identifier for the session.

        Returns:
            The session state dictionary.
        """
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        else:
            # Ensure default keys exist for sessions potentially created before code updates
            session.setdefault("active_history", [])
            session.setdefault("last_summary_turn_index", -1)
            session.setdefault("previous_input_messages", None)
            session.setdefault("user_valves", None)
            session.setdefault("rag_cache", None)
            logger.debug(f"Retrieved existing session state for ID: {session_id}")

        return session

    def update_session_state(self, session_id: str, key: str, value: Any) -> bool:
        """
        Updates a specific key within a session's state.

        Args:
            session_id: The ID of the session to update.
            key: The key within the session state to update.
            value: The new value for the key.

        Returns:
            True if the session exists and was updated, False otherwise.
        """
        session = self.get_session(session_id)
        if session is not None:
            logger.debug(f"Updating session '{session_id}': Setting '{key}'...")
            session[key] = value
            return True
        else:
            logger.warning(f"Cannot update session '{session_id}': Session not found.")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a session state from memory.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if the session was found and deleted, False otherwise.
        """
        if session_id in self.sessions:
            logger.info(f"Deleting session state for ID: {session_id}")
            del self.sessions[session_id]
            return True
        else:
            logger.warning(f"Cannot delete session '{session_id}': Session not found.")
            return False

    def get_active_history(self, session_id: str) -> Optional[List[Dict]]:
        """Gets the active history for a session."""
        session = self.get_session(session_id)
        return session.get("active_history") if session else None

    def set_active_history(self, session_id: str, history: List[Dict]) -> bool:
        """Sets the active history for a session."""
        return self.update_session_state(session_id, "active_history", history)

    def get_last_summary_index(self, session_id: str) -> int:
        """Gets the last summary index for a session."""
        session = self.get_session(session_id)
        return session.get("last_summary_turn_index", -1) if session else -1

    def set_last_summary_index(self, session_id: str, index: int) -> bool:
        """Sets the last summary index for a session."""
        return self.update_session_state(session_id, "last_summary_turn_index", index)

    def get_previous_input_messages(self, session_id: str) -> Optional[List[Dict]]:
        """Gets the previous input messages for a session."""
        session = self.get_session(session_id)
        return session.get("previous_input_messages") if session else None

    def set_previous_input_messages(self, session_id: str, messages: List[Dict]) -> bool:
        """Sets the previous input messages for a session."""
        return self.update_session_state(session_id, "previous_input_messages", messages)

    def get_user_valves(self, session_id: str) -> Optional[Any]:
        """Gets the stored UserValves object for a session."""
        session = self.get_session(session_id)
        return session.get("user_valves") if session else None

    def set_user_valves(self, session_id: str, user_valves: Any) -> bool:
        """Sets the UserValves object for a session."""
        return self.update_session_state(session_id, "user_valves", user_valves)

    def get_rag_cache(self, session_id: str) -> Optional[str]:
        """Gets the stored RAG cache content for a session."""
        session = self.get_session(session_id)
        return session.get("rag_cache") if session else None

    def set_rag_cache(self, session_id: str, cache_content: Optional[str]) -> bool:
        """Sets the RAG cache content for a session."""
        return self.update_session_state(session_id, "rag_cache", cache_content)
