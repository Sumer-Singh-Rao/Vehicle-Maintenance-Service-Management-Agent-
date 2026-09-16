"""
agent/orchestrator.py - LangGraph Agent Orchestrator.
Exposes VehicleMaintenanceAgent using compiled LangGraph StateGraph engine.
"""
import re
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage

from config import GROQ_MODEL, MAX_TOOL_CALLS, GROQ_API_KEY
from agent.graph import build_agent_graph


def sanitize_api_error(err: Any) -> str:
    """Format API error messages cleanly without exposing sensitive info."""
    err_str = str(err)
    if "429" in err_str or "rate limit" in err_str.lower():
        return "AI rate limit reached. Please wait a moment and try again."
    if "401" in err_str or "invalid_api_key" in err_str.lower():
        return "Authentication error: Invalid GROQ_API_KEY."
    return re.sub(r"(gsk_[a-zA-Z0-9]+|sk-[a-zA-Z0-9]+)", "[REDACTED]", err_str)


class VehicleMaintenanceAgent:
    """Agent that handles vehicle maintenance reasoning using LangGraph & Groq API."""

    def __init__(self):
        self.model = GROQ_MODEL
        self.max_tool_calls = MAX_TOOL_CALLS
        self.graph = build_agent_graph(GROQ_API_KEY)

    def is_configured(self) -> bool:
        """Check if Groq API & LangGraph agent workflow are ready."""
        return self.graph is not None

    def run(self, user_message: str, history: Optional[List[Dict[str, str]]] = None, selected_vehicle_id: Optional[int] = None) -> str:
        """Execute reasoning loop using compiled LangGraph StateGraph."""
        if not self.is_configured():
            return f"Please set GROQ_API_KEY in .env to enable the AI assistant with `{self.model}`."

        # Convert Streamlit history format to LangChain messages
        langchain_messages = []
        if history:
            for item in history:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    langchain_messages.append(AIMessage(content=content))

        langchain_messages.append(HumanMessage(content=user_message))

        initial_state = {
            "messages": langchain_messages,
            "user_id": 1,
            "selected_vehicle_id": selected_vehicle_id,
            "pending_confirmation": False,
            "pending_action": None,
            "booking_details": None,
        }

        try:
            # Set recursion limit to enforce tool iteration ceiling
            recursion_limit = (self.max_tool_calls * 2) + 2
            config = {"recursion_limit": recursion_limit}
            final_state = self.graph.invoke(initial_state, config=config)

            messages = final_state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    return str(last_msg.content or "Task completed.")
                return str(getattr(last_msg, "content", "Done."))
            return "No response generated."
        except Exception as e:
            err_msg = str(e)
            if "recursion" in err_msg.lower():
                return f"The workflow reached the maximum allowed limit of {self.max_tool_calls} tool calls. Please refine your request with specific vehicle or location details."
            return f"Groq API Error ({self.model}): {sanitize_api_error(e)}. No model fallback configured."
