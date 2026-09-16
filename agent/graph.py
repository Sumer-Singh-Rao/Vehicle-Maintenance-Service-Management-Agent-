"""
agent/graph.py - LangGraph StateGraph Workflow Engine.
Uses ChatGroq with openai/gpt-oss-120b and LangChain Tools.
"""
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from config import GROQ_MODEL, GROQ_API_KEY, MAX_TOOL_CALLS
from agent.state import AgentState
from agent.prompts import build_system_prompt
from agent.tools import ALL_TOOLS


def build_agent_graph(api_key: Optional[str] = None):
    """
    Construct compiled LangGraph StateGraph agent for VehicleCare AI.
    """
    key = api_key or GROQ_API_KEY
    if not key:
        return None

    try:
        llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=key,
            temperature=0.0
        )
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
    except Exception as e:
        print(f"Notice: Failed to initialize ChatGroq model: {e}")
        return None

    def agent_node(state: AgentState) -> Dict[str, Any]:
        """
        Agent reasoning node. Configures system prompt with active vehicle context
        and invokes the LLM with bound LangChain tools.
        """
        raw_msgs = list(state.get("messages", []))
        sys_prompt = build_system_prompt(state.get("selected_vehicle_id"))

        # Prepend or update SystemMessage at the start of conversation stream
        if not raw_msgs or not isinstance(raw_msgs[0], SystemMessage):
            messages = [SystemMessage(content=sys_prompt)] + raw_msgs
        else:
            messages = [SystemMessage(content=sys_prompt)] + raw_msgs[1:]

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()
