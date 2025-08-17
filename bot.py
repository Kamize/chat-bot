import os
import streamlit as st
from pprint import pprint
from random import randint
from typing import Annotated, Literal
from typing_extensions import TypedDict
from collections.abc import Iterable

import google.generativeai as genai
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Import tools
from tools import get_menu, add_to_order, confirm_order, get_order, clear_order, place_order
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
# genai.configure(api_key=GEMINI_API_KEY)

class OrderState(TypedDict):
    """State representing the customer's order conversation."""
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool

BARISTABOT_SYSINT = (
    "system",
    "You are a BaristaBot, an interactive cafe ordering system. A human will talk to you about the "
    "available products you have and you will answer any questions about menu items (and only about "
    "menu items - no off-topic discussion, but you can chat about the products and their history). "
    "The customer will place an order for 1 or more items from the menu, which you will structure "
    "and send to the ordering system after confirming the order with the human. "
    "\n\n"
    "Add items to the customer's order with add_to_order, and reset the order with clear_order. "
    "To see the contents of the order so far, call get_order (this is shown to you, not the user) "
    "Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will "
    "display the order items to the user and returns their response to seeing the list. Their response may contain modifications. "
    "Always verify and respond with drink and modifier names from the MENU before adding them to the order. "
    "If you are unsure a drink or modifier matches those on the MENU, ask a question to clarify or redirect. "
    "You only have the modifiers listed on the menu. "
    "Once the customer has finished ordering items, Call confirm_order to ensure it is correct then make "
    "any necessary updates and then call place_order. Once place_order has returned, thank the user and "
    "say goodbye!",
)
WELCOME_MSG = "Welcome to the BaristaBot cafe. Type `q` to quit. How may I serve you today?"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")  
    )

def chatbot_with_tools(state: OrderState) -> OrderState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"order": [], "finished": False}
    # Convert BARISTABOT_SYSINT into a SystemMessage
    system_msg = SystemMessage(content=BARISTABOT_SYSINT[1])
    # Pass all messages in the state as history to the LLM
    message_history = [system_msg] + state["messages"]
    new_output = llm_with_tools.invoke(message_history)
    return defaults | state | {"messages": [new_output]}


def order_node(state: OrderState) -> OrderState:
    """The ordering node. This is where the order state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    order = state.get("order", [])
    outbound_msgs = []
    order_placed = False

    for tool_call in tool_msg.tool_calls:
        if tool_call["name"] == "add_to_order":
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"
            order.append(f'{tool_call["args"]["drink"]} ({modifier_str})')
            response = "\n".join(order)
        elif tool_call["name"] == "confirm_order":
            # In a Streamlit app, you would handle this confirmation via a new chat turn.
            # The LLM's confirm_order tool call is handled by the graph which returns control
            # to the LLM to ask the user for confirmation.
            response = "I've received your confirmation request."
        elif tool_call["name"] == "get_order":
            response = "\n".join(order) if order else "(no order)"
        elif tool_call["name"] == "clear_order":
            order.clear()
            response = "Order cleared."
        elif tool_call["name"] == "place_order":
            order_text = "\n".join(order)
            print("Sending order to kitchen:", order_text)
            order_placed = True
            response = str(randint(1, 5))
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outbound_msgs, "order": order, "finished": order_placed}

def maybe_route_to_tools(state: OrderState) -> str:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")
    msg = msgs[-1]
    if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        return "tools" if any(tool["name"] in auto_tools_names for tool in msg.tool_calls) else "ordering"
    return END if state.get("finished", False) else "chatbot"

# Auto-tools will be invoked automatically by the ToolNode
auto_tools = [get_menu]
auto_tools_names = [tool.name for tool in auto_tools]
tool_node = ToolNode(auto_tools)
# Order-tools will be handled by the order node.
order_tools = [add_to_order, confirm_order, get_order, clear_order, place_order]
# The LLM needs to know about all of the tools.
llm_with_tools = llm.bind_tools(auto_tools + order_tools)

graph_builder = StateGraph(OrderState)
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ordering", order_node)

graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("ordering", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_with_order_tools = graph_builder.compile()
