import operator
import logging
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import Tool
from typing import Annotated, List

from models.chat_models import AzureChatOpenAI, AzureModelName
from tools.fake_base1_tools import (
    add_sale,
    delete_sale,
    update_sale,
    query_sales
)

logger = logging.getLogger("utils")

class DBAgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)


class DBAgent:
    def __new__(cls, *args, **kwargs) -> CompiledStateGraph:
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance.agent

    def __init__(self,
        chat_llm: BaseChatModel = None, 
        tools_map: dict = None,
        system_prompt: str = None
    ):
        super().__init__()
        self.tools_map = tools_map or {
            "add_sale": add_sale,
            "delete_sale": delete_sale,
            "update_sale": update_sale,
            "query_sales": query_sales
        }
        self.tools = list(self.tools_map.values())
        self.chat_llm = chat_llm or AzureChatOpenAI(model=AzureModelName.gpt_5_mini)
        self.chat_llm_with_tools = self.chat_llm.bind_tools(self.tools)
        self.system_prompt = (
            system_prompt or 
            "You are a database specialist. You can query, add, update, or delete sales data. "
            "If the user asks for something outside your scope (like data analysis, plotting, or general coding), "
            "explicitly state that you cannot do it and return your findings so far. "
            "Do NOT ask the user for clarification if the request is simply out of your scope."
        )
    
    def chat_node(self, state: DBAgentState) -> DBAgentState:
        messages = state.messages
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.chat_llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def router_node(self, state: DBAgentState) -> DBAgentState:
        message = state.messages[-1]
        if message.tool_calls:
            return "tool_execute"
        else:
            return "chat"

    def tool_execute_node(self, state: DBAgentState) -> DBAgentState:
        ai_message = state.messages[-1]

        tool_messages = []

        for tool_call in ai_message.tool_calls:
            tool_call_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = self.tools_map[tool_name]
            try:
                result = tool.invoke(tool_args)
            except Exception as e:
                result = f"Tool {tool_name} execution failed with error: {e}"
            tool_message = ToolMessage(content=result, name=tool_name, tool_call_id=tool_call_id)
            tool_messages.append(tool_message)
        return {"messages": tool_messages}
    
    @property
    def agent(self) -> StateGraph:
        db_agent_builder = StateGraph(DBAgentState)
        db_agent_builder.add_node("chat", self.chat_node)
        db_agent_builder.add_node("tool_execute", self.tool_execute_node)
        db_agent_builder.add_conditional_edges(
            "chat",
            self.router_node,
            {
                "tool_execute": "tool_execute",
                "chat": END
            }
        )
        db_agent_builder.add_edge("tool_execute", "chat")
        db_agent_builder.set_entry_point("chat")


        db_agent = db_agent_builder.compile()
        return db_agent
