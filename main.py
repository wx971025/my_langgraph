import os
import asyncio
import logging
import threading
from IPython.display import Image, display
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv; load_dotenv(".env")

from models.chat_models import AzureChatOpenAI, AzureModelName
from utils import draw_graph_image

logger = logging.getLogger("utils")

# 定义大模型实例
llm = AzureChatOpenAI(model=AzureModelName.gpt_5_mini)


async def main():
    # 定义状态模式
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # 定义大模型交互节点
    def call_model(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": response}

    # 定义翻译节点
    def translate_message(state: State):
        system_prompt = """
        Please translate the received text in any language into English as output
        """
        messages = state['messages'][-1]
        messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=messages.content)]
        # messages = state['messages'] + [HumanMessage(content="我叫什么")]
        response = llm.invoke(messages)
        return {"messages": response}

    # 构建状态图
    builder = StateGraph(State)

    # 向图中添加节点
    builder.add_node("call_model", call_model)
    builder.add_node("translate_message", translate_message)

    # 构建边
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "translate_message")
    builder.add_edge("translate_message", END)

    # 编译图
    memory_saver = MemorySaver()
    simple_short_graph = builder.compile(checkpointer=memory_saver)

    draw_graph_image(simple_short_graph)

    
    thread_id = threading.get_ident()
    logger.info(f"Current thread id: {thread_id}")
    config = {"configuration": {"thread_id": thread_id}}
    for chunk in simple_short_graph.stream({"messages": ["你好，我叫木羽"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()


    for chunk in simple_short_graph.stream({"messages": ["请问我叫什么？"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
