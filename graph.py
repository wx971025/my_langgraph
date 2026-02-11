from dotenv import load_dotenv; load_dotenv(".env")
import os
import sys
import json
import asyncio
import uuid
import argparse
import rich
import re
import tiktoken
from traceback import format_exc
from contextlib import AsyncExitStack
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamWriter, RunnableConfig, Command, Interrupt
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import Tool
from langchain_core.messages import (
    AnyMessage, 
    BaseMessage,
    HumanMessage, 
    AIMessage, 
    AIMessageChunk,
    RemoveMessage,
    ToolMessage, 
    SystemMessage
)
from typing import TypedDict, Annotated, List, cast, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from tools import (
    tavily_search_tool,
    xinzhi_weather_tool,
    openweather_current_weather_tool,
    openweather_forecast_weather_tool,
    python_repl_tool,
    gpt_image_1_5_generation_tool,
)
from tools.mcp_tools import get_local_mcp_tools
from models.chat_models import (
    AzureChatOpenAI, 
    AzureChatOpenAIModelName,
    ChatDeepSeek, 
    ChatDeepSeekModelName, 
)
from utils import logger, detect_language


def add_images(left: Dict[str, str], right: Dict[str, str] | None) -> Dict[str, str]:
    if not isinstance(right, dict):
        return left or {}
    return {**(left or {}), **right}

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    system_message: str = Field(default="")
    user_approved: Dict[str, Literal['y', 'n']] = Field(default_factory=dict)
    image_register: Annotated[Dict[str, str], add_images] = Field(default_factory=dict)


class SummarizationMiddleware(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    system_message: str = Field(default="")
    summary_messages: List[AnyMessage] = Field(default_factory=list)


def _replace_messages(msgs: List[AnyMessage]) -> list:
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *msgs]

def _get_messages_tokens(messages: List[AnyMessage]) -> int:
    tokenizer = tiktoken.get_encoding("o200k_base")
    contents = []
    for message in messages:
        if isinstance(message, BaseMessage):
            content = message.content
            if isinstance(content, list):
                content = "".join([item.get("text", "") for item in content])
            elif isinstance(content, str):
                pass
            contents.append(content)
    return len(tokenizer.encode("".join(contents)))

def summarization_middleware():
    _summarization_token_threhold = 20000
    async def summarization_node(state: SummarizationMiddleware) -> SummarizationMiddleware:
        messages = state["messages"]
        system_message = state["system_message"]

        if not messages[-1].type == "human":
            logger.warning(f"Last message is not a human message, skipping summarization")
            return {"summary_messages": messages}

        content_tokens = _get_messages_tokens(messages)
        
        if content_tokens > _summarization_token_threhold:
            logger.info(f"Input prompt is too long, summarizing...")

            user_message = messages.pop(-1)
            user_messages_token = _get_messages_tokens([user_message])

            model = ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat)
            summarization_messages = [
                SystemMessage(content="你是一个专业的摘要助手，请将以下对话历史进行摘要，保留关键决策点和技术细节, 不要太长, 保留关键信息即可"),
                *messages[:-1],
                HumanMessage(content="请将以上所有对话历史进行摘要，保留关键决策点和关键信息, 不要超过2000个token"),
            ]
            response = await model.ainvoke(summarization_messages)
            response = cast(AIMessage, response)

            summarized_system_message = response.content + "\n\n" + system_message

            logger.info(f"Summarizing system messages tokens number is: {_get_messages_tokens([system_message])}")
            new_messages = []
            all_token_count = 0
            for message in messages[::-1]:
                if hasattr(message, "content") and message.content:
                    token_count = _get_messages_tokens([message])
                    all_token_count += token_count
                    if all_token_count > _summarization_token_threhold - user_messages_token - 3000:
                        break
                new_messages.append(message)

            new_messages = new_messages[::-1]
            new_messages.append(user_message)

            summarization_tokens = _get_messages_tokens(new_messages)
            logger.info(f"Summarized input tokens: {summarization_tokens}")

            return {"summary_messages": new_messages, "system_message": summarized_system_message}
        else:
            return {"summary_messages": messages, "system_message": system_message}

    builder = StateGraph(SummarizationMiddleware)
    builder.add_node("summarization", summarization_node)
    builder.add_edge("summarization", END)
    builder.set_entry_point("summarization")
    return builder.compile()


def build_agent(
    model: BaseChatModel, 
    tools: List[Tool] = None, 
    checkpointer: BaseCheckpointSaver = None,
    store: BaseStore = None
):
    builder = StateGraph(AgentState)

    async def preprocessing_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        system_message = state["system_message"]
    
        image_register = {}
        image_records = []

        for message_idx, message in enumerate(messages, start=1):
            if isinstance(message, HumanMessage):
                content = message.content
                if isinstance(content, list):
                    contents = content
                    for content_item in contents:
                        if content_item.get("type") == "image":
                            image_id = f"IMG_{uuid.uuid4().hex[:8].upper()}"
                            image_url = content_item.get("url")
                            image_register[image_id] = image_url
                            image_records.append({
                                "id": image_id,
                                "type": "用户上传",
                                "message_index": message_idx,
                                "url": image_url
                            })
            elif isinstance(message, (AIMessage, AIMessageChunk)):
                content = message.content
                if isinstance(content, str):
                    # 改进的正则表达式：匹配 URL，但排除末尾的常见标点符号
                    pattern = r"https?://[^\s<>\"'`\)\]\}]+"
                    matches = re.findall(pattern, content)
                    for match in matches:
                        image_id = f"IMG_{uuid.uuid4().hex[:8].upper()}"
                        image_register[image_id] = match
                        image_records.append({
                            "id": image_id,
                            "type": "AI生成",
                            "message_index": message_idx,
                            "url": match
                        })

        if image_records:
            system_message_content = "\n\n" + "="*60 + "\n"
            system_message_content += "📸 对话中的图片索引表（内部使用）\n"
            system_message_content += "="*60 + "\n\n"
            system_message_content += "⚠️ 重要：图片ID是内部标识符，仅供你理解图片引用关系，绝对不要在任何回复中显示给用户！\n\n"
            system_message_content += "使用规则：\n"
            system_message_content += "1. 图片ID（IMG_XXXXXXXX）仅用于你内部识别图片，不要泄露给用户\n"
            system_message_content += "2. 当用户提到图片时（如\"第一张图\"、\"刚才的图片\"），你需要在内部使用对应的图片ID来理解\n"
            system_message_content += "3. 在回复用户时，使用自然语言描述图片（如\"第一张图片\"、\"你上传的图片\"），不要显示图片ID\n"
            system_message_content += "4. 图片ID格式为：IMG_XXXXXXXX（大写字母和数字），区分大小写\n\n"
            system_message_content += "-"*60 + "\n"
            system_message_content += "| 序号 | 图片ID（内部） | URL| 来源 | 出现位置 |\n"
            system_message_content += "|------|----------------|------|------|----------|\n"
            
            for idx, record in enumerate(image_records, 1):
                system_message_content += (
                    f"| {idx} | {record['id']} | {record['url']} | {record['type']} | "
                    f"第{record['message_index']}条消息 |\n"
                )
            
            system_message_content += "-"*60 + "\n\n"
            system_message_content += "理解示例（仅用于你内部理解，不要告诉用户）：\n"
            first_image_id = image_records[0]['id'] if image_records else "IMG_XXXXXXXX"
            system_message_content += f"- 用户说\"看第一张图片\" → 你内部识别为 {first_image_id}，但回复时只说\"第一张图片\"\n"
            system_message_content += "- 用户说\"刚才那张图\" → 你根据上下文判断对应的图片ID，但回复时用自然语言描述\n"
            system_message_content += "- ❌ 错误示例：回复中出现 \"IMG_XXXXXXXX\" 或任何图片ID格式\n"
            system_message_content += "- ✅ 正确示例：回复中使用 \"第一张图片\"、\"你上传的图片\"、\"刚才显示的图片\" 等自然语言\n"
            system_message_content += "="*60 + "\n"
            
            system_message += system_message_content
        
        return {"system_message": system_message, "image_register": image_register}


    async def call_summarization_middleware(state: AgentState) -> dict:
        subgraph_input = {
            "messages": state["messages"], 
            "system_message": state["system_message"], 
            "summary_messages": []
        }
        summarization_agent = await summarization_middleware().ainvoke(subgraph_input)
        summary_messages = summarization_agent.get("summary_messages", [])
        summary_system_message = summarization_agent.get("system_message", "")
        return {
            "messages": _replace_messages(summary_messages), 
            "system_message": summary_system_message
        }

    async def indent_recognition_node(
        state: AgentState, config: RunnableConfig, store: BaseStore, writer: StreamWriter
    ) -> AgentState:
        node_metadata = {"langgraph_node": "indent_recognition_node"}
        model_with_tools = model.bind_tools(tools or [])

        messages = state["messages"]
        system_message = state["system_message"]

        full_response: str | None = None
        async for chunk in model_with_tools.astream(
            [SystemMessage(content=system_message)] + messages
        ):
            chunk = cast(AIMessageChunk, chunk)

            if full_response is None:
                full_response = chunk
            else:
                full_response += chunk

            if hasattr(chunk, "content") and chunk.content:
                writer((chunk, node_metadata))
        print('\n')
        return {"messages": [full_response]}

    def router_node(state: AgentState) -> str:
        node_metadata = {"langgraph_node": "router_node"}
        messages = state["messages"]
        last_message: AnyMessage = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools_execution"
        else:
            return END

    async def tools_execution_node(
        state: AgentState, config: RunnableConfig, store: BaseStore, writer: StreamWriter
    ) -> AgentState:
        node_metadata = {"langgraph_node": "tools_execution_node"}

        messages: List[AnyMessage] = state["messages"]
        user_approved: Dict[str, Literal['y', 'n']] = state["user_approved"]
        tools_map = {tool.name: tool for tool in tools}
        tool_calls = messages[-1].tool_calls

        logger.info(f"======== Tool calls =========")
        for tool_call in tool_calls:
            tool_call_name = tool_call.get("name")
            tool_call_args = tool_call.get("args")
            tool_call_id = tool_call.get("id")

            if tool_call_name not in tools_map:
                logger.warning(f"Tool call name: {tool_call_name} not found in tools map")
                messages.append(
                    ToolMessage(
                        content=(
                            f"Tool call name: {tool_call_name} not found in tools list, "
                            f"please use another tool in tools list"
                        ),
                        name=tool_call_name, 
                        tool_call_id=tool_call_id
                    )
                )
                continue
            
            if user_approved.get(tool_call_name, 'y') == 'n':
                # 用户手动拒绝执行
                logger.info(f"User rejected the tool call: {tool_call_name}")
                messages.append(
                    ToolMessage(
                        content=f"用户主动拒绝了对[{tool_call_name}]工具的调用, 但这并不表示该工具不可用.", 
                        name=tool_call_name, 
                        tool_call_id=tool_call_id
                    )
                )
                continue
            
            if tool_call_name == "gpt_image_1_5_generation" and tool_call_args.get("image_id"):
                image_id = tool_call_args.get("image_id")
                if image_id.startswith("IMG_"):
                    tool_call_args["image_id"] = state["image_register"].get(image_id)
                elif image_id.startswith("http"):
                    tool_call_args["image_id"] = image_id

            tool = tools_map[tool_call_name]
            logger.info(f"\nTool call name: {tool_call_name}, \nTool call args: {json.dumps(tool_call_args, ensure_ascii=False, indent=4)}")

            if hasattr(tool, 'ainvoke'):
                content = await tool.ainvoke(tool_call_args)
            else:
                content = await asyncio.to_thread(tool.invoke, tool_call_args)

            display_content = str(content)[:100] + "..." if len(content) > 100 else content
            logger.info(f"\n[{tool_call_name}] tool call result: {display_content}\n\n")

            messages.append(ToolMessage(content=content, name=tool_call_name, tool_call_id=tool_call_id))
        return {"messages": messages}

    builder.add_node("preprocessing", preprocessing_node)
    builder.add_node("summarization_middleware", call_summarization_middleware)
    builder.add_node("indent_recognition", indent_recognition_node)
    builder.add_node("tools_execution", tools_execution_node)

    builder.add_edge("preprocessing", "summarization_middleware")
    builder.add_edge("summarization_middleware", "indent_recognition")
    builder.add_conditional_edges(
        "indent_recognition", 
        router_node,
        ["tools_execution", END]
    )
    builder.add_edge("tools_execution", "indent_recognition")
    builder.set_entry_point("preprocessing")

    agent = builder.compile(
        checkpointer=checkpointer, 
        store=store,
        interrupt_before=['tools_execution'],
        interrupt_after=None,
    )

    return agent


async def main(thread_id: str):
    stack = AsyncExitStack()
    checkpointer = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string('db/memory/graph.db'))

    local_mcp_tools = await get_local_mcp_tools()
    tools: List[Tool] = [
        tavily_search_tool,
        xinzhi_weather_tool,
        openweather_current_weather_tool,
        openweather_forecast_weather_tool,
        python_repl_tool,
        gpt_image_1_5_generation_tool,
        *local_mcp_tools,
    ]
    tool_require_approval = [
        "gpt_image_1_5_generation", 
        "xinzhi_weather",
        "openweather_current_weather",
        "openweather_forecast_weather",
    ]

    model = ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat)
    model = AzureChatOpenAI(AzureChatOpenAIModelName.o1_mini)

    agent: CompiledStateGraph = build_agent(
        model=model, 
        tools=tools, 
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": thread_id
        },
    }

    state = await agent.aget_state(config)

    if state.values.get("messages"):
        messages = state.values["messages"]
        init_tokens_count = _get_messages_tokens(messages)
    else:
        init_tokens_count = 0
    logger.info(f"Initial tokens count: {init_tokens_count}")

    while True:
        user_input: str = input(f"[User {thread_id}]: ")
        if user_input.lower() in ["exit", "quit", "bye", "q"]:
            break

        if user_input.lower().strip() in ["clear", "cls"]:
            try:
                await checkpointer.adelete_thread(thread_id)
                logger.info(f"Thread [{thread_id}] memory cleared, please restart...")
            except Exception as e:
                logger.error(f"Error deleting thread [{thread_id}]: {format_exc()}")
            finally:
                break

        if user_input.strip() == '':
            content_blocks = [
                {
                    "type": "text", 
                    "text": "这副图里画了什么"
                },
                {
                    "type": "image",
                    "url": "https://surgepix-ai-1316642525.cos.ap-seoul.myqcloud.com/test/images/download/20260205_100407_862_b2c17e_watermarked.png"
                }
            ]
            logger.info(f"User input is empty, using default input: 这副图里画了什么")
        else:
            content_blocks = [
                {
                    "type": "text", 
                    "text": user_input
                }
            ]

        init_state = {
            "messages": [HumanMessage(content_blocks=content_blocks)],
            "user_approved": {},
            "image_register": {},
            "system_message": "你是一个专业的, 多功能的Agent"
        }

        while True:
            answer_flag = False
            async for chunk, metadata in agent.astream(
                input=init_state,
                config=config,
                stream_mode="custom",
            ):
                if not answer_flag:
                    answer_flag = True
                    print("\n\n[Answer]: ", end="", flush=True)
                chunk = cast(AIMessageChunk, chunk)
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            
            snapshot = await agent.aget_state(config)

            if snapshot.tasks and snapshot.tasks[0].name == "tools_execution":
                tool_calls: List[Dict[str, Any]] = snapshot.values["messages"][-1].tool_calls
                user_approved = {}
                for tool_call in tool_calls:
                    tool_call_name = tool_call.get("name")
                    tool_call_args = tool_call.get("args")
                    if tool_call_name in tool_require_approval:
                        while True:
                            tool_call_args_str = json.dumps(tool_call_args, ensure_ascii=False, indent=4)
                            approved: Literal['y', 'n'] = input(
                                f"[Tool approval] Do you approve the tool call: \nTool name: [{tool_call_name}]\nTool args: {tool_call_args_str}\n(y/n): "
                            ).lower().strip()
                            if approved not in ['y', 'n']:
                                logger.warning(f"Invalid approval: {approved}, please enter 'y' or 'n'")
                                continue
                            break
                        user_approved[tool_call.get("name")] = approved
                    else:
                        user_approved[tool_call.get("name")] = "y"
                await agent.aupdate_state(config, {'user_approved': user_approved})
                init_state = None
            else:
                break

    await stack.aclose()

class Args(BaseModel):
    thread_id: str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread_id", nargs='?', type=str, default='demo')
    args: Args = parser.parse_args()
    thread_id = args.thread_id
    asyncio.run(main(thread_id))
