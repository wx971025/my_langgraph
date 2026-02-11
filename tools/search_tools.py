import os
import json
import requests
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from utils import logger, detect_language
from utils.decorator import time_record

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=False,
    include_raw_content=False,
    include_images=False,
    include_image_descriptions=False,
    search_depth="basic",
)


class BingSearchTool:
    _api_key = os.environ["BING_SEARCH_API_KEY"]

    @classmethod
    def search(cls, query: str, count: int = 5) -> str:
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": cls._api_key
        }
        lang: Literal["zh", "en"] = detect_language(query)
        if lang == "zh":
            mkt = "zh-hans-CN"
            setLang = "zh-hans"
        else:
            mkt = "en-US"
            setLang = "en"
        params = {
            "q": query, 
            "answerCount": 1,
            "count": count,
            "mkt": mkt,
            "safeSearch": "Strict",
            "setLang": setLang,
        }
        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            response_json = response.json()
            return response_json
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request Error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return None



class BingSearchToolSchema(BaseModel):
    query: str = Field(..., description="The query to search for")
    count: int = Field(..., description="The number of results to return")



@tool("bing_search", args_schema=BingSearchToolSchema)
@time_record
def bing_search_tool(
    query: str,
    count: int = 5,
) -> str:
    """Use Bing Search API to search for news and hot topics

    Args:
        query (str): The query to search for
        count (int, optional): The number of results to return. Defaults to 5.

    Returns:
        str: The search results in JSON format
    
    .. deprecated:: 
        This function is deprecated and may be removed in a future version.
    """
    result = BingSearchTool.search(query, count)
    if result:
        return json.dumps(result, ensure_ascii=False, indent=4)
    else:
        return f"Error: No results found for query: {query}."


if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv(".env")
    from models.chat_models import ChatDeepSeek, ChatDeepSeekModelName
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    model = ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat)
    tools = [bing_search_tool]
    tools_map = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    messages = [HumanMessage(content="看下北京天气")]
    response_message: AIMessage = model_with_tools.invoke(messages)

    if response_message.content:
        print(response_message.content)

    while hasattr(response_message, "tool_calls") and response_message.tool_calls:
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")
            tool_id = tool_call.get("id")
            logger.info(f"Tool name: {tool_name}, \nTool args: {json.dumps(tool_args, ensure_ascii=False, indent=4)}")
            if tool_name not in tools_map:
                logger.warning(f"Tool name: {tool_name} not found in tools list")
                content = f"Error: Tool name: {tool_name} not found in tools list, Please use another tool"
            else:
                tool = tools_map[tool_name]
                content = tool.invoke(tool_args)
        messages.append(ToolMessage(content=content, name=tool_name, tool_call_id=tool_id))
        response_message = model_with_tools.invoke(messages)
        if response_message.content:
            print(response_message.content)
    
    print("Done!")
