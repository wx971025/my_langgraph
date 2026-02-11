import os
import json
import time
import hmac
import base64
import urllib.parse
import hashlib
import requests
from typing import Dict, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils import logger
from utils.decorator import time_record


class XinzhiWeatherToolSchema(BaseModel):
    city: str = Field(..., description="The city to get the weather, Chinese city name")

class XinzhiWeatherTool:
    _pubilc_key = os.environ["XINZHI_API_PUBILC_KEY"]
    _private_key = os.environ["XINZHI_API_PRIVATE_KEY"]
    
    @classmethod
    def get_weather(cls, city: str):
        data = {
            "language": "zh-Hans",
            "location": city,
            "public_key": cls._pubilc_key,
            "ts": round(time.time()),
            "ttl": 10,
            "unit": "c"
        }
        data_list = []
        for k, v in data.items():
            data_list.append(f"{k}={v}")
        data_res = "&".join(data_list)


        hmac_sha1 = hmac.new(cls._private_key.encode('utf-8'), data_res.encode('utf-8'), hashlib.sha1).digest()
        base64_encoded = base64.b64encode(hmac_sha1).decode('utf-8')
        url_encoded_sig = urllib.parse.quote(base64_encoded)

        data_res += f"&sig={url_encoded_sig}"

        responses = requests.get(f"https://api.seniverse.com/v3/weather/now.json?{data_res}")
        
        return responses.json()


@tool("xinzhi_weather", args_schema=XinzhiWeatherToolSchema)
@time_record
def xinzhi_weather_tool(
    city: str,
) -> str:
    """Get the weather of a city, This tool can only obtain real-time weather conditions and cannot access future weather forecasts.
    Args:
        city: The city to get the weather, Chinese city name
    Returns:
        The weather of the city
    """
    result: Dict = XinzhiWeatherTool.get_weather(city)
    return json.dumps(result, ensure_ascii=False, indent=4)


class OpenWeatherToolSchema(BaseModel):
    city: str = Field(..., description="The city to get the weather, English city name")

class OpenWeatherTool:
    _api_key = os.environ["OPENWEATHER_API_KEY"]

    @classmethod
    def get_current_weather(cls, city: str) -> Optional[Dict]:
        """Get the weather of a city

        Args:
            city (str): The city to get the weather, English city name

        Raises:
            requests.exceptions.RequestException: HTTP Request Error
            Exception: Error getting weather for city

        Returns:
            Optional[Dict]: The weather of the city
        """
        try:
            result = cls.geocoding(city)
            if result is None:
                logger.error(f"No latitude or longitude found for city: {city}")
                return None
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": result["lat"],
                "lon": result["lon"],
                "appid": cls._api_key,
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            response_json = response.json()
            return response_json
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request Error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting weather for city: {city}: {str(e)}")
            return None

    @classmethod
    def get_forecast_weather(cls, city: str) -> Optional[Dict]:
        """Get the forecast weather of a city
        Args:
            city (str): The city to get the forecast weather, English city name
        Returns:
            Optional[Dict]: The forecast weather of the city
        """
        url = "https://api.openweathermap.org/data/2.5/forecast"
        try:
            result = cls.geocoding(city)
            if result is None:
                logger.error(f"No latitude or longitude found for city: {city}")
                raise Exception(f"No latitude or longitude found for city: {city}")

            params = {
                "lat": result["lat"],
                "lon": result["lon"],
                "appid": cls._api_key,
                "mode": "json",
                "units": "standard",
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            response_json = response.json()
            return response_json
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request Error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return None


    @classmethod
    def geocoding(cls, city: str) -> Optional[Dict]:
        """Get the latitude and longitude of a city

        Args:
            city (str): The city to get the latitude and longitude, English city name
        Returns:
            Optional[Dict]: The latitude and longitude of the city
        """
        logger.info(f"Getting lat and lon for city: {city}")
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        try:
            response = requests.get(url, params={"q": city, "limit": 1, "appid": cls._api_key})
            response.raise_for_status()
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                response_item = response_json[0]
                lat, lon = response_item.get("lat", None), response_item.get("lon", None)

                if all([lat, lon]):
                    return {"lat": lat, "lon": lon}
                else:
                    logger.error(f"No latitude or longitude found for city: {city}")
                    return None
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request Error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return None


@tool("openweather_current_weather", args_schema=OpenWeatherToolSchema)
@time_record
def openweather_current_weather_tool(
    city: str,
) -> str:
    """Get the weather of a city, This tool can only obtain real-time weather conditions and cannot access future weather forecasts.
    Args:
        city: The city to get the weather, English city name
    Returns:
        The weather of the city
    """

    result: Optional[Dict] = OpenWeatherTool.get_current_weather(city)
    if result:
        return json.dumps(result, ensure_ascii=False, indent=4)
    else:
        return f"Error: No weather found for city: {city}. Please try again."


@tool("openweather_forecast_weather", args_schema=OpenWeatherToolSchema)
@time_record
def openweather_forecast_weather_tool(
    city: str,
) -> str:
    """Get the forecast weather of a city
    Args:
        city: The city to get the forecast weather, English city name
    Returns:
        The forecast weather of the city
    """
    result: Optional[Dict] = OpenWeatherTool.get_forecast_weather(city)
    if result:
        return json.dumps(result, ensure_ascii=False, indent=4)
    else:
        return f"Error: No forecast weather found for city: {city}. Please try again."



if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv(".env")
    from models.chat_models import ChatDeepSeek, ChatDeepSeekModelName
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    model = ChatDeepSeek(ChatDeepSeekModelName.deepseek_chat)
    tools = [openweather_current_weather_tool, openweather_forecast_weather_tool]
    tools_map = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    messages = [HumanMessage(content="北京这几天的天气情况都咋样")]
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
    