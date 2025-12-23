import os
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from enum import StrEnum
from pydantic import PrivateAttr
from typing import (
    Any, List, Dict, 
    cast, override,
    Iterator, AsyncIterator
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    BaseMessage, 
    AIMessage, 
    AIMessageChunk,
    convert_to_messages,
)
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)

from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI as BaseAzureChatOpenAI


class AzureModelName(StrEnum):
    o1 = "o1"
    o1_mini = "o1-mini"
    o3_mini = "o3-mini"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1_nano = "gpt-4.1-nano"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"
    gpt_5_chat = "gpt-5-chat"


class AzureChatOpenAI(BaseAzureChatOpenAI):
    def __init__(self, model: AzureModelName, **kwargs):
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        super().__init__(
            azure_deployment=model, 
            api_key=api_key,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            **kwargs
        )

class GeminiModelName(StrEnum):
    gemini_3_pro_image_preview = "google/gemini-3-pro-image-preview"
    gemini_2_5_flash_image = "google/gemini-2-5-flash-image"


class OpenrouterChatOpenAI(BaseChatModel):
    _client: OpenAI = PrivateAttr()
    _async_client: AsyncOpenAI = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(self, model: GeminiModelName, **kwargs):
        super().__init__(**kwargs)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=api_key, 
        )
        self._async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=api_key, 
        )
        self._model = model

    def _convert_input_to_gemini_messages(self, input: LanguageModelInput) -> List[Dict]:
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
            
        raw_messages = []
        if hasattr(input, "to_messages"):
            raw_messages = input.to_messages()
        elif isinstance(input, list):
            raw_messages = input
        else:
            raw_messages = convert_to_messages(input)

        final_messages = []
        for message in raw_messages:
            if isinstance(message, dict):
                final_messages.append(message)
            elif isinstance(message, BaseMessage):
                match message.type:
                    case "human":
                        role = "user"
                    case "ai":
                        role = "assistant"
                    case "system":
                        role = "system"
                    case "tool":
                        role = "tool"
                    case "chat":
                        role = getattr(message, "role", "user")
                    case _:
                        role = "user"
                final_messages.append({"role": role, "content": message.content})
            else:
                raise ValueError(f"Unrecognized message element: {type(message)}")
        return final_messages

    @override
    def _generate(self, 
        messages: list[BaseMessage], 
        stop: list[str] | None = None, 
        run_manager: CallbackManagerForLLMRun | None = None, 
        **kwargs: Any
    ) -> ChatResult:
        formatted_messages = self._convert_input_to_gemini_messages(messages)

        aspect_ratio = kwargs.pop("aspect_ratio", None)
        resolution = kwargs.pop("resolution", None)
        
        extra_body = {
            "modalities": ["image", "text"]
        }

        if aspect_ratio is not None and \
            aspect_ratio not in ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:4", "4:5"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            aspect_ratio = "1:1"
        
        if resolution is not None and resolution not in ["1k", "2k"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            resolution = "1k"
        
        image_config = {}
        if resolution:
            image_config["image_size"] = resolution.upper()
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        if image_config:
            extra_body["image_config"] = image_config

        response = self._client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            stop=stop,
            **kwargs
        )
        response = cast(ChatCompletion, response)

        message = response.choices[0].message

        content = message.content
        reasoning = ""
        reasoning_details = []
        if hasattr(message, "reasoning"):
            reasoning = message.reasoning
            reasoning_details = message.reasoning_details

        images = []
        if hasattr(message, "images"):
            images = message.images

        if hasattr(response, "usage"):
            usage = response.usage.model_dump()

        ai_message = AIMessage(
            content = content, 
            reasoning = reasoning, 
            reasoning_details = reasoning_details,
            images = images, 
            usage = usage
        )
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
    
    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any
    ) -> ChatResult:
        formatted_messages = self._convert_input_to_gemini_messages(messages)

        aspect_ratio = kwargs.pop("aspect_ratio", None)
        resolution = kwargs.pop("resolution", None)
        
        extra_body = {
            "modalities": ["image", "text"]
        }

        if aspect_ratio is not None and \
            aspect_ratio not in ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:4", "4:5"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            aspect_ratio = "1:1"
        
        if resolution is not None and resolution not in ["1k", "2k"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            resolution = "1k"
        
        image_config = {}
        if resolution:
            image_config["image_size"] = resolution.upper()
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        if image_config:
            extra_body["image_config"] = image_config

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            stop=stop,
            **kwargs
        )
        response = cast(ChatCompletion, response)
        message = response.choices[0].message
        content = message.content
        reasoning = ""
        reasoning_details = []
        if hasattr(message, "reasoning"):
            reasoning = message.reasoning
            reasoning_details = message.reasoning_details

        images = []
        if hasattr(message, "images"):
            images = message.images

        if hasattr(response, "usage"):
            usage = response.usage.model_dump()

        generation = ChatGeneration(
            message=AIMessage(
                content=content,
                reasoning=reasoning,
                reasoning_details=reasoning_details,
                images=images,
                usage=usage
            )
        )
        return ChatResult(generations=[generation])

    @override
    def _stream(self, 
        messages: list[BaseMessage], 
        stop: list[str] | None = None, 
        run_manager: CallbackManagerForLLMRun | None = None, 
        **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        formatted_messages = self._convert_input_to_gemini_messages(messages)

        aspect_ratio = kwargs.pop("aspect_ratio", None)
        resolution = kwargs.pop("resolution", None)
        
        extra_body = {
            "modalities": ["image", "text"]
        }

        if aspect_ratio is not None and \
            aspect_ratio not in ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:4", "4:5"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            aspect_ratio = "1:1"
        
        if resolution is not None and resolution not in ["1k", "2k"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            resolution = "1k"
        
        image_config = {}
        if resolution:
            image_config["image_size"] = resolution.upper()
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        if image_config:
            extra_body["image_config"] = image_config

        response = self._client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            stop=stop,
            stream=True,
            **kwargs
        )
        response = cast(ChatCompletion, response)
        for chunk in response:
            chunk = cast(ChatCompletionChunk, chunk)
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason
            delta = choice.delta

            content = delta.content
            reasoning = getattr(delta, "reasoning", None)
            reasoning_details = getattr(delta, "reasoning_details", None)

            images = getattr(delta, "images", [])

            usage = None
            if hasattr(chunk, "usage"):
                usage = chunk.usage.model_dump() if hasattr(chunk.usage, "model_dump") else chunk.usage
            
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content,
                    images=images,
                    reasoning=reasoning,
                    reasoning_details=reasoning_details,
                    finish_reason=finish_reason,
                    usage=usage,
                ),
            )

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any
    ) ->AsyncIterator[ChatGenerationChunk]:
        formatted_messages = self._convert_input_to_gemini_messages(messages)

        aspect_ratio = kwargs.pop("aspect_ratio", None)
        resolution = kwargs.pop("resolution", None)
        
        extra_body = {
            "modalities": ["image", "text"]
        }

        if aspect_ratio is not None and \
            aspect_ratio not in ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:4", "4:5"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            aspect_ratio = "1:1"
        
        if resolution is not None and resolution not in ["1k", "2k"]:
            # 设置了, 但是没在允许的范围内，则使用默认值
            resolution = "1k"
        
        image_config = {}
        if resolution:
            image_config["image_size"] = resolution.upper()
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        if image_config:
            extra_body["image_config"] = image_config

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            stop=stop,
            stream=True,
            extra_body=extra_body,
            **kwargs
        )
        async for chunk in response:
            chunk = cast(ChatCompletionChunk, chunk)
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason
            delta = choice.delta

            content = delta.content
            reasoning = getattr(delta, "reasoning", None)
            reasoning_details = getattr(delta, "reasoning_details", None)

            images = getattr(delta, "images", [])

            usage = None
            if hasattr(chunk, "usage"):
                usage = chunk.usage.model_dump() if hasattr(chunk.usage, "model_dump") else chunk.usage

            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content,
                    images=images,
                    reasoning=reasoning,
                    reasoning_details=reasoning_details,
                    finish_reason=finish_reason,
                    usage=usage,
                ),
            )

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat-openai"
