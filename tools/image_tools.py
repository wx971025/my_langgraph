import os
import json
import base64
import requests
import uuid
from traceback import format_exc
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal, Optional

from models.image_models import AzureImageGenerationModel
from utils import logger
from utils.tencent_cos import tencent_cos
from utils.decorator import time_record


class AzureImageGenerationSchema(BaseModel):
    prompt: str = Field(..., description="The prompt to generate image from.")
    image_id: Optional[str] = Field(None, description="The image id to edit. Not image url.")
    size: Literal["auto", "1536x1024", "1024x1024", "1024x1536"] = Field("1024x1024", description="The size of the generated image.")
    quality: Literal["auto", "low", "medium", "high"] = Field("medium", description="The quality of the generated image.")
    n: int = Field(1, description="The number of images to generate.")
    transparent: bool = Field(False, description="Whether to generate a transparent image.")

@tool("gpt_image_1_5_generation", args_schema=AzureImageGenerationSchema)
@time_record
async def gpt_image_1_5_generation_tool(
    prompt: str, 
    image_id: Optional[str] = None,
    size: str = "1024x1024", 
    quality: str = "medium", 
    n: int = 1, 
    transparent: bool = False,
) -> str:
    """
    Generate or edit an image using Azure OpenAI gpt-image-1.5 model.
    If the image_id is provided, the image will be edited.
    Otherwise, the image will be generated.
    please input the image id, not image url.
    example:
    ```
    {
        ...
        "image_id": "IMG_XXXXXXXX",
        ...
    }
    ```
    Return the markdown of the generated image url.
    """
    try:
        image_url = image_id
        image_model = AzureImageGenerationModel()
        image_flag, image_urls = await image_model.generate_image(
            prompt, image_url=image_url, size=size, quality=quality, transparent=transparent
        )
        if not image_flag:
            logger.warning(f"Error generating image: {image_urls}")
            return f"Error: No image data found in response. {image_urls}"
            
        return (
            f"Images generated successfully, "
            f"Please wrap the following link in markdown format and display it to the user."
            f"\n\n{'\n'.join(image_urls)}"
        )
    except Exception as e:
        logger.error(f"Error generating image: {format_exc()}")
        return "Error: No image data found in response. Please try again."
