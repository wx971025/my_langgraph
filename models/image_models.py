import os
import httpx
import base64
import uuid
import json
from abc import abstractmethod
from typing import Literal, List, overload, Tuple
from traceback import format_exc

from utils import logger
from utils.tencent_cos import tencent_cos
from utils.download import download_images

class BaseImageGenerationModel:
    @abstractmethod
    def generate_image(self, **kwargs) -> str:
        raise NotImplementedError("generate_image method must implement this class")


class AzureImageGenerationModel(BaseImageGenerationModel):
    def __init__(self):
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.base_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    async def _generate_image_with_text(self, 
        prompt: str, 
        size: Literal["auto", "1536x1024", "1024x1024", "1024x1536"] = "1024x1024", 
        quality: Literal["auto", "low", "medium", "high"] = "medium", 
        transparent: bool = False
    ) -> Tuple[bool, List[str]]:
        logger.info(f"Generating image with gpt-image-1.5 (text to image)")
        endpoint = f"{self.base_url}/openai/deployments/gpt-image-1.5/images/generations?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "output_compression": 100,
            "output_format": "png",
            "n": 1,
        }

        if transparent:
            payload["background"] = "transparent"
    
        try:
            async with httpx.AsyncClient(timeout=3000) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

            image_urls = []
            if "data" in result:
                for idx, item in enumerate(result["data"]):
                    if "b64_json" in item:
                        image_data = base64.b64decode(item["b64_json"])
                        filename = f"generated_image_{uuid.uuid4().hex[:6]}_{idx}.png"
                        file_path = os.path.join("temp/images", filename)
                        
                        with open(file_path, "wb") as f:
                            f.write(image_data)
                        image_urls.append(tencent_cos.save_to_tencent_cos(file_path, rm_local_file=True))

            if not image_urls:
                logger.warning(f"No image data found in response. Response: {json.dumps(result)}")
                return False, f"Error: No image data found in response. Please try again."
            else:
                logger.info(f"Generated {len(image_urls)} images successfully. \n{'\n'.join(image_urls)}")
                return True, image_urls
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return False, str(format_exc())


    async def _generate_image_with_image(self,
        prompt: str,
        image_url: str,
        size: Literal["auto", "1536x1024", "1024x1024", "1024x1536"] = "1024x1024",
        quality: Literal["auto", "low", "medium", "high"] = "medium",
        transparent: bool = False
    ) -> Tuple[bool, List[str]]:
        logger.info(f"Edits image with gpt-image-1.5 (image to image): {prompt}")
        endpoint = (
            f"{self.base_url}/openai/deployments/gpt-image-1.5/images/edits?api-version={self.api_version}"
        )
        headers = {
            "Api-key": self.api_key
        }
        data = {
            "model": "gpt-image-1.5",
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "output_compression": 100,
            "output_format": "png",
            "n": 1,
        }

        if transparent:
            data["background"] = "transparent"

        try:
            async with httpx.AsyncClient(timeout=3000) as client:
                image_paths = await download_images([image_url])
                image_path = image_paths[0]
                with open(image_path, "rb") as f:
                    files = {
                        "image[]": (os.path.basename(image_path), f, "image/png")
                    }
                    response = await client.post(
                        endpoint, 
                        headers=headers, 
                        files=files, 
                        data=data
                    )
                response.raise_for_status()
                result = response.json()

                image_urls = []
                if "data" in result:
                    for idx, item in enumerate(result["data"]):
                        if "b64_json" in item:
                            image_data = base64.b64decode(item["b64_json"])
                            filename = f"generated_image_{uuid.uuid4().hex[:8]}_{idx}.png"
                            file_path = os.path.join("temp/images", filename)
                            with open(file_path, "wb") as f:
                                f.write(image_data)
                            image_url = tencent_cos.save_to_tencent_cos(file_path, rm_local_file=True)
                            if image_url:
                                image_urls.append(f"{image_url}")

                if not image_urls:
                    logger.warning(f"No image data found in response. Response: {json.dumps(result)}")
                    return False, "No image data found in response."
                else:
                    logger.info(f"Edited {len(image_urls)} images successfully. \n{'\n'.join(image_urls)}")
                    return True, image_urls
        except httpx.HTTPStatusError as e:
            logger.error(f"Error editing image: {e}")
            return False, str(response.content.decode())
        except Exception as e:
            logger.error(f"Error editing image: {format_exc()}")
            return False, str(format_exc())

    @overload
    async def generate_image(self,
        prompt: str,
        size: Literal["auto", "1536x1024", "1024x1024", "1024x1536"] = "1024x1024",
        quality: Literal["auto", "low", "medium", "high"] = "medium",
        transparent: bool = False,
    ) -> List[str] | str:
        ...

    @overload
    async def generate_image(self,
        prompt: str,
        image_url: str,
        size: Literal["auto", "1536x1024", "1024x1024", "1024x1536"] = "1024x1024",
        quality: Literal["auto", "low", "medium", "high"] = "medium",
        transparent: bool = False,
    ) -> List[str] | str:
        ...

    async def generate_image(self,
        prompt: str,
        image_url: str = None,
        size: Literal["auto", "1536x1024", "1024x1024", "1024x1536"] = "1024x1024",
        quality: Literal["auto", "low", "medium", "high"] = "medium",
        n: int = 1,
        transparent: bool = False,
    ) -> List[str] | str:
        """Generate an image using Azure OpenAI gpt-image-1.5 model.

        Args:
            prompt (str): The prompt to generate image from.
            image_path (str, optional): The path to the image to edit. Defaults to None.
            size (Literal["auto", "1536x1024", "1024x1024", "1024x1536"], optional): The size of the generated image. Defaults to "1024x1024".
            quality (Literal["auto", "low", "medium", "high"], optional): The quality of the generated image. Defaults to "medium".
            transparent (bool, optional): Whether to generate a transparent image. Defaults to False.

        Returns:
            List[str] | str: The path to the generated image.
        """
        if image_url:
            return await self._generate_image_with_image(prompt, image_url, size, quality, transparent)
        else:
            return await self._generate_image_with_text(prompt, size, quality, transparent)
