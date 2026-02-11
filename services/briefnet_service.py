import os
import requests
from base64 import b64decode
from typing import Optional, List
import asyncio
import httpx

from utils import logger


async def remove_background(image_paths: List[str], output_path: Optional[str] = None) -> List[str]:
    async def async_process(client: httpx.AsyncClient, image_path: str, output_path: Optional[str] = None):
        url = "http://172.29.0.6:7791/birefnet/remove_background"
        if not os.path.exists(image_path):
            return "Sorry, Image not found!"
        ext = image_path.split(".")[-1]

        try:
            with open(image_path, "rb") as f:
                files = {"file": (image_path, f, f"image/{ext}")}
                response = await client.post(url, files=files)
                
            response_dict = response.json()
            if response_dict["code"] == 200:
                image_base64 = response_dict["image_base64"]
                if output_path is None:
                    output_path = image_path.replace(f".{ext}", f"_no_bg.{ext}")
                else:
                    image_name = os.path.basename(image_path).replace(f".{ext}", f"_no_bg.{ext}")
                    output_path = os.path.join(output_path, image_name)
                    
                with open(output_path, "wb") as f:
                    f.write(b64decode(image_base64))
                return output_path
            else:
                return "Sorry, " + response_dict["message"]
        except Exception as e:
            return "Sorry, " + str(e)

    async with httpx.AsyncClient() as client:
        tasks = [async_process(client, image_path, output_path) for image_path in image_paths]
        results = await asyncio.gather(*tasks)

    for result in results:
        if result.startswith("Sorry"):
            logger.error(result)
    correct_results = [result for result in results if not result.startswith("Sorry, ")]
    logger.info(f"Removed background for {len(correct_results)} images")
    return results
