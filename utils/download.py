import os
import requests
import httpx
import asyncio
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Optional, List
from traceback import format_exc

from utils import logger


async def download_images(urls: List[str], output_path: Optional[str] = None) -> List[Optional[str]]:
    """下载多张图片到本地 temp/images/download 文件夹, 用当前日期时间命名

    Args:
        urls (List[str]): 图片的URL列表
        output_path (Optional[str], optional): 图片保存路径, 一定要在temp/images下.

    Returns:
        List[str]: 本地保存文件的绝对路径列表
    """
    if output_path is None:
        output_path: Path = Path("temp", "images", "download")
    else:
        path_obj = Path(output_path)
        if not str(output_path).startswith("temp"):
            output_path: Path = Path("temp") / path_obj
        else:
            output_path: Path = path_obj
    output_path.mkdir(parents=True, exist_ok=True)

    async def async_process(client: httpx.AsyncClient, image_url: str, output_path: Optional[str] = None) -> str:
        try:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            await asyncio.sleep(0.01)
            logger.info(f"Downloading image: {image_url}")
            resp = await client.get(image_url, timeout=15)
            resp.raise_for_status()
            ct = resp.headers.get('Content-Type', '')
            ext = None
            content_type_map = {
                "jpeg": "jpg",
                "jpg": "jpg",
                "png": "png",
                "webp": "webp",
                "gif": "gif"
            }
            for key, val in content_type_map.items():
                if key in ct:
                    ext = val
                    break
            else:
                if '.' in image_url.split("/")[-1]:
                    ext = image_url.split("/")[-1].split(".")[-1]
            if ext is None:
                logger.warning(f"无法确定下载图片扩展名, 使用默认扩展名: png")
                ext = "png"

        except Exception as e:
            logger.error(format_exc())
            logger.error(f"图片下载失败: {image_url}")
            return ""

        filename = f"{now_str}_{uuid4().hex[:6]}.{ext}"
        filepath: Path = output_path / filename
        with open(filepath, "wb") as f:
            f.write(resp.content)
        logger.info(f"Downloaded image successfully: {image_url} to {filepath}")
        return str(filepath)
        
    async with httpx.AsyncClient() as client:
        tasks = [async_process(client, url, output_path) for url in urls]
        image_paths = await asyncio.gather(*tasks)
        return image_paths
    

async def download_files(file_urls: List[str], output_path: Optional[str] = Path("temp/temp_files")) -> List[Optional[str]]:
    """下载c端上传的文件到本地 temp/temp_files 文件夹

    Args:
        file_urls (List[str]): _description_
        output_path (Optional[str], optional): _description_. Defaults to None.

    Returns:
        List[Optional[str]]: _description_
    """
    async def async_process(client: httpx.AsyncClient, file_url: str, output_path: Optional[str | Path] = Path("temp/temp_files")) -> str:
        try:
            resp = await client.get(file_url, timeout=15)
            resp.raise_for_status()
            if isinstance(output_path, str):
                output_path: Path = Path(output_path)
            file_name = file_url.split("/")[-1]
            file_output_path: Path = output_path / file_name
            with open(file_output_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Downloaded file: {file_url} to {file_output_path}")
            return str(file_output_path)
        except Exception as e:
            logger.error(f"文件下载失败: {file_url}\n{e}")
            return ""
            
    async with httpx.AsyncClient() as client:
        tasks = [async_process(client, file_url, output_path) for file_url in file_urls]
        file_paths: List[Optional[str]] = await asyncio.gather(*tasks)
        return file_paths
