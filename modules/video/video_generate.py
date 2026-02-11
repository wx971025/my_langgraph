import os
import base64
import uuid
import json
import requests
import asyncio
import time
import httpx
from pathlib import Path
import pandas as pd
from io import BytesIO
from abc import abstractmethod
from PIL import Image, ImageOps
from openai import OpenAI
from openai.types import VideoSeconds, VideoSize
from traceback import format_exc
from typing import List, Literal, Callable, Optional, TypeAlias
from pydantic import BaseModel, Field
from dotenv import load_dotenv; load_dotenv(".env")

from utils import logger
from utils.tencent_cos import tencent_cos
from modules.video.video_utils import (
    build_sora2_input_reference, 
    resize_image_to_size
)


cat_image = "assets/test_images/cat.jpg"
horse_image = "assets/test_images/horse.png"
phone_image = "assets/test_images/phone.png"
phone2_image = "assets/test_images/phone2.png"
female_image = "assets/test_images/female.jpg"
cup_image = "assets/test_images/cup.jpg"
hat_image = "assets/test_images/hat.jpg"
desk_image = "assets/test_images/desk.jpg"
male_image = "assets/test_images/male.png"
start_frame1_image = "assets/test_images/start_frame1.png"
end_frame1_image = "assets/test_images/end_frame1.png"
start_frame2_image = "assets/test_images/start_frame2.png"
end_frame2_image = "assets/test_images/end_frame2.png"
start_frame3_image = "assets/test_images/start_frame3.png"
end_frame3_image = "assets/test_images/end_frame3.png"
start_frame4_image = "assets/test_images/start_frame4.png"
end_frame4_image = "assets/test_images/end_frame4.png"
start_frame5_image = "assets/test_images/start_frame5.png"
end_frame5_image = "assets/test_images/end_frame5.png"
man2_image = "assets/test_images/man2.png"
man_walk_image = "assets/test_images/man_walk.png"
man3_image = "assets/test_images/man3.png"
man4_image = "assets/test_images/man4.png"

def get_video_file_name(
    task_id: str, *, size: Optional[tuple[int, int]] = None, ext: str="mp4"
) -> str:
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    if size:
        file_name = f"{task_id}_{time_str}_{size[0]}x{size[1]}.{ext}"
    else:
        file_name = f"{task_id}_{time_str}.{ext}"
    return file_name

class BaseVideoModel:
    def __init__(self):
        self._model_name: str | None = None

    @property
    def name(self) -> str | None:
        if self._model_name is None:
            raise ValueError("model name is not set")
        return self._model_name

    @abstractmethod
    def get_task_list(self, save: bool=False) -> bool:
        raise NotImplementedError("get_task_list method must implement this class")

    @abstractmethod
    def delete_task(self, video_id: str) -> bool:
        raise NotImplementedError("delete_task method must implement this class")

    @abstractmethod
    def generate_video(self, prompt: str, images: List[str]=None, **kwargs) -> str:
        raise NotImplementedError("generate_video method must implement this class")


class ViduVideoModel(BaseVideoModel):
    def __init__(self):
        self.headers =  {
            "Content-Type": "application/json",
            "Authorization": f"Token {os.getenv('VIDU_API_KEY')}"
        }
        self._model_name = "vidu"

    @property
    def name(self) -> str:
        return self._model_name

    def vidu_get_credit(self, credit_type: Literal["test", "metered"] = "test") -> str:
        """获取vidu的剩余测试额度"""
        data = {
            "show_detail": False,
        }
        try:
            response = requests.get(
                url='https://api.vidu.com/ent/v2/credits',
                headers=self.headers,
                json=data,
                timeout=500,
            )
            response.raise_for_status()
            response_json = response.json()
            for remain in response_json['remains']:
                if remain['type'] == credit_type:
                    credit_remain = f"{credit_type} credit: {remain['credit_remain']}"
                    break
            return credit_remain
        except Exception as e:
            logger.error(f"Error getting credit: {format_exc()}")
            return f"Error getting credit"

    def delete_task(self, task_id: str) -> bool:
        """取消vidu任务"""
        url=f'https://api.vidu.com/ent/v2/tasks/{task_id}/cancel'
        body = {
            "id": task_id
        }
        try:
            response = requests.post(
                url=url,
                headers=self.headers,
                json=body,
                timeout=500,
            )
            response.raise_for_status()
            print(self.get_task_list(save=False))
            return True
        except requests.exceptions.HTTPError as e:
            logger.error(f"Error canceling task: {format_exc()}\ntask id: {task_id}")
            return False


    def get_task_list(self, save: bool=False) -> bool:
        """获取vidu任务列表"""
        url=f'https://api.vidu.com/ent/v2/tasks'
        response = requests.get(
            url=url,
            headers=self.headers,
            timeout=500,
        )
        response.raise_for_status()
        response_json = response.json()
        tasks = response_json['tasks']
        next_page_token = response_json['next_page_token']

        while next_page_token:
            response = requests.get(
                url=url,
                headers=self.headers,
                params={"pager.page_token": next_page_token},
                timeout=500,
            )
            response.raise_for_status()
            response_json = response.json()
            tasks.extend(response_json['tasks'])
            next_page_token = response_json['next_page_token']

        df = pd.DataFrame.from_records(tasks)

        def get_url(creations: list):
            if len(creations) == 0:
                return None
            else:
                return [creation["url"] for creation in creations]

        df["urls"] = df["creations"].apply(get_url)
        save_df = df[["id", "type", "model", "state", "prompt", "urls"]]

        print(save_df[["id", "type", "model", "state", "urls"]])

        try:
            if save:
                save_df.to_csv("temp/vidu_task_list.csv", index=True)
            return True
        except Exception as e:
            logger.error(f"Error saving task list: {format_exc()}")
            return False


    async def vidu_get_task_result(self, task_id: str) -> str:
        async with httpx.AsyncClient() as client:
            t1 = time.time()
            while True:
                response = await client.get(
                    url=f"https://api.vidu.com/ent/v2/tasks/{task_id}/creations",
                    headers=self.headers,
                )
                state = response.json()["state"]

                match state:
                    case "success":
                        video_url = response.json()["creations"][0]["url"]
                        video_name = get_video_file_name(task_id)
                        Path(f"temp/videos/vidu").mkdir(parents=True, exist_ok=True)
                        with open(f"temp/videos/vidu/{video_name}", "wb") as f:
                            f.write(requests.get(video_url).content)
                        logger.info(self.vidu_get_credit("test"))
                        logger.info(f"{task_id} 成功, 保存到 temp/videos/vidu/{video_name}")
                        return video_url
                    case "processing":
                        logger.info(f"{task_id} Processing...")
                    case "queueing":
                        logger(f"{task_id} Queueing...")
                    case "created":
                        logger(f"{task_id} Created...")
                    case "failed":
                        logger(f"{task_id} Failed...")
                        return ''
                await asyncio.sleep(3)

    async def vidu_polling(self, task_id) -> str:
        """轮询vidu任务状态"""
        async with httpx.AsyncClient() as client:
            t1 = time.time()
            while True:
                response = await client.get(
                    url=f"https://api.vidu.com/ent/v2/tasks/{task_id}/creations",
                    headers=self.headers,
                )
                state = response.json()["state"]

                match state:
                    case "success":
                        logger.info(f"{task_id} success...{time.time()-t1:.2f}s")
                        video_url = response.json()["creations"][0]["url"]
                        time_str = f"{time.time() - t1:.2f}"
                        video_name = f"{task_id}_{time_str}.mp4"
                        save_path = f"temp/videos/{video_name}"
                        with open(save_path, "wb") as f:
                            f.write(requests.get(video_url).content)
                        self.vidu_get_credit("test")
                        logger.info(f"{task_id} 成功, 保存到 {save_path}")
                        return video_url
                    case "processing":
                        logger.info(f"Vidu {task_id} Processing...{time.time()-t1:.2f}s")
                    case "queueing":
                        logger.info(f"Vidu {task_id} Queueing...{time.time()-t1:.2f}s")
                    case "created":
                        logger.info(f"Vidu {task_id} Created...{time.time()-t1:.2f}s")
                    case "failed":
                        logger.info(f"Vidu {task_id} Failed...{time.time()-t1:.2f}s")
                        return ''
                await asyncio.sleep(3)

    class Text2VideoBodySchema(BaseModel):
        model: Literal["viduq2", "viduq1"] = Field("viduq2", description="The model to use for video generation. default: viduq2")
        style: Literal["general", "anime"]
        prompt: str = Field(..., max_length=2000, description="The prompt for video generation, max 2000 chars.")
        duration: int = Field(..., ge=1, le=10, description="The duration of the video, in seconds.")
        aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = Field(
            "1:1", description="The aspect ratio of the video. default: 1:1"
        )
        resolution: Literal["540p", "720p", "1080p"] = Field(
            "720p", description="The resolution of the video. default: 720p"
        )
        movement_amplitude: Literal["auto", "small", "medium", "large"] = Field(
            "auto", description="The movement amplitude of the video. default: auto"
        )
        bgm: bool = Field(default=True, description="Whether to add background music to the video. default: True")
        watermark: bool = Field(default=False, description="Whether to add watermark to the video. default: False")
        seed: int = Field(default=0, description="The seed for video generation. default: random")
        payload: str = Field(default="{}", description="The payload for video generation.")


    async def _vidu_async_text2video(self, 
        prompt: str, 
        aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = "9:16",
        resolution: Literal["540p", "720p", "1080p"] = "720p"
    ) -> str:
        """vidu文生视频"""
        logger.info(f"开始vidu文生视频: {prompt}")
        url='https://api.vidu.com/ent/v2/text2video'
        request_id = str(uuid.uuid4().hex)
        print(f"request_id: {request_id}")

        payload = json.dumps({
            "request_id": request_id
        })

        body = self.Text2VideoBodySchema(
            model="viduq2",
            style="general",
            prompt=prompt,
            duration=5,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude="auto",
            bgm=True,
            watermark=False,
            payload=payload
        ).model_dump()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=url,
                json=body,
                headers=self.headers,
                timeout=500,
            )
            task_id = response.json()["task_id"]
        
        return await self.vidu_polling(task_id)


    class Img2VideoBodySchema(BaseModel):
        model: Literal[
            "vidu2.0", "viduq1", "viduq1-classic", "viduq2-pro-fast", "viduq2-pro", "viduq2-turbo"
        ] = Field("viduq2-pro-fast", description="The model to use for video generation. default: viduq2-pro-fast")
        images: List[str] = Field(..., description="The images to use for video generation.")
        prompt: str = Field(..., max_length=2000, description="The prompt for video generation, max 2000 chars.")
        audio: bool = Field(default=True, description="Whether to add audio to the video. default: True")
        voice_id: str = Field(default="female-chengshu", description="The voice id to use for audio generation.")
        is_rec: bool = Field(default=False, description="Whether to use recommended prompt words. default: False")
        duration: int = Field(..., ge=1, le=8, description="The duration of the video, in seconds.")
        resolution: Literal["540p", "720p", "1080p"] = Field(
            "720p", description="The resolution of the video. default: 720p"
        )
        movement_amplitude: Literal["auto", "small", "medium", "large"] = Field(
            "auto", description="The movement amplitude of the video. default: auto"
        )
        watermark: bool = Field(default=False, description="Whether to add watermark to the video. default: False")
        seed: int = Field(default=0, description="The seed for video generation. default: random")
        payload: str = Field(default="{}", description="The payload for video generation.")


    async def _vidu_async_img2video(self,
        prompt: str, 
        images: List[str], 
        resolution: Literal["540p", "720p", "1080p"] = "720p"
    ) -> str:
        """vidu图生视频"""
        logger.info(f"开始生成vidu图生视频: {prompt}")
        url='https://api.vidu.com/ent/v2/img2video'
        request_id = str(uuid.uuid4().hex)
        print(f"request_id: {request_id}")
        
        payload = json.dumps({
            "request_id": request_id
        })

        image_b64s = []
        for image in images:
            image_b64 = f"data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode("utf-8")
            image_b64s.append(image_b64)
        
        body = self.Img2VideoBodySchema(
            model="viduq2-pro-fast",
            images=image_b64s,
            prompt=prompt,
            audio=True,
            voice_id="female-chengshu",
            is_rec=False,
            duration=7,
            payload=payload,
            resolution=resolution,
            movement_amplitude="auto",
            watermark=False,
            seed=0,
        ).model_dump()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=url,
                json=body,
                headers=self.headers,
                timeout=500,
            )
            task_id = response.json()["task_id"]

        return await self.vidu_polling(task_id)


    class StartEnd2videoBodySchema(BaseModel):
        model: Literal[
            "vidu2.0", "viduq1", "viduq1-classic", "viduq2-pro-fast", "viduq2-pro", "viduq2-turbo"
        ] = Field("viduq2-pro-fast", description="The model to use for video generation. default: viduq2-pro-fast")
        images: List[str] = Field(..., description="The start image to use for video generation.")
        prompt: str = Field(..., max_length=2000, description="The prompt for video generation, max 2000 chars.")
        is_rec: bool = Field(default=False, description="Whether to use recommended prompt words. default: False")
        duration: int = Field(..., ge=1, le=8, description="The duration of the video, in seconds.")
        seed: int = Field(default=0, description="The seed for video generation. default: random")
        resolution: Literal["540p", "720p", "1080p"] = Field(
            "720p", description="The resolution of the video. default: 720p"
        )
        movement_amplitude: Literal["auto", "small", "medium", "large"] = Field(
            "auto", description="The movement amplitude of the video. default: auto"
        )
        bgm: bool = Field(default=True, description="Whether to add background music to the video. default: True")
        watermark: bool = Field(default=False, description="Whether to add watermark to the video. default: False")
        payload: str = Field(default="{}", description="The payload for video generation.")


    async def _vidu_async_startend2video(self, 
        prompt: str, 
        images: List[str], 
        resolution: Literal["540p", "720p", "1080p"] = "720p"
    ):
        """vidu首尾帧生成视频"""
        logger.info(f"开始vidu首尾帧生成视频: {prompt}")
        url='https://api.vidu.com/ent/v2/start-end2video'
        request_id = str(uuid.uuid4().hex)
        print(f"request_id: {request_id}")
        
        payload = json.dumps({
            "request_id": request_id
        })

        assert len(images) == 2, "images must have 2 images"

        image_b64s = []
        for image in images:
            image_b64 = f"data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode("utf-8")
            image_b64s.append(image_b64)

        body = self.StartEnd2videoBodySchema(
            model="viduq2-pro-fast",
            images=image_b64s,
            prompt=prompt,
            is_rec=False,
            duration=5,
            seed=0,
            resolution=resolution,
            movement_amplitude="auto",
            bgm=True,
            watermark=False,
            payload=payload
        ).model_dump()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=url,
                json=body,
                headers=self.headers,
                timeout=500,
            )
            task_id = response.json()["task_id"]
        return await self.vidu_polling(task_id)
                    

    class ReferenceSubjectSchema(BaseModel):
        id: str = Field(..., description="The id of the subject.")
        images: List[str] = Field(..., description="The images of the subject.")
        voice_id: str = Field(default="female-chengshu", description="The voice id to use for audio generation.")

    class Reference2videoBodySchema(BaseModel):
        model: Literal[
            "vidu2.0", "viduq1", "viduq1-classic", "viduq2"
        ] = Field("viduq2", description="The model to use for video generation. default: viduq2")
        subjects: List = Field(..., description="The subjects of the video generation.")
        prompt: str = Field(..., max_length=2000, description="The prompt for video generation, max 2000 chars.")
        duration: int = Field(..., ge=1, le=10, description="The duration of the video, in seconds.")
        seed: int = Field(default=0, description="The seed for video generation. default: random")
        aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = Field(
            "1:1", description="The aspect ratio of the video. default: 1:1"
        )
        resolution: Literal["540p", "720p", "1080p"] = Field(
            "720p", description="The resolution of the video. default: 720p"
        )
        movement_amplitude: Literal["auto", "small", "medium", "large"] = Field(
            "auto", description="The movement amplitude of the video. default: auto"
        )
        watermark: bool = Field(default=False, description="Whether to add watermark to the video. default: False")
        payload: str = Field(default="{}", description="The payload for video generation.")


    async def _vidu_async_reference2video(self, 
        prompt: str, 
        images: List[str], 
        aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = "9:16", 
        resolution: Literal["540p", "720p", "1080p"] = "720p"
    ):
        """vidu参考生视频"""
        logger.info(f"开始vidu参考生视频: {prompt}")
        url='https://api.vidu.com/ent/v2/reference2video'
        request_id = str(uuid.uuid4().hex)
        print(f"request_id: {request_id}")
        
        payload = json.dumps({
            "request_id": request_id
        })

        images_b64 = []
        for image in images:
            image_b64 = f"data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode("utf-8")
            images_b64.append(image_b64)

        subjects = [
            {
                "id": f"{idx}",
                "images": [image],
                "voice_id": "female-chengshu"
            } for idx, image in enumerate(images_b64, start=1)
        ]

        body = self.Reference2videoBodySchema(
            model="viduq2",
            subjects=subjects,
            prompt=prompt,
            duration=10,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude="auto",
            watermark=False,
            payload=payload
        ).model_dump()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=url,
                json=body,
                headers=self.headers,
                timeout=500,
            )

            task_id = response.json()["task_id"]
        return await self.vidu_polling(task_id)

    async def generate_video(self, prompt: str, images: List[str], **kwargs) -> str:
        """default is vidu_reference2video"""
        aspect_ratio = kwargs.pop("aspect_ratio", "9:16")
        resolution = kwargs.pop("resolution", "720p")
        return await self._vidu_async_reference2video(prompt, images, aspect_ratio, resolution)



class Sora2VideoModel:
    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        base_url = os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1/"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model_name = "sora2"

    @property
    def name(self) -> str:
        return self._model_name

    def get_task_list(self, save: bool=False) -> bool:
        """获取sora2任务列表"""
        page = self.client.videos.list()
        df = pd.DataFrame.from_records(page.data)
        print(df)
        if save:
            try:
                df.to_csv("temp/sora2_task_list.csv", index=True)
            except Exception as e:
                logger.error(f"Error saving task list: {format_exc()}")
                return False
        return True

    def delete_task(self, video_id: str) -> bool:
        """删除sora2任务"""
        try:
            self.client.videos.delete(video_id)
            self.get_task_list(save=False)
        except Exception as e:
            logger.error(f"Error deleting task: {format_exc()}, task id: {video_id}")
            return False
        return True

    async def _sora2_polling(self, video_id: str) -> Optional[str]:
        """轮询sora2任务状态"""
        t1 = time.time()
        video = self.client.videos.retrieve(video_id)
        while video.status not in ["completed", "failed", "cancelled"]:
            print(f"Sora2 {video_id} {video.status}...{time.time()-t1:.2f}s")
            await asyncio.sleep(3)
            video = self.client.videos.retrieve(video_id)
        if video.status == "completed":
            logger.info(f"Video successfully completed: {video.id}")
            content = self.client.videos.download_content(video_id, variant="video")
            file_name = get_video_file_name(video_id)
            Path(f"temp/videos/sora2").mkdir(parents=True, exist_ok=True)
            content.write_to_file(f"temp/videos/sora2/{file_name}")
            return file_name
        elif video.status == "failed":
            logger.info(f"Video creation failed. Status: {video.status}")
            return None
        elif video.status == "cancelled":
            logger.info(f"Video creation cancelled. Status: {video.status}")
            return None
        else:
            return None

    class Sora2VideoBodySchema(BaseModel):
        model: Literal["sora-2"] = Field("sora-2", description="The model to use for video generation. default: sora-2")
        prompt: str = Field(..., description="The prompt for video generation.")
        size: VideoSize = Field("720x1280", description="The size of the video. default: 720x1280")
        seconds: VideoSeconds = Field("4", description="The duration of the video, in seconds. default: 4")


    async def _sora2_asycn_gen_video(self, 
        prompt: str, 
        images: List[str]=None, 
        size: Literal["720x1280", "1080x1920"] = "720x1280",
        seconds: Literal["4", "12"] = "12",
    ) -> Optional[str]:
        input_reference = None
        if images:
            if len(images) > 1:
                input_reference = build_sora2_input_reference(images, size)
            else:
                input_reference = resize_image_to_size(images[0], size)

        body = self.Sora2VideoBodySchema(
            model="sora-2",
            prompt=prompt,
            size=size,
            seconds=seconds,
        ).model_dump()

        if input_reference:
            body["input_reference"] = input_reference

        video = self.client.videos.create(**body)
        result = await self._sora2_polling(video.id)

        try:
            if hasattr(input_reference, "close"):
                input_reference.close()
        except Exception:
            logger.warning(f"Error closing input reference: {format_exc()}")
        return result

    async def generate_video(self, prompt: str, images: List[str]=None, **kwargs) -> str:
        size = kwargs.pop("size", "720x1280")
        seconds = kwargs.pop("seconds", "12")
        return await self._sora2_asycn_gen_video(prompt, images, size, seconds)


Wan2_6_Text2VideoSize: TypeAlias = Literal[
    "1280*720", "720*1280", "960*960", "1088*832", "832*1088",
    "1920*1080", "1080*1920", "1440*1440", "1632*1248", "1248*1632"
]
Wan2_6_FirstFrame2VideoSize: TypeAlias = Literal[
    "720P", "1080P"
]
Wan2_6_Duration: TypeAlias = Literal[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]

Wan2_6_VideoTaskStatus: TypeAlias = Literal[
    "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"
]

class Wan2_6_Text2VideoInputSchema(BaseModel):
    prompt: str = Field(..., description="The prompt for video generation.")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt for video generation.")
    audio_url: Optional[str] = Field(None, description="The audio url for video generation.")

class Wan2_6_Text2VideoParametersSchema(BaseModel):
    size: Wan2_6_Text2VideoSize = Field("1280*720", description="The size of the video. default: 1280*720")
    prompt_extend: bool = Field(True, description="Whether to extend the prompt. default: True")
    duration: int = Field(5, description="The duration of the video. default: 5")
    shot_type: Literal["single", "multi"] = Field("multi", description="The shot type of the video. default: multi")
    watermark: bool = Field(False, description="Whether to add watermark to the video. default: False")

class Wan2_6_Text2VideoBodySchema(BaseModel):
    model: Literal["wan2.6-t2v"] = Field("wan2.6-t2v", description="The model to use for video generation. default: wan2.6-t2v")
    input: Wan2_6_Text2VideoInputSchema = Field(..., description="The input for video generation.")
    parameters: Wan2_6_Text2VideoParametersSchema = Field(..., description="The parameters for video generation.")



class Wan2_6_FirstFrame2VideoInputSchema(BaseModel):
    prompt: str = Field(..., description="The prompt for video generation.")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt for video generation.")
    img_url: str = Field(..., description="The image url for video generation.")

class Wan2_6_FirstFrame2VideoParametersSchema(BaseModel):
    resolution: Wan2_6_FirstFrame2VideoSize = Field("720P", description="The size of the video. default: 720P")
    duration: int = Field(5, description="The duration of the video. default: 5")
    prompt_extend: bool = Field(True, description="Whether to extend the prompt. default: True")
    shot_type: Literal["single", "multi"] = Field("multi", description="The shot type of the video. default: multi")
    audio: bool = Field(True, description="Whether to add audio to the video. default: True")
    watermark: bool = Field(False, description="Whether to add watermark to the video. default: False")

class Wan2_6_FirstFrame2VideoBodySchema(BaseModel):
    model: Literal["wan2.6-i2v", "wan2.6-i2v-flash", "wan2.6-i2v-preview"] = Field("wan2.6-i2v", description="The model to use for video generation. default: wan2.6-i2v")
    input: Wan2_6_FirstFrame2VideoInputSchema = Field(..., description="The input for video generation.")
    parameters: Wan2_6_FirstFrame2VideoParametersSchema = Field(..., description="The parameters for video generation.")



class Wan2_6_Reference2VideoInputSchema(BaseModel):
    prompt: str = Field(..., description="The prompt for video generation.")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt for video generation.")
    reference_urls: List[str] = Field(..., description="The image urls for video generation.")

class Wan2_6_Reference2VideoParametersSchema(BaseModel):
    size: Wan2_6_Text2VideoSize = Field(..., description="The size of the video. default: 1920*1080")
    duration: int = Field(5, description="The duration of the video. default: 5")
    shot_type: Literal["single", "multi"] = Field("multi", description="The shot type of the video. default: multi")
    audio: bool = Field(True, description="Whether to add audio to the video. default: True")
    watermark: bool = Field(False, description="Whether to add watermark to the video. default: False")

class Wan2_6_Reference2VideoBodySchema(BaseModel):
    model: Literal["wan2.6-r2v-flash"] = Field("wan2.6-r2v-flash", description="The model to use for video generation. default: wan2.6-r2v-flash")
    input: Wan2_6_Reference2VideoInputSchema = Field(..., description="The input for video generation.")
    parameters: Wan2_6_Reference2VideoParametersSchema = Field(..., description="The parameters for video generation.")


class Wan2_6_VideoModel(BaseVideoModel):
    def __init__(self):
        self._model_name = "wan2_6"
        self.api_key = os.getenv("ALI_CLOUD_API_KEY")
        if self.api_key is None:
            raise ValueError("ALI_CLOUD_API_KEY is not set")

        self.headers = {
            "X-DashScope-Async": "enable",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.task_status = {
            "PENDING": "任务排队中...",
            "RUNNING": "任务处理中...",
            "SUCCEEDED": "任务执行成功!",
            "FAILED": "任务执行失败!",
            "CANCELED": "任务已取消!",
            "UNKNOWN": "任务不存在或状态未知!",
        }

    @property
    def name(self) -> str:
        return self._model_name

    def get_task_list(self, save: bool = False):
        """获取wan2_6任务列表"""
        return True

    def delete_task(self, video_id: str):
        """删除wan2_6任务"""
        return True
    
    async def _image_to_b64(self, image_path: str) -> str:
        ext = image_path.split(".")[-1]
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        b64_str = f"data:image/{ext};base64," + base64.b64encode(image_bytes).decode("utf-8")
        return b64_str
    
    async def _image_to_url(self, image_paths: List[str]) -> List[str]:
        urls = tencent_cos.async_save_to_tencent_cos(image_paths, file_type="image")
        return urls


    async def _get_task_status(self, task_id: str) -> str:
        url = f"https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        t1 = time.time()
        while True:
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                response_json = response.json()
                if "output" in response_json and "task_status" in response_json["output"] and response_json["output"]["task_status"] in self.task_status:
                    video_status: Wan2_6_VideoTaskStatus = response_json["output"]["task_status"]
                    if video_status in ["PENDING", "RUNNING"]:
                        video_status_cn = self.task_status[video_status]
                        print(f"Wan2.6 {task_id} {video_status_cn}...{time.time()-t1:.2f}s")
                        await asyncio.sleep(3)
                        continue
                    elif video_status == "SUCCEEDED":
                        video_url = response_json["output"]["video_url"]
                        print(f"Wan2.6 {task_id} video_url: {video_url}")
                        Path(f"temp/videos/wan2_6").mkdir(parents=True, exist_ok=True)
                        file_name = get_video_file_name(task_id)
                        with open(f"temp/videos/wan2_6/{file_name}", "wb") as f:
                            f.write(requests.get(video_url).content)
                        return response_json["output"]["video_url"]
                    elif video_status == "FAILED":
                        logger.error(f"Wan2.6 {task_id} failed: {response_json['message']}")
                        return response_json["message"]
                    elif video_status == "CANCELED":
                        logger.error(f"Wan2.6 {task_id} canceled: {response_json['message']}")
                        return response_json["message"]
                    elif video_status == "UNKNOWN":
                        logger.error(f"Wan2.6 {task_id} unknown: {response_json['message']}")
                        return response_json["message"]
            except Exception as e:
                logger.error(f"Error getting task status: {format_exc()}")
                return None
    
    async def _wan2_6_async_text_to_video(self, prompt: str, **kwargs) -> str:
        """文生视频"""
        logger.info(f"开始wan2.6文生视频...")
        try:
            url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
            input = Wan2_6_Text2VideoInputSchema(
                prompt=prompt,
                negative_prompt=None,
                audio_url=None,
            ).model_dump()

            parameters = Wan2_6_Text2VideoParametersSchema(
                size="1280*720",
                prompt_extend=True,
                duration=10,
                shot_type="multi",
            ).model_dump()

            body = Wan2_6_Text2VideoBodySchema(
                model="wan2.6-t2v",
                input=input,
                parameters=parameters,
            ).model_dump()

            response = requests.post(url, headers=self.headers, json=body)
            response.raise_for_status()
            task_id = response.json()["output"]["task_id"]

            result = await self._get_task_status(task_id)
            return result

        except Exception as e:
            logger.error(f"Error generating video: {format_exc()}")
            return None
    
    async def _wan2_6_async_first_frame_to_video(self, prompt: str, image_path: str, **kwargs) -> str:
        """首帧生视频"""
        logger.info(f"开始wan2.6首帧生成视频...")
        if Path(image_path).exists():
            img_url = await self._image_to_b64(image_path)
        else:
            raise ValueError(f"Image {image_path} not found")
        
        url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"

        input = Wan2_6_FirstFrame2VideoInputSchema(
            prompt=prompt,
            img_url=img_url,
        ).model_dump()

        parameters = Wan2_6_FirstFrame2VideoParametersSchema(
            resolution="720P",
            duration=10,
            prompt_extend=True,
            shot_type="multi",
            audio=True,
            watermark=False,
        ).model_dump()

        body = Wan2_6_FirstFrame2VideoBodySchema(
            model="wan2.6-i2v",
            input=input,
            parameters=parameters,
        ).model_dump()

        response = requests.post(url, headers=self.headers, json=body)
        response.raise_for_status()
        task_id = response.json()["output"]["task_id"]

        result = await self._get_task_status(task_id)

        return result

    async def _wan2_6_async_reference_to_video(self, prompt: str, images: List[str], **kwargs) -> str:
        """参考生成视频"""
        logger.info(f"开始wan2.6参考生成视频...")
        if len(images) > 5:
            logger.warning(f"参考生成视频图片数量超过5，只取前5张")
            images = images[:5]
            
        image_urls = await self._image_to_url(images)
        
        url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"

        input = Wan2_6_Reference2VideoInputSchema(
            prompt=prompt,
            reference_urls=image_urls,
        ).model_dump()

        parameters = Wan2_6_Reference2VideoParametersSchema(
            size="1920*1080",
            duration=10,
            shot_type="multi",
            audio=True,
            watermark=False,
        ).model_dump()

        body = Wan2_6_Reference2VideoBodySchema(
            model="wan2.6-r2v-flash",
            input=input,
            parameters=parameters,
        ).model_dump()

        response = requests.post(url, headers=self.headers, json=body)
        response.raise_for_status()
        task_id = response.json()["output"]["task_id"]

        result = await self._get_task_status(task_id)
        return result

    async def generate_video(self, prompt: str, images: List[str]=None, **kwargs) -> str:
        try:
            if images:
                if len(images) > 1:
                    raise ValueError("Only one image is supported for first frame to video generation")
                # result = await self._wan2_6_async_first_frame_to_video(prompt, images[0], **kwargs)
                result = await self._wan2_6_async_reference_to_video(prompt, images, **kwargs)
            else:
                result = await self._wan2_6_async_text_to_video(prompt, images, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error generating video: {format_exc()}")
            return None


async def batch_generate_video(
    video_model: BaseVideoModel, prompts: List[str] | str, images_list: List[List[str]] | str = None
):
    generate_video = video_model.generate_video
    if isinstance(prompts, str):
        prompts = [prompts] 
    
    tasks = []
    if images_list and images_list[0]:
        assert len(prompts) == len(images_list), "prompts and images_list must have the same length"
        for prompt, images in zip(prompts, images_list):
            tasks.append(asyncio.create_task(generate_video(prompt, images)))
            await asyncio.sleep(1)
    else:
        for prompt in prompts:
            tasks.append(asyncio.create_task(generate_video(prompt)))
            await asyncio.sleep(1)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    vivo_image1_path = "assets/vivo/1.png"
    vivo_image2_path = "assets/vivo/2.png"
    vivo_image3_path = "assets/vivo/3.png"
    vivo_image4_path = "assets/vivo/4.png"
    vivo_image5_path = "assets/vivo/5.png"
    vivo_image6_path = "assets/vivo/6.png"

    prompt_json_path = "prompt/prompt.json"
    with open(prompt_json_path, "r") as f:
        prompt = json.load(f)
        prompt_str = json.dumps(prompt, ensure_ascii=False)

    repeat_num = 1
    prompts = [prompt_str] * repeat_num
    images = [
        [
            vivo_image1_path,
            vivo_image2_path,
            vivo_image3_path,
            vivo_image4_path,
            vivo_image5_path,
            # vivo_image6_path,
        ] for _ in range(repeat_num)
    ]

    model = Wan2_6_VideoModel()
    asyncio.run(batch_generate_video(model, prompts, images_list=images))    
    # model.get_task_list(save=True)
