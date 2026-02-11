from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import os
from typing import Literal, Optional, Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils import calculate_md5, logger


class TencentCOS:
    """腾讯云COS工具类

    Args:
        bucket_name (str): 桶名, default: surgepix-ai-1316642525
    """

    bucket_info_dict: Dict = {
        "surgepix-ai-1316642525": {
            "region": "ap-seoul",
        }
    }

    def __init__(self, bucket_name: str = "surgepix-ai-1316642525"):
        self.bucket_name = bucket_name

        secret_id = os.getenv('TENCENT_COS_SECRET_ID')
        secret_key = os.getenv('TENCENT_COS_SECRET_KEY')
        if not secret_id or not secret_key:
            raise ValueError("TENCENT_COS_SECRET_ID or TENCENT_COS_SECRET_KEY is not set")

        region = self.bucket_info_dict[self.bucket_name]["region"]
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Scheme='https')
        self.client = CosS3Client(config)


    def _get_cos_object_url(self, cos_path: str) -> Optional[str]:
        """获取cos对象的url"""
        response: bool = self.client.object_exists(
            Bucket=self.bucket_name,
            Key=cos_path,
        )
        if response:
            url: str = self.client.get_object_url(
                Bucket=self.bucket_name,
                Key=cos_path
            )
            return url
        else:
            return None

    def save_to_tencent_cos(self,
        file_path: str, 
        file_type: Literal["image"]="image",
        rm_local_file: bool = False
    ) -> str:
        if file_type == "image":
            cos_path = file_path.replace("temp/", "my/")

        if not os.path.exists(file_path):
            return ""
        
        file_md5_str = calculate_md5(file_path, input_type="file")
        
        logger.info(f"上传文件到腾讯云cos: {file_path} -> {cos_path}")
        self.client.upload_file(
            Bucket=self.bucket_name,
            LocalFilePath=file_path,
            Key=cos_path,
            PartSize=10,
            MAXThread=5,
            EnableMD5=True,
            ContentMD5=file_md5_str,
        )
        logger.info(f"上传文件到腾讯云cos成功: {file_path} -> {cos_path}")
        if rm_local_file:
            os.remove(file_path)
            logger.info(f"删除本地文件: {file_path}")
        return self._get_cos_object_url(cos_path)
    
    def async_save_to_tencent_cos(
        self, file_paths: List[str] | str, file_type: Literal["image"]="image", rm_local_file: bool = False
    ) -> List[Optional[str]]:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.save_to_tencent_cos, file_path, file_type, rm_local_file) for file_path in file_paths]
            results = [future.result() for future in futures]
        return results

tencent_cos = TencentCOS()
