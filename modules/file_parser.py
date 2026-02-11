import os
import magic
from concurrent.futures import ProcessPoolExecutor
from langchain_core.documents import Document
from unstructured.documents.elements import Element
from unstructured.partition.md import partition_md
from unstructured.partition.html import partition_html
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.csv import partition_csv
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_doc
from unstructured.partition.docx import partition_docx
from unstructured.partition.image import partition_image
from typing import List, Dict, Any

from utils import logger


class FileParser:
    def __init__(self):
        pass

    def _check_file_type(self, file_path: str) -> str:
        try:
            if file_path.startswith("http") or file_path.startswith("https"):
                return 'html'

            mime_type = magic.from_file(file_path, mime=True)
            if mime_type == 'text/plain':
                if file_path.lower().endswith('.md') or \
                    file_path.lower().endswith('.markdown'):
                    return 'md'
                elif file_path.lower().endswith('.json'):
                    return 'json'
                return 'txt'
            
            if mime_type == 'application/vnd.ms-excel':
                if file_path.lower().endswith('.xls'):
                    return 'xls'
                elif file_path.lower().endswith('.xlsx'):
                    return 'xlsx'
                elif file_path.lower().endswith('.csv'):
                    return 'csv'
                return 'xlsx'

            mime_map = {
                'application/json': 'json',
                'application/pdf': 'pdf',
                'text/csv': 'csv',
                'application/csv': 'csv',
                'application/msword': 'doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.ms-powerpoint': 'ppt',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'ppt',
                'text/markdown': 'md',
                'text/x-markdown': 'md',
                'text/html': 'html',
                'image/jpeg': 'jpg',
                'image/png': 'png',
                'image/webp': 'webp',
            }
            
            file_ext = os.path.splitext(file_path)[1][1:]
            file_type = mime_map.get(mime_type, "unknown")

            if file_type != file_ext:
                logger.warning(
                    f"File type {file_type} does not match extension {file_ext}"
                )
            return file_type
        except Exception as e:
            logger.warning(f"Error checking file type: {e}")
            return "unknown"
    
    def _docx_parser(self, file_path: str) -> List[Document]:
        elements: List[Element] = partition_docx(file_path)
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents
    
    def _doc_parser(self, file_path: str) -> List[Document]:
        elements: List[Element] = partition_doc(file_path)
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents

    def _markdown_parser(self, file_path: str) -> List[Document]:
        elements: List[Element] = partition_md(file_path)
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents

    def _html_parser(self, file_path: str) -> List[Document]:
        if file_path.startswith("http") or file_path.startswith("https"):
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
            elements: List[Element] = partition_html(
                url = file_path,
                headers=headers,
                ssl_verify=False,
                include_page_breaks=False,
                encoding="utf-8"
            )
        else:
            elements: List[Element] = partition_html(
                file_path = file_path,
            )
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents
    
    def _xlsx_parser(self, file_path: str) -> List[Document]:
        elements: List[Element] = partition_xlsx(file_path)
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents

    def _csv_parser(self, file_path: str) -> List[Document]:
        elements: List[Element] = partition_csv(file_path)
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents

    def _pdf_parser(self, file_path: str) -> List[Document]:
        elements: List[Element] = partition_pdf(
            filename=file_path,
            strategy="hi_res", # 使用hi_res模式进行高精度解析
            extract_images_in_pdf=True, # 提取pdf中的图片
            extract_image_block_types=["Table","Image"], # 提取表格和图片
            extract_image_block_output_dir="./images", # 保存图片到images目录
            languages=["eng","zho"],
            split_pdf_page=True, # 大文件分块处理，优化性能
            infer_table_structure=True, # 是否尝试推断表格结构，会下载一个ocr模型
            include_page_breaks=True,    # 是否包含页码信息
        )
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents

    def _image_parser(self, file_path: str) -> List[Document]:
        elements: List[Element] = partition_image(
            file_path,
            strategy="ocr_only",
            languages=["eng","chi_sim"],
            include_page_breaks=False
        )
        metadata = lambda element: {k: v for k, v in element.metadata.__dict__.items() if not k.startswith('_')}
        documents = [Document(page_content=element.text, metadata=metadata(element)) for element in elements]
        return documents


    def parse(self, file_path: str) -> Dict[str, Any]:
        file_type = self._check_file_type(file_path)
        if file_type == 'docx':
            documents = self._docx_parser(file_path)
        elif file_type == 'doc':
            documents = self._doc_parser(file_path)
        elif file_type == 'md':
            documents = self._markdown_parser(file_path)
        elif file_type == 'html':
            documents = self._html_parser(file_path)
        elif file_type == 'xlsx':
            documents = self._xlsx_parser(file_path)
        elif file_type == 'csv':
            documents = self._csv_parser(file_path)
        elif file_type == 'pdf':
            documents = self._pdf_parser(file_path)
        elif file_type in ['jpg', 'png', 'webp']:
            documents = self._image_parser(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return {"file_path": file_path, "documents": documents}


    def batch_parse(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.parse, file_path) for file_path in file_paths]
            documents = [future.result() for future in futures]
        return documents


if __name__ == "__main__":
    file_parser = FileParser()
    documents = file_parser.parse("assets/test_file/甬兴证券-AI行业点评报告：海外科技巨头持续发力AI，龙头公司中报业绩亮眼.pdf")
    print(documents)
    