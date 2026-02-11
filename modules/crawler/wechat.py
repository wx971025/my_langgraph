from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from typing import List
import time

from utils import logger



class WeChatCrawler:    
    def __init__(self, headless: bool = False, timeout: int = 10):
        """
        初始化爬虫
        
        Args:
            headless: 是否使用无头模式（不显示浏览器窗口）
            timeout: 默认等待超时时间（秒）
        """
        self.timeout = timeout
        self.driver = self._init_driver(headless)
        
    
    def _init_driver(self, headless: bool) -> webdriver.Chrome:
        """
        初始化 Chrome WebDriver
        
        Args:
            headless: 是否使用无头模式
            
        Returns:
            Chrome WebDriver 实例
        """
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # 使用 webdriver_manager 自动管理 ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 设置隐式等待
        driver.implicitly_wait(self.timeout)
        
        logger.info("Chrome WebDriver 初始化成功")
        return driver
    
    def open_url(self, url: str) -> bool:
        try:
            self.driver.get(url)
            logger.info(f"已打开页面: {url}")
            return True
        except Exception as e:
            logger.error(f"打开页面失败: {e}")
            return False
    
    def wait_for_element(self, by: By, value: str, timeout: int = None) -> bool:
        """
        等待元素出现
        
        Args:
            by: 定位方式 (By.ID, By.XPATH, By.CSS_SELECTOR 等)
            value: 定位值
            timeout: 超时时间，默认使用初始化时设置的值
            
        Returns:
            元素是否出现
        """
        timeout = timeout or self.timeout
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return True
        except TimeoutException:
            logger.warning(f"等待元素超时: {by}={value}")
            return False
    
    def find_element(self, by: By, value: str):
        try:
            return self.driver.find_element(by, value)
        except NoSuchElementException:
            logger.warning(f"未找到元素: {by}={value}")
            return None
    
    def find_elements(self, by: By, value: str):
        return self.driver.find_elements(by, value)
    
    def get_page_source(self) -> str:
        """获取网页源代码"""
        return self.driver.page_source
    
    def screenshot(self, filename: str) -> bool:
        try:
            self.driver.save_screenshot(filename)
            logger.info(f"截图已保存: {filename}")
            return True
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return False
    
    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info("浏览器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def get_album_article_urls(url: str) -> List[str]:
    """获取合集文章URLs"""
    with WeChatCrawler(headless=True) as crawler:
        crawler.open_url(url)
        title = crawler.find_element(By.XPATH, "//div[@id='js_tag_name' and @class='album__label-title bold']").text
        ul_element = crawler.find_element(By.XPATH, "//ul[@wah-hotarea='click']")
        articles = ul_element.find_elements(By.XPATH, "//li[@class='album__list-item js_album_item js_wx_tap_highlight wx_tap_cell']")    

        article_urls = []
        for article in articles:
            article_url = article.get_attribute("data-link")
            article_urls.append(article_url)
        logger.info(f"获取到 {len(article_urls)} 篇文章")
        return title, article_urls

def get_book_content(url: str) -> str:
    title, article_urls = get_album_article_urls(url)
    contents = []
    with WeChatCrawler(headless=True) as crawler:
        for article_url in article_urls:
            crawler.open_url(article_url)
            content_element = crawler.find_element(By.XPATH, "//div[@id='js_article']")
            contents.append(content_element.text)

    return title, contents


from ebooklib import epub
def create_epub(book_name: str, english_book_name: str, author: str, chapter_content: str, file_path: str = None):
    import re

    chapter_pattern = re.compile(r'(Chapter[\d一二三四五六七八九十零百千万]+.*?)(?=(?:Chapter[\d一二三四五六七八九十零百千万]+)|\Z)', re.DOTALL)

    chapter_contents = []
    for match in chapter_pattern.finditer(chapter_content):
        content = match.group(1).strip()
        if content:
            chapter_contents.append(content)
    """
    生成包含目录的epub电子书
    Args:
        book_name: 书籍名称
        english_book_name: 英文书籍名称（可用于文件名）
        author: 作者名
        chapter_contents: 每一章的内容列表，内容字符串以'ChapterX'开头
        file_path: 保存路径 (可选，不填则用书名)
    """
    book = epub.EpubBook()
    book.set_identifier(english_book_name or book_name)
    book.set_title(book_name)
    book.set_language('zh')
    book.add_author(author)
    
    # 创建章节对象和目录
    chap_objs = []
    toc = []
    spine = ['nav']

    for idx, content in enumerate(chapter_contents, 1):
        # 尝试找到章节标题
        lines = content.splitlines()
        title = f'Chapter{idx}'
        chapter_text = content
        if lines and lines[0].strip().startswith('Chapter'):
            title = lines[0].strip()
            chapter_text = "\n".join(lines[1:]).strip()

        c = epub.EpubHtml(
            title=title,
            file_name=f'chap_{idx}.xhtml',
            lang='zh'
        )
        # 适当用<p>包裹段落
        html_body = ""
        for paragraph in chapter_text.split('\n'):
            paragraph = paragraph.strip()
            if paragraph:
                html_body += f"<p>{paragraph}</p>\n"
        # 没有内容时添加一个空段落
        if not html_body:
            html_body = "<p></p>"
        c.content = f"<h1>{title}</h1>{html_body}"
        book.add_item(c)
        chap_objs.append(c)
        toc.append(epub.Link(f'chap_{idx}.xhtml', title, f'chap_{idx}'))
        spine.append(c)
    
    if not file_path:
        # epub文件名: 英文名优先，否则中文名
        safe_title = english_book_name or book_name
        safe_title = book_name
        file_path = f"{safe_title}.epub"
    
    # toc采用章节对象目录树
    book.toc = tuple(chap_objs)
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(f'./epub/{file_path}', book, {})
    print(f"epub已生成: {file_path}")


from pydantic import BaseModel, Field

class ChapterContent(BaseModel):
    book_name: str = Field(description="书籍名称")
    english_book_name: str = Field(default=None, description="英文书籍名称")
    author: str = Field(description="作者")
    # chapter_contents: List[str] = Field(description="章节内容")

def llm_process(url):
    import os
    os.makedirs('./epub', exist_ok=True)

    from models.chat_models import AzureChatOpenAI, AzureChatOpenAIModelName
    model = AzureChatOpenAI(AzureChatOpenAIModelName.o1).with_structured_output(ChapterContent)


    title, contents = get_book_content(url)

    for i, content in enumerate(contents):
        content = content.replace("\n\n\n\n\n", "\n\n\n")
        content = content.replace("\n\n\n\n", "\n\n\n")
        content = content.replace("\n\n", "\n\n\n")
        c_list = []
        for c in content.split('\n'):
            c = c.replace(" ", "")
            c_list.append(c)
        content = "\n".join(c_list)
        contents[i] = f"Chapter{i+1}\n\n{content}"


    messages = [
        {
            "role": "user",
            "content": "\n\n".join(contents)
        }
    ]

    t1 = time.time()
    response: ChapterContent = model.invoke(messages)
    book_name = response.book_name
    english_book_name = response.english_book_name
    author = response.author
    # chapter_contents = response.chapter_contents
    print(f"处理时间: {time.time() - t1:.3f} 秒")

    with open(f"./epub/{book_name}.txt", "w", encoding="utf-8") as f:
        contents_str = "\n\n".join(contents)
        f.write(contents_str)
    os.chmod(f"./epub/{book_name}.txt", 0o777)
    import pdb; pdb.set_trace()

    with open(f"./epub/{book_name}.txt", "r", encoding="utf-8") as f:
        chapter_contents = f.read()

    create_epub(
        book_name=book_name,
        english_book_name=english_book_name,
        author=author,
        chapter_content=chapter_contents,
    )





if __name__ == "__main__":
    url = "https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MjM5Nzk5NTI0MQ==&action=getalbum&album_id=1888874498930606084"
    llm_process(url)



