import re

def format_chat_message(content: str) -> str:
    """
    Format chat message to display images from URLs.
    Detects image URLs in the content and converts them to HTML img tags.
    """

    def _img_block(url: str, alt: str = "Generated Image") -> str:
        # 1) 点击图片在新标签打开原图（原尺寸）
        # 2) 提供下载链接（跨域时浏览器可能忽略 download，但仍可打开后另存为）
        # 3) no-referrer 规避部分 COS/对象存储防盗链
        safe_alt = alt.replace('"', "&quot;")
        return (
            "<div class=\"chat-img-wrap\">"
            f"<a class=\"chat-img-link\" href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">"
            f"<img data-imgwrap=\"1\" src=\"{url}\" alt=\"{safe_alt}\" referrerpolicy=\"no-referrer\" "
            "style=\"max-width: 100%; max-height: 400px; border-radius: 8px; cursor: zoom-in;\" />"
            "</a>"
            "<div class=\"chat-img-actions\">"
            f"<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">原图</a>"
            " · "
            f"<a href=\"{url}\" download>下载</a>"
            "</div>"
            "</div>"
        )

    # A) 处理模型直接输出的 <img src="...">（把它包一层可点击/可下载）
    #    跳过已经加了 data-imgwrap 的 img，避免重复包裹
    html_img_pattern = r'<img(?![^>]*data-imgwrap)[^>]*\ssrc=["\'](https?://[^"\']+)["\'][^>]*>'

    def html_img_sub(match: re.Match) -> str:
        url = match.group(1)
        return _img_block(url)

    content = re.sub(html_img_pattern, html_img_sub, content)

    # B) 处理 markdown 图片 ![alt](url)
    md_img_pattern = r'!\[([^\]]*)\]\((https?://[^\)]+)\)'

    def md_sub(match: re.Match) -> str:
        alt = match.group(1) or "Generated Image"
        url = match.group(2)
        return _img_block(url, alt=alt)

    content = re.sub(md_img_pattern, md_sub, content)
    
    url_pattern = r'(https?://[^\s<"]+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s<"]*)?)'
    
    parts = []
    last_idx = 0
    
    for match in re.finditer(url_pattern, content):
        start, end = match.span()
        url = match.group(1)
        
        lookback = content[max(0, start-20):start]
        if 'src="' in lookback or 'href="' in lookback:
            parts.append(content[last_idx:end])
        else:
            parts.append(content[last_idx:start])
            parts.append(_img_block(url))
            
        last_idx = end
        
    parts.append(content[last_idx:])
    return "".join(parts)
