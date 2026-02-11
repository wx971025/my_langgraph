from PIL import Image
from io import BytesIO
from math import ceil

from utils import logger


def resize_image_to_size(image_path: str, target_size: str) -> BytesIO:
    """
    将图片缩放到目标尺寸
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸，如 "720x1280"
    
    Returns:
        BytesIO: 缩放后的图片数据
    """
    target_w, target_h = map(int, target_size.split("x"))
    
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        current_w, current_h = img.size
        
        # 如果尺寸已经匹配，直接返回
        if current_w == target_w and current_h == target_h:
            bio = BytesIO()
            img.save(bio, format="PNG")
            bio.seek(0)
            bio.name = "input_reference.png"
            return bio
        
        # 缩放图片到目标尺寸
        logger.info(f"Resizing image from {current_w}x{current_h} to {target_w}x{target_h}")
        resized_img = img.resize((target_w, target_h), Image.LANCZOS)
        
        bio = BytesIO()
        resized_img.save(bio, format="PNG")
        bio.seek(0)
        bio.name = "input_reference.png"
        return bio


def build_sora2_input_reference(images_path: list[str], resolution: str) -> BytesIO:
    """
    多图 -> 自适应拼图（不裁切）-> 输出与 resolution 一致的 PNG BytesIO
    可直接作为 Sora2 input_reference 传入。
    """
    if not images_path:
        raise ValueError("images_path 不能为空")

    W, H = map(int, resolution.lower().split("x"))
    bg = (245, 245, 245)

    # 先读入所有图片尺寸（只读 size，不保留像素）
    img_sizes = []
    for p in images_path:
        with Image.open(p) as im:
            img_sizes.append(im.size)  # (w,h)

    n = len(images_path)

    def layout_score(cols: int) -> float:
        rows = ceil(n / cols)
        cell_w = W / cols
        cell_h = H / rows

        total_area = 0.0
        for (iw, ih) in img_sizes:
            s = min(cell_w / iw, cell_h / ih)
            total_area += (iw * s) * (ih * s)

        # 惩罚空格子
        empty = cols * rows - n
        penalty = empty * (cell_w * cell_h) * 0.15
        return total_area - penalty

    # 穷举 cols=1..n，选最佳布局
    best_cols = max(range(1, n + 1), key=layout_score)
    cols = best_cols
    rows = ceil(n / cols)

    # 生成画布
    canvas = Image.new("RGB", (W, H), bg)

    # 逐格放图（contain：不裁切，居中，可能留白）
    for idx, p in enumerate(images_path):
        r = idx // cols
        c = idx % cols

        x1 = int(c * (W / cols))
        y1 = int(r * (H / rows))
        x2 = int((c + 1) * (W / cols)) if c < cols - 1 else W
        y2 = int((r + 1) * (H / rows)) if r < rows - 1 else H
        cell_w = x2 - x1
        cell_h = y2 - y1

        with Image.open(p) as im:
            im = im.convert("RGB")
            iw, ih = im.size
            s = min(cell_w / iw, cell_h / ih)
            nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
            im2 = im.resize((nw, nh), Image.LANCZOS)

            px = x1 + (cell_w - nw) // 2
            py = y1 + (cell_h - nh) // 2
            canvas.paste(im2, (px, py))

    bio = BytesIO()
    canvas.save(bio, format="PNG")
    bio.seek(0)
    bio.name = "input_reference.png"  # 让 SDK 推断 mimetype=image/png
    return bio
