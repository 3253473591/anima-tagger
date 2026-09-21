import os
import shutil
from pathlib import Path
from PIL import Image

SRC_DIR = r"<数据集根目录>"
DST_DIR = r"<输出目录>"
MAX_SIZE = 1536
EXTENSIONS = {".png", ".jpg", ".jpeg"}

def process_image(src_path: Path, dst_path: Path) -> str:
    """
    处理单张图片。返回处理说明字符串。
    """
    with Image.open(src_path) as img:
        # 1. 强制转为 RGB 模式，防止透明背景被模型当成颜色特征
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 将透明通道作为 mask 贴到白底上
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        max_dim = max(width, height)

        if max_dim <= MAX_SIZE:
            # 无需缩小，直接复制
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path)
            return f"复制 (无需缩小)  {width}x{height}"

        # 2. 用 LANCZOS 算法等比缩放
        ratio = MAX_SIZE / max_dim
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized = img.resize((new_width, new_height), Image.LANCZOS)

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. 根据后缀名智能保存，防止 PNG 保存报错
        ext = dst_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            resized.save(dst_path, quality=95, optimize=True)
        else:
            resized.save(dst_path, optimize=True)

        return f"缩放 {width}x{height} → {new_width}x{new_height}"

def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)

    if not src.exists():
        print(f"[错误] 源目录不存在: {src}")
        return

    # 收集所有图片文件
    images: list[Path] = []
    for ext in EXTENSIONS:
        images.extend(src.rglob(f"*{ext}"))
        images.extend(src.rglob(f"*{ext.upper()}"))

    # 去重并排序
    images = sorted(set(images))

    if not images:
        print("[提示] 未找到任何图片文件")
        return

    print(f"找到 {len(images)} 张图片，开始处理...\n")

    copied_count = 0
    resized_count = 0
    error_count = 0

    for img_path in images:
        rel_path = img_path.relative_to(src)
        dst_path = dst / rel_path

        try:
            result = process_image(img_path, dst_path)
            print(f"  [{rel_path}]  {result}")
            if "缩放" in result:
                resized_count += 1
            else:
                copied_count += 1
        except Exception as e:
            print(f"  [{rel_path}]  错误: {e}")
            error_count += 1

    print(f"\n========== 完成 ==========")
    print(f"  已缩放: {resized_count} 张")
    print(f"  已复制: {copied_count} 张")
    print(f"  失败:   {error_count} 张")
    print(f"  输出目录: {dst}")

if __name__ == "__main__":
    main()