import os
from pathlib import Path
from PIL import Image

BASE_DIR = r"<数据集根目录>\process\Character Art"
DST_BASE_DIR = r"<数据集根目录>\process\Character Art"
EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_RATIO = 0.005  # 默认裁剪比例 3.5%

# 裁剪任务配置列表
# sides: 指定要裁剪的方向组合，支持 "top", "bottom", "left", "right" 任意组合
CROP_TASKS = [
    {"folder": "上", "sides": ["top"], "ratio": 0.10},
    {"folder": "下", "sides": ["bottom"], "ratio": 0.10},
    # 可继续添加更多文件夹...
    # {"folder": "MNO", "sides": ["right"], "ratio": 0.05},  # 支持单独设置比例
]


def crop_image(src_path: Path, dst_path: Path, sides: list[str], ratio: float) -> str:
    """
    根据指定方向裁剪图片。
    sides: 包含 "top"/"bottom"/"left"/"right" 的列表
    ratio: 裁剪比例 (0~0.5)
    """
    with Image.open(src_path) as img:
        width, height = img.size
        
        # 计算各方向裁剪像素数
        crop_h = int(height * ratio)
        crop_w = int(width * ratio)
        
        left = crop_w if "left" in sides else 0
        top = crop_h if "top" in sides else 0
        right = (width - crop_w) if "right" in sides else width
        bottom = (height - crop_h) if "bottom" in sides else height
        
        # 安全检查
        if right <= left or bottom <= top:
            return f"跳过 (尺寸不足) {width}x{height}"
        
        cropped = img.crop((left, top, right, bottom))
        new_w, new_h = cropped.size
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_kwargs = {"optimize": True}
        if src_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95
        cropped.save(dst_path, **save_kwargs)
        
        side_str = "+".join(sides)
        return f"[{side_str}] {width}x{height} → {new_w}x{new_h}"

def process_folder(task: dict):
    """处理单个文件夹的裁剪任务"""
    folder_name = task["folder"]
    sides = task["sides"]
    ratio = task.get("ratio", DEFAULT_RATIO)
    
    src = Path(BASE_DIR) / folder_name
    dst = Path(DST_BASE_DIR) / folder_name
    
    if not src.exists():
        print(f"\n[警告] 文件夹不存在，跳过: {src}")
        return
    
    # 收集图片（去重）
    images = set()
    for ext in EXTENSIONS:
        images.update(src.rglob(f"*{ext}"))
        images.update(src.rglob(f"*{ext.upper()}"))
    images = sorted(images)
    
    if not images:
        print(f"\n[提示] {folder_name}: 未找到图片")
        return
    
    side_desc = "+".join(sides)
    print(f"\n{'='*50}")
    print(f"📁 {folder_name} | 裁剪方向: {side_desc} | 比例: {ratio*100:.1f}%")
    print(f"   共 {len(images)} 张图片")
    print(f"{'='*50}")
    
    success = skip = error = 0
    for img_path in images:
        rel = img_path.relative_to(src)
        dst_path = dst / rel
        try:
            result = crop_image(img_path, dst_path, sides, ratio)
            print(f"  ✅ {rel}  {result}")
            if "跳过" in result:
                skip += 1
            else:
                success += 1
        except Exception as e:
            print(f"  ❌ {rel}  错误: {e}")
            error += 1
    
    print(f"  📊 完成: {success} | 跳过: {skip} | 失败: {error}")

def main():
    if not CROP_TASKS:
        print("[错误] CROP_TASKS 为空，请配置裁剪任务")
        return
    
    print(f"🚀 批量裁剪工具启动")
    print(f"   源目录: {BASE_DIR}")
    print(f"   输出目录: {DST_BASE_DIR}")
    print(f"   任务数: {len(CROP_TASKS)}")
    
    for task in CROP_TASKS:
        process_folder(task)
    
    print(f"\n{'='*50}")
    print("🎉 全部任务完成!")

if __name__ == "__main__":
    main()