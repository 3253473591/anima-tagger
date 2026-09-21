import os
from PIL import Image

def add_background_to_image(img_path, output_path, bg_color=(255, 255, 255)):
    """
    为单张透明 PNG 添加有色背景并保存

    Args:
        img_path (str): 输入图片路径
        output_path (str): 输出图片路径（必须是完整文件路径）
        bg_color (tuple): RGB 背景色
    """
    try:
        img = Image.open(img_path).convert('RGBA')
    except Exception as e:
        print(f"  跳过文件 '{img_path}': {e}")
        return False

    # 创建背景
    background = Image.new('RGBA', img.size, bg_color + (255,))
    combined = Image.alpha_composite(background, img)
    combined_rgb = combined.convert('RGB')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_rgb.save(output_path, format='PNG')
    return True


def process_path(input_path, output_path=None, bg_color=(255, 255, 255)):
    """
    智能处理输入路径（文件或文件夹）

    Args:
        input_path (str): 输入路径，可以是 PNG 文件或文件夹
        output_path (str, optional): 输出路径，可以是文件或文件夹。
                                     - 若 input 是文件，output 可以是文件或文件夹（缺省则自动生成）
                                     - 若 input 是文件夹，output 必须是文件夹（缺省则自动创建）
        bg_color (tuple): RGB 背景色
    """
    if not os.path.exists(input_path):
        print(f"错误：输入路径不存在 -> {input_path}")
        return

    # ---------- 单张图片模式 ----------
    if os.path.isfile(input_path):
        # 检查扩展名
        if not input_path.lower().endswith('.png'):
            print(f"警告：'{input_path}' 不是 PNG 文件，跳过。")
            return

        # 确定输出路径
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_bg{ext}"
        elif os.path.isdir(output_path):
            # 输出是文件夹，则保留原文件名
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_path, filename)

        # 执行单张处理
        print(f"处理单张图片: {input_path} -> {output_path}")
        success = add_background_to_image(input_path, output_path, bg_color)
        if success:
            print("  完成")
        return

    # ---------- 文件夹模式 ----------
    if os.path.isdir(input_path):
        # 确定输出文件夹
        if output_path is None:
            output_path = os.path.join(input_path, "output_bg")
        elif os.path.isfile(output_path):
            print("错误：输入为文件夹时，输出路径必须为文件夹（或留空自动创建）")
            return

        # 确保输出文件夹存在
        os.makedirs(output_path, exist_ok=True)

        # 遍历输入文件夹下的所有 PNG 文件
        png_files = [f for f in os.listdir(input_path)
                     if f.lower().endswith('.png') and os.path.isfile(os.path.join(input_path, f))]

        if not png_files:
            print(f"在 '{input_path}' 中未找到任何 PNG 文件。")
            return

        print(f"批量处理模式：找到 {len(png_files)} 个 PNG 文件")
        success_count = 0
        for filename in png_files:
            src = os.path.join(input_path, filename)
            dst = os.path.join(output_path, filename)   # 保持同名
            print(f"  处理: {filename}")
            if add_background_to_image(src, dst, bg_color):
                success_count += 1
        print(f"批量处理完成，成功 {success_count}/{len(png_files)} 个文件，输出目录：{output_path}")
        return

    print(f"错误：'{input_path}' 既不是文件也不是文件夹")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # ----- 请在此处修改配置 -----
    INPUT = r"<输入目录或单个 PNG 文件>"           # 可以是单个PNG文件，也可以是文件夹路径
    OUTPUT = r"<输出目录>"                 # 可为None（自动生成）、文件路径或文件夹路径
    BG_COLOR = (0, 0, 0)    # 黑色背景
    # -----------------------------

    process_path(INPUT, OUTPUT, BG_COLOR)