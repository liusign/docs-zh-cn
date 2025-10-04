import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# 定义Cityscapes类别和颜色映射
classes_data = [
    {"name": "unlabeled", "color": (0, 0, 0), "id": 0},
    {"name": "ego vehicle", "color": (0, 0, 0), "id": 1},
    {"name": "rectification border", "color": (0, 0, 0), "id": 2},
    {"name": "out of roi", "color": (0, 0, 0), "id": 3},
    {"name": "static", "color": (0, 0, 0), "id": 4},
    {"name": "dynamic", "color": (111, 74, 0), "id": 5},
    {"name": "ground", "color": (81, 0, 81), "id": 6},
    {"name": "road", "color": (128, 64, 128), "id": 7},
    {"name": "sidewalk", "color": (244, 35, 232), "id": 8},
    {"name": "parking", "color": (250, 170, 160), "id": 9},
    {"name": "rail track", "color": (230, 150, 140), "id": 10},
    {"name": "building", "color": (70, 70, 70), "id": 11},
    {"name": "wall", "color": (102, 102, 156), "id": 12},
    {"name": "fence", "color": (190, 153, 153), "id": 13},
    {"name": "guard rail", "color": (180, 165, 180), "id": 14},
    {"name": "bridge", "color": (150, 100, 100), "id": 15},
    {"name": "tunnel", "color": (150, 120, 90), "id": 16},
    {"name": "pole", "color": (153, 153, 153), "id": 17},
    {"name": "polegroup", "color": (153, 153, 153), "id": 18}
]

# 创建初始DataFrame
df = pd.DataFrame(classes_data)

# 设置图片文件夹路径
image_folder = "./runs/detect/exp22"  # 请替换为您的图片文件夹路径

# 检查文件夹是否存在
if not os.path.exists(image_folder):
    print(f"错误: 文件夹 '{image_folder}' 不存在")
    exit(1)

# 获取所有jpg图片
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

if len(image_files) == 0:
    print(f"警告: 在 '{image_folder}' 中没有找到.jpg图片")
    exit(1)

print(f"找到 {len(image_files)} 张图片")

# 为每张图片创建一个字典来存储颜色占比
image_stats = defaultdict(dict)

# 处理每张图片
for img_file in tqdm(image_files, desc="处理图片"):
    img_path = os.path.join(image_folder, img_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图片: {img_file}")
        continue
    
    # 转换颜色空间为RGB（OpenCV默认是BGR）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 计算总像素数
    total_pixels = img.shape[0] * img.shape[1]
    
    # 初始化当前图片的颜色统计
    img_name = os.path.splitext(img_file)[0]
    image_stats[img_name]['total_pixels'] = total_pixels
    
    # 对每个类别统计像素数
    for class_info in classes_data:
        color = class_info['color']
        # 创建颜色掩码
        mask = np.all(img_rgb == color, axis=-1)
        pixel_count = np.sum(mask)
        percentage = (pixel_count / total_pixels) * 100
        
        # 存储结果
        class_name = class_info['name']
        image_stats[img_name][class_name] = percentage

# 将统计结果添加到DataFrame
# 修复：移除了不必要的内层循环
for img_name, stats in image_stats.items():
    df[img_name] = df['name'].map(lambda x: stats.get(x, 0))

# 重新排列列，将图片名列放在id列之后
cols = ['name', 'color', 'id'] + list(image_stats.keys())
df = df[cols]

# 显示结果
print("\n统计结果预览:")
print(df.head(10))

# 显示每张图片的总百分比（用于验证）
print("\n每张图片的总百分比（应该接近100%）:")
for img_name in list(image_stats.keys())[:5]:  # 只显示前5张
    total_percentage = df[img_name].sum()
    print(f"{img_name}: {total_percentage:.2f}%")

# 保存结果到CSV文件
output_file = 'image_color_statistics.csv'
df.to_csv(output_file, index=False)
print(f"\n结果已保存到: {output_file}")
