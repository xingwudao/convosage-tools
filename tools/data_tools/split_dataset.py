import os
import random
import shutil

def split_dataset(image_dir, output_dir, train_ratio=0.8):
    """
    划分数据集为训练集和验证集
    Args:
        image_dir: 原始图片目录
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    # 创建目录
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

    # 获取所有已标注的图片
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    # 只保留有对应标签文件的图片
    images = []
    for img in all_images:
        label = img.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(image_dir, '../green_labels', label)  # 使用green_labels目录
        if os.path.exists(label_path):
            images.append(img)
    
    if not images:
        print("错误：没有找到已标注的图片！")
        return
        
    print(f"找到 {len(images)} 张已标注的图片")
    random.shuffle(images)
    
    # 计算划分点
    split_point = int(len(images) * train_ratio)
    train_images = images[:split_point]
    val_images = images[split_point:]
    
    # 移动文件
    for img in train_images:
        # 移动图片
        shutil.copy2(
            os.path.join(image_dir, img),
            os.path.join(output_dir, 'images/train', img)
        )
        # 移动标签（如果存在）
        label = img.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(image_dir, '../green_labels', label)  # 使用green_labels目录
        if os.path.exists(label_path):
            shutil.copy2(
                label_path,
                os.path.join(output_dir, 'labels/train', label)
            )
    
    for img in val_images:
        # 移动图片
        shutil.copy2(
            os.path.join(image_dir, img),
            os.path.join(output_dir, 'images/val', img)
        )
        # 移动标签（如果存在）
        label = img.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(image_dir, '../green_labels', label)  # 使用green_labels目录
        if os.path.exists(label_path):
            shutil.copy2(
                label_path,
                os.path.join(output_dir, 'labels/val', label)
            )
    
    print(f"数据集划分完成:")
    print(f"训练集: {len(train_images)} 张图片")
    print(f"验证集: {len(val_images)} 张图片")

if __name__ == "__main__":
    # 设置路径
    dataset_dir = "dataset"
    image_dir = os.path.join(dataset_dir, "green_images")
    output_dir = os.path.join(dataset_dir, "split_green_dataset")

    # 执行划分
    split_dataset(image_dir, output_dir)
