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
    train_img_dir = os.path.join(output_dir, 'dark_images/train')
    val_img_dir = os.path.join(output_dir, 'dark_images/val')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # 清空输出目录中的旧文件
    for dir_path in [train_img_dir, val_img_dir]:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'清理旧文件时出错: {e}')

    # 获取所有已标注的图片
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    # 只保留有对应标签文件的图片
    images = []
    for img in all_images:
        label = img.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(image_dir, '../dark_labels', label)
        if os.path.exists(label_path):
            images.append(img)

        # 移动标签（如果存在）
        label = img.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(image_dir, '../dark_labels', label)
        if os.path.exists(label_path):
            shutil.copy2(
                label_path,
                os.path.join(output_dir, 'dark_images/train', label)
            )
    """
    划分数据集为训练集和验证集
    Args:
        image_dir: 原始图片目录
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    # 创建目录
    os.makedirs(os.path.join(output_dir, 'dark_images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dark_images/val'), exist_ok=True)

    # 获取所有已标注的图片
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    # 只保留有对应标签文件的图片
    images = []
    for img in all_images:
        label = img.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(image_dir, '../dark_labels', label)
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
            os.path.join(output_dir, 'dark_images/train', img)
        )
        # 复制图片
        src_img = os.path.join(image_dir, img)
        dst_img = os.path.join(train_img_dir, img)
        shutil.copy2(src_img, dst_img)
        
        # 复制标签
        label = img.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(image_dir, '../dark_labels', label)
        dst_label = os.path.join(train_img_dir, label)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"警告：找不到训练图片 {img} 的标签文件")
    
    # 复制验证集文件
    for img in val_images:
        # 复制图片
        src_img = os.path.join(image_dir, img)
        dst_img = os.path.join(val_img_dir, img)
        shutil.copy2(src_img, dst_img)
        
        # 复制标签
        label = img.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(image_dir, '../dark_labels', label)
        dst_label = os.path.join(val_img_dir, label)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"警告：找不到验证图片 {img} 的标签文件")
    
    print(f"\n数据集划分完成:")
    print(f"训练集: {len(train_images)} 张图片")
    print(f"验证集: {len(val_images)} 张图片")
    print(f"\n训练集目录: {train_img_dir}")
    print(f"验证集目录: {val_img_dir}")

if __name__ == "__main__":
    # 设置路径
    dataset_dir = "dataset"
    image_dir = os.path.join(dataset_dir, "dark_images")

    try:
        # 执行划分
        split_dataset(image_dir, dataset_dir)
    except Exception as e:
        print(f"发生错误: {str(e)}")
