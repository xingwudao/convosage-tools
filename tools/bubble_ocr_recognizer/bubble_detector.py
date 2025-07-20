# -*- coding: utf-8 -*-
'''
气泡检测器
CLASS：
0：light/dark bubble
1：others
2：green bubble

models:
1. light mode: yolov8n.pt-light_bubble_train.pt
2. dark mode: yolov8n.pt-dark_bubble_train_1.pt
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
import torch
from PIL import Image
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR

class BubbleDetector:
    def __init__(self):
        """初始化气泡检测器"""
        # 项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 加载模型
        self.light_model = YOLO(os.path.join(self.project_root, "runs/detect/light_bubble_train/weights/best.pt"))
        self.dark_model = YOLO(os.path.join(self.project_root, "runs/detect/dark_bubble_train_1/weights/best.pt"))
        
        # 定义不同模式下的类别名称
        self.light_class_names = ['light_bubble', 'others', 'green_bubble']
        self.dark_class_names = ['dark_bubble', 'others', 'green_bubble']
        self.conf_threshold = 0.25  # 恢复原始置信度阈值
        print("✅ 模型加载完成")

    def detect_bubbles(self, image_path, mode):
        """检测气泡"""

        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 计算图像特征
        brightness = np.mean(image)
        variance = np.std(image)

        if mode == 'dark':
            model = self.dark_model
            # 仅对深色图像进行简单的亮度增强
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            print(f"使用深色模式模型 (亮度: {brightness:.1f}, 方差: {variance:.1f})")
        else:
            model = self.light_model
            print(f"使用浅色模式模型 (亮度: {brightness:.1f}, 方差: {variance:.1f})")
        

        # 进行推理
        results = model.predict(image, conf=self.conf_threshold)

        # 处理结果
        bubbles = []
        # 根据模式选择正确的类别名称
        class_names = self.dark_class_names if mode == 'dark' else self.light_class_names
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                if class_id < len(class_names):
                    bubbles.append({
                        "class": class_names[class_id],
                        "confidence": box.conf,
                        "bbox": box.xyxy
                    })

        return bubbles
    def draw_bubbles(self, image_path, output_path, bubbles):
        """
        for debugging
        在图像上绘制检测到的气泡
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            bubbles: 检测到的气泡列表
        """

        # 读取图像
        image = cv2.imread(image_path)
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i, bubble in enumerate(bubbles):
            try:
                # 转换bbox Tensor为列表并解包
                bbox = bubble['bbox'].squeeze().tolist()
                if len(bbox) != 4:
                    print(f"警告: 气泡{i}的bbox形状不正确: {bbox}, 跳过")
                    continue
                    
                x1, y1, x2, y2 = map(int, bbox)
                class_name = bubble['class']
                confidence = float(bubble['confidence'])
            
            # 绘制边界框
                if bubble['class'] == 'green_bubble':
                    # 用户发送气泡，红色框
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                elif bubble['class'] == 'light_bubble':
                    # 对方发送气泡，绿色框
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                elif bubble['class'] == 'dark_bubble':
                    # 对方发送气泡，绿色框
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                else:
                    # 其他气泡，黄色框
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
            except Exception as e:
                print(f"绘制气泡{i}时出错: {e}")
                continue
        # 保存图像
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"检测结果已保存到: {output_path}")

def main():
    """主函数：演示如何使用气泡检测器"""
    # 初始化检测器
    detector = BubbleDetector()
    
    while True:
        print("\n=== 气泡检测器 ===")
        print("1. 深色模式")
        print("2. 浅色模式")
        print("q. 退出")
        
        choice = input("请选择模式 (1/2/q): ").strip().lower()
        
        if choice == 'q':
            print("退出程序")
            break
            
        if choice not in ['1', '2']:
            print("无效的选择，请重试")
            continue

        if choice == '1':
            print("使用深色模式")
            mode = 'dark'

        else:
            print("使用浅色模式")
            mode = 'light'
            
        # 获取图片路径
        image_path = input("请输入图片完整路径: ").strip().strip('"').strip("'")
        if not os.path.exists(image_path):
            print(f"错误: 图片不存在 - {image_path}")
            continue
            
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(image_path), 'debug')
        os.makedirs(output_dir, exist_ok=True)
        
        # 记录开始时间
        start_time = time.time()
        
        print("\n开始检测...")
        # 检测气泡
        bubbles = detector.detect_bubbles(image_path, mode='dark' if choice == '1' else 'light')
        print(f"检测到 {len(bubbles)} 个气泡")
        
        # 输出每个气泡的详细信息
        for bubble in bubbles:
            try:
                conf = float(bubble['confidence'])
                bbox = bubble['bbox'].squeeze().tolist()
                print(f"  类别: {bubble['class']}, 置信度: {conf:.2f}, 边界框: {bbox}")
            except Exception as e:
                print(f"格式化输出错误: {e}")
        
        # 可视化结果
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_filename}_detected.jpg')
        detector.draw_bubbles(image_path, output_path, bubbles)
        
        # 输出统计信息
        end_time = time.time()
        print(f"\n处理完成!")
        print(f"用时: {end_time - start_time:.2f} 秒")
        print(f"检测结果已保存到: {output_path}")
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset', 'test')
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录不存在 - {image_dir}")
        return
    
    # 获取目录中的所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"错误: 目录 {image_dir} 中没有图像文件")
        return
    
    # 确保debug目录存在
    debug_dir = os.path.join(image_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # 处理每一张图片
    total_bubbles = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\n处理图像: {image_file}")
        
        # 检测气泡
        bubbles = detector.detect_bubbles(image_path)
        total_bubbles += len(bubbles)
        print(f"检测到 {len(bubbles)} 个气泡")
        
        # 输出每个气泡的详细信息
        for bubble in bubbles:
            try:
                # 处理置信度
                conf = float(bubble['confidence'])
            
                # 处理边界框
                if bubble['bbox'].dim() == 0:  # 标量
                    bbox = [float(bubble['bbox'])]
                elif bubble['bbox'].numel() == 1:  # 单元素Tensor
                    bbox = [float(bubble['bbox'])]
                else:  # 多元素Tensor
                    bbox = [float(x) for x in bubble['bbox'].flatten().tolist()]
                
                print(f"  类别: {bubble['class']}, 置信度: {conf:.2f}, 边界框: {bbox}")
            except Exception as e:
                print(f"格式化输出错误: {e}")
                print(f"原始数据 - 类别: {bubble['class']}, 置信度: {bubble['confidence']}, 边界框: {bubble['bbox']}")
        
        # 可视化结果
        base_filename = os.path.splitext(image_file)[0]
        output_path = os.path.join(debug_dir, f'{base_filename}_detected.jpg')
        detector.draw_bubbles(image_path, output_path, bubbles)
    
    # 输出总体统计信息
    end_time = time.time()
    total_time = end_time - start_time
    print("\n统计信息:")
    print(f"处理图片总数: {len(image_files)}")
    print(f"检测到气泡总数: {total_bubbles}")
    print(f"总用时: {total_time:.2f} 秒")
    print(f"平均每张用时: {total_time/len(image_files):.2f} 秒")
    print(f"检测结果已保存到: {debug_dir}")
if __name__ == "__main__":
    main()