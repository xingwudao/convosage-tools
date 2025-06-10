import cv2
import numpy as np
import json
import base64
import os
from paddleocr import PaddleOCR

class ChatBubbleRecognizer:
    def __init__(self, lang="ch", use_angle_cls=True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def recognize_with_speaker_from_path(self, image_path):
        img = cv2.imread(image_path)
        img = self._check_image(img)
        return self._recognize_with_speaker_core(img)

    def recognize_with_speaker_from_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = self._check_image(img)
        return self._recognize_with_speaker_core(img)

    def recognize_with_speaker_from_base64(self, image_base64):
        image_bytes = base64.b64decode(image_base64)
        return self.recognize_with_speaker_from_bytes(image_bytes)

    def _recognize_with_speaker_core(self, img):
        if img is None:
            return json.dumps({"error": "图片读取失败"}, ensure_ascii=False)
        results = []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 白色气泡
        lower_white = np.array([0, 0, 200])  # 提高白色阈值
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        kernel_open = np.ones((3,3), np.uint8)
        kernel_close = np.ones((5,5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_open)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_close)
        contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_rects = []
        h_img, w_img = img.shape[:2]
        # 增加白色气泡最小面积限制
        for cnt in contours_white:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 1000 or area > 0.5 * w_img * h_img:  # 过滤太小和太大的区域
                continue
            white_rects.append((x, y, w, h))

        # 绿色气泡参数优化
        lower_green = np.array([35, 40, 200])  # 提高绿色亮度阈值
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_rects = []
        # 调整过滤条件
        for cnt in contours_green:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if w < 40 or h < 25 or area < 1000:  # 适当提高最小尺寸要求
                continue
            green_rects.append((x, y, w, h))
        # 合并相近的绿色气泡
        #green_rects = merge_rects(green_rects, y_thresh=20, x_thresh=15)

        # 橙色气泡（转账/红包等）
        lower_orange = np.array([10, 100, 180])
        upper_orange = np.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orange_rects = []
        for cnt in contours_orange:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if w < 60 or h < 30 or area < 2000:
                continue
            orange_rects.append((x, y, w, h))
        # orange_rects = merge_rects(orange_rects)  # 先注释掉合并

        # 统一收集所有气泡及说话人
        all_bubbles = []
        for x, y, w, h in green_rects:
            all_bubbles.append({"speaker": "B", "position": (x, y, w, h)})
        for x, y, w, h in white_rects:
            all_bubbles.append({"speaker": "A", "position": (x, y, w, h)})
        for x, y, w, h in orange_rects:
            all_bubbles.append({"speaker": "A", "position": (x, y, w, h)})  # 转账一般属于A

        # 按y坐标排序，保持聊天顺序
        all_bubbles = sorted(all_bubbles, key=lambda r: r["position"][1])

        # 画出所有气泡的矩形框用于调试
        debug_img = img.copy()
        for x, y, w, h in green_rects + white_rects + orange_rects:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite("debug_bubbles.jpg", debug_img)

        # 统一做OCR
        for bubble in all_bubbles:
            x, y, w, h = bubble["position"]
            # 扩大裁剪区域，避免文字被截断
            x = max(0, x - 2)
            y = max(0, y - 2)
            w = min(w + 4, img.shape[1] - x)
            h = min(h + 4, img.shape[0] - y)
            
            crop = img[y:y+h, x:x+w]
            if len(crop.shape) == 2:  # 如果是单通道图像
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            elif crop.shape[2] == 4:  # 如果是RGBA图像
                crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
                
            # 图像预处理
            crop = self._preprocess_for_ocr(crop)
            
            try:
                ocr_result = self.ocr.ocr(crop)  # 关闭方向分类，提高速度
            except Exception as e:
                print(f"OCR错误：{str(e)}")
                continue
            
            rec_texts = []
            # Process OCR results based on different output formats
            if ocr_result and len(ocr_result) > 0:
                img_result = ocr_result[0]
                if isinstance(img_result, list):
                    # 按y坐标排序确保文本顺序正确
                    img_result = sorted(img_result, key=lambda x: x[0][0][1])
                    # 过滤低置信度的结果
                    rec_texts = [line[1][0] for line in img_result 
                                if len(line) >= 1 and float(line[1][1]) > 0.5]  # 添加置信度阈值
                elif isinstance(img_result, dict) and 'rec_texts' in img_result:
                    # Method 2: Extract from dictionary keys (for document analysis model output)
                    rec_texts = img_result['rec_texts']
                else:
                    # Sort text by y-coordinate if needed
                    img_result = sorted(img_result, key=lambda x: x[0][0][1])
                    rec_texts = [line[1][0] for line in img_result if len(line) >= 2]

            recognized_text = "\n".join(rec_texts).strip()
            if recognized_text:
                results.append({
                    "speaker": bubble["speaker"],
                    "content": recognized_text,
                    "position": [int(x), int(y), int(w), int(h)]
                })

        return json.dumps({"lines": results}, ensure_ascii=False, indent=2)

    def _preprocess_for_ocr(self, crop):
        """OCR前的图像预处理"""
        # 图像去噪
        denoised = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)
        
        # 提高对比度
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 二值化
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _debug_save_masks(self, mask_white, mask_green, debug_dir="debug"):
        """保存中间结果用于调试"""
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(f"{debug_dir}/mask_white.jpg", mask_white)
        cv2.imwrite(f"{debug_dir}/mask_green.jpg", mask_green)

    def _debug_draw_rects(self, img, white_rects, green_rects, debug_dir="debug"):
        """绘制检测到的矩形框"""
        debug_img = img.copy()
        # 白色气泡用红色框
        for x, y, w, h in white_rects:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # 绿色气泡用蓝色框
        for x, y, w, h in green_rects:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(f"{debug_dir}/debug_bubbles.jpg", debug_img)

    def _check_image(self, img):
        """检查图像格式并进行必要的转换"""
        if img is None:
            return None
            
        # 如果图片太大，进行等比例缩放
        max_side = 2000  # 设置最大边长
        height, width = img.shape[:2]
        if max(height, width) > max_side:
            scale = max_side / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            
        # 确保图像是BGR格式
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img


if __name__ == "__main__":
    recognizer = ChatBubbleRecognizer()
    
    try:
        print("请选择输入方式:")
        print("1. 文件路径")
        print("2. 字节流")
        print("3. base64编码")
        choice = input("请输入选择(1-3): ")

        if choice == "1":
            image_path = input("请输入图片路径: ")
            if not os.path.exists(image_path):
                print("错误：文件不存在！")
                exit(1)
            print("\n文件路径识别结果:")
            print(recognizer.recognize_with_speaker_from_path(image_path))
        
        elif choice == "2":
            image_path = input("请输入图片路径: ")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            print("\n字节流识别结果:")
            print(recognizer.recognize_with_speaker_from_bytes(image_bytes))
        
        elif choice == "3":
            image_path = input("请输入图片路径: ")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode()
            print("\nbase64识别结果:")
            print(recognizer.recognize_with_speaker_from_base64(image_base64))
        
        else:
            print("无效的选择!")
    
    except Exception as e:
        print(f"发生错误：{str(e)}")