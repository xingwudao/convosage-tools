#-- coding: utf-8 -*-
'''
整图OCR处理+气泡匹配

'''
import cv2
import numpy as np
import os
import json
import logging
from bubble_detector import BubbleDetector

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("警告: PaddleOCR未安装")


class FullImageOCRProcessor:
    def __init__(self, detect_threshold=0.5, ocr_confidence=0.1, min_text_length=1):
        self.detector = BubbleDetector()
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr_engine = PaddleOCR(
                    use_textline_orientation=True,
                    lang= ["ch", "en"],
                    text_det_thresh=0.2,  # 降低检测阈值（默认0.3），允许更多候选文本区域
                    text_det_box_thresh=0.2,  # 文本框置信度阈值（补充此参数，默认0.5，需降低）
                    text_det_unclip_ratio=2.5,  # 扩大检测框范围（默认2.0），避免漏检边缘文字
                    text_recognition_batch_size=6,
)
                print("✅ PaddleOCR模型加载成功")
            except Exception as e:
                print(f"PaddleOCR模型加载失败: {e}")
                self.ocr_engine = None
        else:
            self.ocr_engine = None
        self.detect_threshold = detect_threshold
        self.ocr_confidence = ocr_confidence  # 单个文本块最低置信度阈值
        self.min_text_length = min_text_length  # 单个文本块最小长度
        self.min_bubble_area = 1000
        logging.basicConfig(level=logging.INFO)

    def process_image(self, image_path):
        # 1. 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 2. 检测气泡（获取所有气泡的边界框）
        mode = self._determine_image_mode(image)
        bubbles = self.detector.detect_bubbles(image_path, mode)
        if not bubbles:
            logging.info("未检测到气泡，无需处理文本")
            return []
        
        # 3. 整图OCR，获取所有文本块（先于气泡匹配，符合"先整图OCR"逻辑）
        if self.ocr_engine is None:
            logging.error("OCR引擎未初始化，无法执行整图识别")
            return []
        print("开始整图OCR，获取所有文本块...")
        ocr_results = self.ocr_engine.predict(image)
        text_blocks = self._parse_ocr_results(ocr_results)  # 解析为统一格式的文本块列表
        print(f"整图OCR完成，共检测到 {len(text_blocks)} 个有效文本块")

        # 4. 为每个气泡创建"识别队列"（存储属于该气泡的文本块）
        bubble_queues = {i: [] for i in range(len(bubbles))}  # 键：气泡索引，值：文本块列表

        # 5. 遍历文本块，中心点落在任意气泡框内即可归属，支持跳过其他类别气泡
        for block in text_blocks:
            cx, cy = block["center"]
            matched = False
            for bubble_idx, bubble in enumerate(bubbles):
                bubble_bbox = self._parse_bubble_bbox(bubble["bbox"])
                x1, y1, x2, y2 = bubble_bbox
                # 跳过其他类别气泡
                if bubble.get("class", "") == "others":
                    continue
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    bubble_queues[bubble_idx].append(block)
                    matched = True
                    break
            if not matched:
                # 可在此处处理未匹配文本块，如记录或输出
                pass

        # 6. 处理每个气泡的识别队列，生成最终结果
        results = []
        debug_dir = "debug_output_fullimg"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 6. 处理每个气泡的识别队列，生成最终结果（每个气泡只输出一次）
        for bubble_idx, bubble in enumerate(bubbles):
            bubble_queue = bubble_queues[bubble_idx]
            if not bubble_queue:
                print(f"气泡 {bubble_idx+1} 无匹配文本块，跳过")
                continue
            # 过滤低置信度/短文本的文本块
            valid_blocks = [
                b for b in bubble_queue
                if b["confidence"] >= self.ocr_confidence 
                and len(b["text"]) >= self.min_text_length
            ]
            if not valid_blocks:
                print(f"气泡 {bubble_idx+1} 无有效文本块（过滤后），跳过")
                continue
            if bubble["class"] == "others":
                print(f"气泡 {bubble_idx+1} 属于其他类别，跳过")
                continue
            # 按文本块位置排序（从上到下，避免拼接顺序混乱）
            valid_blocks.sort(key=lambda x: x["center"][1])  # 按y坐标升序
            # 拼接文本和计算置信度
            recognized_text = ''.join([b["text"] for b in valid_blocks])
            avg_confidence = sum(b["confidence"] for b in valid_blocks) / len(valid_blocks)
            # 生成结果字典（只添加一次）
            bubble_bbox = self._parse_bubble_bbox(bubble["bbox"])
            speaker = self._determine_speaker(bubble)
            result_dict = {
                "speaker": speaker,
                "position": bubble_bbox,
                "type": "text" if "bubble" in bubble.get("class", "") else "others",
                "detection_confidence": float(bubble["confidence"]),
                "combined_text": recognized_text,
                "combined_confidence": avg_confidence
            }
            # 检查是否已存在该气泡（位置完全一致）
            if not any(r["position"] == bubble_bbox for r in results):
                results.append(result_dict)
                print(f"✅ 气泡 {bubble_idx+1} 识别完成: '{recognized_text}'（置信度: {avg_confidence:.2f}）")

        # 按气泡 bbox 中心 y 坐标排序（从上到下）
        def bubble_center_y(bubble):
            pos = bubble.get("position", [0,0,0,0])
            return (pos[1] + pos[3]) // 2
        results.sort(key=bubble_center_y)
        # 保存结果
        self._save_results(results, image_path)
        return results

    def _parse_ocr_results(self, ocr_results):
        """解析OCR结果为统一格式的文本块列表：[{"text":..., "confidence":..., "center":(cx, cy)}, ...]"""
        text_blocks = []
        if not ocr_results:
            return text_blocks
        # PaddleOCR新版返回可能是 list[dict] 或 dict
        if isinstance(ocr_results, list) and len(ocr_results) > 0 and isinstance(ocr_results[0], dict):
            # 处理 list[dict] 格式
            for idx, res in enumerate(ocr_results):
                rec_texts = res.get('rec_texts', [])
                rec_scores = res.get('rec_scores', [])
                rec_polys = res.get('rec_polys', [])
                print(f"[调试] OCR rec_texts[{idx}]:", rec_texts)
                print(f"[调试] OCR rec_scores[{idx}]:", rec_scores)
                print(f"[调试] OCR rec_polys[{idx}]:", rec_polys)
                if rec_texts and rec_scores and rec_polys:
                    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                        text = str(text).strip()
                        try:
                            confidence = float(score)
                        except Exception:
                            confidence = 0.0
                        if not text:
                            continue
                        pts = np.array(poly, dtype=np.float32)
                        if pts.ndim != 2 or pts.shape[1] != 2:
                            continue
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))
                        text_blocks.append({
                            "text": text,
                            "confidence": confidence,
                            "center": (cx, cy)
                        })
            return text_blocks
        # PaddleOCR新版返回dict格式（包含 rec_texts、rec_scores、rec_polys）
        if isinstance(ocr_results, dict):
            rec_texts = ocr_results.get('rec_texts', [])
            rec_scores = ocr_results.get('rec_scores', [])
            rec_polys = ocr_results.get('rec_polys', [])
            print("[调试] OCR rec_texts:", rec_texts)
            print("[调试] OCR rec_scores:", rec_scores)
            print("[调试] OCR rec_polys:", rec_polys)
            if rec_texts and rec_scores and rec_polys:
                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                    text = str(text).strip()
                    try:
                        confidence = float(score)
                    except Exception:
                        confidence = 0.0
                    if not text:
                        continue
                    pts = np.array(poly, dtype=np.float32)
                    if pts.ndim != 2 or pts.shape[1] != 2:
                        continue
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    text_blocks.append({
                        "text": text,
                        "confidence": confidence,
                        "center": (cx, cy)
                    })
                return text_blocks
            # 若有 result 字段则递归处理
            ocr_results = ocr_results.get('result', [])
        if not isinstance(ocr_results, list) or len(ocr_results) < 1:
            return text_blocks
        # 兼容嵌套list格式
        blocks = ocr_results[0] if isinstance(ocr_results[0], list) else ocr_results
        for block in blocks:
            if not isinstance(block, (list, tuple)) or len(block) < 2:
                continue
            poly, rec_result = block[0], block[1]
            if not isinstance(rec_result, (list, tuple)) or len(rec_result) < 2:
                continue
            text = str(rec_result[0]).strip()
            try:
                confidence = float(rec_result[1])
            except Exception:
                confidence = 0.0
            if not text:
                continue
            pts = np.array(poly, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            text_blocks.append({
                "text": text,
                "confidence": confidence,
                "center": (cx, cy)
            })
        return text_blocks

    def _parse_bubble_bbox(self, bbox):
        """解析气泡边界框为[x1, y1, x2, y2]"""
        bbox_list = bbox.squeeze().tolist()
        if not isinstance(bbox_list[0], (int, float)):
            bbox_list = bbox_list[0]
        return [int(coord) for coord in bbox_list[:4]]  # 取前4个坐标（x1, y1, x2, y2）

    def _determine_speaker(self, bubble):
        """根据气泡类别判断说话者"""
        bubble_class = bubble.get("class", "")
        if "green" in bubble_class:
            return "user"
        elif "light" in bubble_class or "dark" in bubble_class:
            return "ta"
        return ""

    def _save_results(self, results, image_path):
        """保存识别结果到JSON文件"""
        results_dir = os.path.join(os.path.dirname(image_path), 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results_fullimg.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"识别结果已保存到: {results_path}")

    def _determine_image_mode(self, image):
        return "light"


if __name__ == "__main__":
    import datetime
    debug_dir = "debug_output_fullimg"
    output_dir = "wechat_records"
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 清空调试目录
    for f in os.listdir(debug_dir):
        try:
            os.remove(os.path.join(debug_dir, f))
        except Exception as e:
            logging.warning(f"清理调试文件时出错: {e}")
    
    # 获取输入和模式
    image_path = input("请输入测试图片路径: ").strip() or "test_comic.jpg"
    while True:
        mode = input("请选择模式 (1: 浅色, 2: 深色): ").strip()
        if mode in ('1', '2'):
            mode = 'light' if mode == '1' else 'dark'
            print(f"已选择{mode}模式")
            break
        print("无效选择，请输入 1 或 2")
    
    # 执行处理
    processor = FullImageOCRProcessor()
    processor._determine_image_mode = lambda x: mode
    result = processor.process_image(image_path)
    
    # 输出简化结果
    if result:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"wechat_fullimg_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        simplified_records = [
            {
                "speaker": record.get("speaker", "unknown"),
                "text": record.get("combined_text", "")
            } for record in result
        ]
        output = {
            "source": "WeChat",
            "image_path": image_path,
            "timestamp": timestamp,
            "records": simplified_records
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logging.info(f"微信聊天记录已保存到: {json_path}")
        
        # 打印识别结果
        print("\n识别结果汇总:")
        for i, record in enumerate(result, 1):
            print(f"\n[气泡 {i}]")
            print(f"说话者: {record['speaker']}")
            print(f"位置: {record['position']}")
            print(f"合并文本: {record['combined_text']}")
            print(f"平均置信度: {record['combined_confidence']:.2f}")
    else:
        logging.warning("未检测到有效识别结果")