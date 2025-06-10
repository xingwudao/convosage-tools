from paddleocr import PaddleOCR
import json
import base64
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import emoji
import re

class ChatTextRecognizer:
    def __init__(self, lang="ch", use_angle_cls=True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        self.x_threshold = None
        self.current_message = None
        self.current_image = None

    def recognize_with_speaker_from_path(self, image_path):
        """从图片文件路径识别文本"""
        img = cv2.imread(image_path)
        if img is None:
            return json.dumps({"error": f"无法读取图片：{image_path}"}, ensure_ascii=False)
        return self._recognize_with_speaker_core(img)

    def recognize_with_speaker_from_bytes(self, image_bytes):
        """从字节数据识别文本"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._recognize_with_speaker_core(img)

    def recognize_with_speaker_from_base64(self, image_base64):
        """从base64编码识别文本"""
        image_bytes = base64.b64decode(image_base64)
        return self.recognize_with_speaker_from_bytes(image_bytes)

    def _is_image_content(self, box, img):
        """检测指定区域是否包含图片内容"""
        try:
            # 将坐标转换为整数
            x1, y1 = int(min(p[0] for p in box)), int(min(p[1] for p in box))
            x2, y2 = int(max(p[0] for p in box)), int(max(p[1] for p in box))

            # 确保坐标在图片范围内
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return False

            # 提取区域
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                return False

            # 计算区域的一些特征
            # 1. 计算方差（图片区域通常有更高的方差)
            std_dev = np.std(roi)
            # 2. 计算梯度（图片区域通常有更多的边缘)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            avg_gradient = np.mean(gradient_mag)

            # 更新判断条件，提高准确性
            area = (x2 - x1) * (y2 - y1)
            img_area = w * h
            area_ratio = area / img_area

            # 图片通常占据较大区域，且有较高的复杂度
            return (std_dev > 30 and
                    avg_gradient > 20 and
                    area_ratio > 0.03 and  # 至少占图片面积的3%
                    (x2 - x1) > w * 0.2 and  # 宽度至少为图片宽度的20%
                    (y2 - y1) > 80)  # 高度至少80像素

        except Exception as e:
            print(f"图片检测出错：{str(e)}")
            return False

    def _detect_message_type(self, text, box):
        """检测消息类型，支持文本、表情、图片、转账和红包"""
        if text is None:
            return "text"
            
        cleaned_text = text.strip()
        if cleaned_text and emoji.emoji_count(cleaned_text) == len(cleaned_text):
            return "emoji"

        if box is not None and self.current_image is not None and self._is_image_content(box, self.current_image):
            return "image"

        # 识别转账和红包
        transfer_pattern = r'[\d.,]+\.?\d*元?|\¥[\d.,]+\.?\d*|转账|已收钱|已退还|已收款|农信转账'
        red_envelope_pattern = r'红包'
        if re.search(transfer_pattern, cleaned_text):
            return "transfer"
        elif re.search(red_envelope_pattern, cleaned_text):
            return "red_envelope"

        return "text"

    def _is_timestamp(self, text):
        """检查文本是否为时间戳"""
        timestamp_patterns = [
            r'\d{1,2}:\d{2}',  # 12:34
            r'周[一二三四五六日] \d{1,2}:\d{2}',  # 周二 11:12
            r'星期[一二三四五六日] \d{1,2}:\d{2}',  # 星期二 11:12
            r'昨天 \d{1,2}:\d{2}',  # 昨天 12:34
            r'今天 \d{1,2}:\d{2}'  # 今天 12:34
        ]
        return any(re.search(pattern, text.strip()) for pattern in timestamp_patterns)

    def _is_quote(self, text):
        """检查是否为引用消息"""
        # 微信中的引用格式是"用户名/昵称：消息内容"
        quote_pattern = r'^[^：]+：'  # 匹配任意非冒号字符后跟冒号
        return bool(re.match(quote_pattern, text.strip()))

    def _clean_quote(self, text):
        """清理引用标记（移除用户名和冒号）"""
        quote_pattern = r'^[^：]+：'
        return re.sub(quote_pattern, '', text.strip())

    def _should_merge_with_previous(self, current_text, is_on_left):
        """判断是否应该与前一条消息合并"""
        if not self.current_message:
            return False

        # 获取当前消息类型和前一条消息类型
        try:
            current_type = self._detect_message_type(current_text, None)
            prev_type = self.current_message.get('message_type')
            
            # 如果是表情符号，直接合并
            if current_type == 'emoji' and prev_type == 'emoji':
                self.current_message['content'] += current_text
                return True
                
            # 如果当前消息和前一条消息都是转账相关，允许合并
            if current_type == 'transfer' and prev_type == 'transfer':
                return True
                
            # 如果是时间戳，不合并
            if self._is_timestamp(current_text):
                return False

            # 如果说话人不同，不合并
            if (is_on_left and self.current_message['speaker'] != 'TA') or \
                    (not is_on_left and self.current_message['speaker'] != '我'):
                return False

            # 如果当前消息是图片类型，不合并
            if prev_type == 'image':
                return False

            # 如果是引用消息且与当前消息内容相似，跳过
            if self._is_quote(current_text):
                original_text = self._clean_quote(current_text)
                similarity = self._calculate_similarity(original_text, self.current_message['content'])
                if similarity > 0.7:  # 相似度阈值
                    return False

            return True
            
        except Exception as e:
            print(f"消息合并检查时出错：{str(e)}")
            return False

    def _calculate_similarity(self, text1, text2):
        """计算两段文本的相似度"""
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        lcs = lcs_length(text1, text2)
        max_len = max(len(text1), len(text2))
        return lcs / max_len if max_len > 0 else 0

    def _process_text_block(self, text, box, img_width):
        """处理单个文本块"""
        text = text.strip()
        if not text:
            return None

        # 处理表情符号
        text = emoji.demojize(text, delimiters=("", ""))
        text = emoji.emojize(text, delimiters=("", ""))

        # 根据x坐标判断说话人
        is_on_left = box[0][0] < img_width / 2
        speaker = 'TA' if is_on_left else '我'

        # 处理时间戳
        if self._is_timestamp(text):
            if self.current_message:
                self.current_message['timestamp'] = text
            return None

        # 判断是否需要合并消息
        if self._should_merge_with_previous(text, is_on_left):
            self.current_message['content'] += text
            return None

        message = {
            'speaker': speaker,
            'content': text,
            'message_type': self._detect_message_type(text, box),
            'timestamp': None
        }

        self.current_message = message
        return message

    def _post_process_messages(self, messages):
        """对识别出的消息进行后处理"""
        if not messages:
            return messages

        result = []
        current = None

        for msg in messages:
            if not current:
                current = msg.copy()
                continue

            # 检查是否是引用消息
            if self._is_quote(msg['content']):
                original_text = self._clean_quote(msg['content'])
                # 检查是否与之前的消息重复
                skip = False
                for prev_msg in reversed(result[-3:]):  # 只检查最近的3条消息
                    if self._calculate_similarity(original_text, prev_msg['content']) > 0.7:
                        skip = True
                        break
                if skip:
                    continue            # 尝试合并消息
            if current['speaker'] == msg['speaker']:
                # 处理转账消息的合并
                if current['message_type'] == 'transfer' and msg['message_type'] == 'transfer':
                    current['content'] += ' ' + msg['content']
                    continue
                    
                # 处理普通文本消息的合并
                if current['message_type'] == msg['message_type'] == 'text':
                    # 检查当前内容是否是未完成的句子
                    is_incomplete = not any(current['content'].strip().endswith(p) for p in '。！？，.!?,:;')
                    if is_incomplete:
                        current['content'] += msg['content']
                        continue

            # 不能合并，保存当前消息并开始新消息
            result.append(current)
            current = msg.copy()

        if current:
            result.append(current)

        return result

    def _recognize_with_speaker_core(self, img):
        if img is None:
            return json.dumps({"error": "图片读取失败"}, ensure_ascii=False)

        self.current_image = img
        ocr_result = self.ocr.predict(img)
        messages = []
        self.current_message = None

        if ocr_result and len(ocr_result) > 0:
            img_result = ocr_result[0]
            img_width = img.shape[1]

            if isinstance(img_result, dict) and 'rec_texts' in img_result:
                texts = img_result['rec_texts']
                boxes = img_result['dt_polys']

                # 按y坐标排序
                blocks = sorted(zip(texts, boxes), key=lambda x: x[1][0][1])

                for text, box in blocks:
                    message = self._process_text_block(text, box, img_width)
                    if message:
                        messages.append(message)

                # 对消息进行后处理
                messages = self._post_process_messages(messages)

                # 处理时间戳
                if messages:
                    timestamp = None
                    for i, msg in enumerate(messages):
                        if self._is_timestamp(msg['content']):
                            timestamp = msg['content']
                            messages.pop(i)
                            break

                    if timestamp and messages:
                        messages[-1]['timestamp'] = timestamp

        return json.dumps({
            "messages": messages,
            "chat_app": "微信",
            "total_messages": len(messages)
        }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    recognizer = ChatTextRecognizer()

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