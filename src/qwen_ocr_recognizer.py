import os
import cv2
import numpy as np
import json
import base64
import re
from typing import Dict, Union
from dashscope import MultiModalConversation

class QwenChatRecognizer:
    """通义千问聊天截图识别器
    
    用于识别微信聊天截图中的对话内容，包括说话人、消息内容、类型和时间戳。
    支持多种输入方式：文件路径、字节流、Base64编码。
    输出标准化的 JSON 格式，包含 lines 列表，每条消息有固定的四个字段。
    
    示例:
        >>> recognizer = QwenChatRecognizer()
        >>> result = recognizer.recognize_with_speaker_from_path("chat.jpg")
        >>> print(result)
        {
            "lines": [
                {
                    "speaker": "我",
                    "content": "你好",
                    "type": "text", 
                    "timestamp": "10:30"
                }
            ]
        }
    """
    def __init__(self, api_key=None, debug=False):  # 默认关闭调试模式
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_key.txt')
            try:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        self.api_key = f.read().strip()
            except Exception as e:
                print(f"读取配置文件失败：{str(e)}")
        
        if not self.api_key:
            raise ValueError("""DASHSCOPE_API_KEY 未设置！请使用以下方式之一设置：
1. 设置环境变量 DASHSCOPE_API_KEY
2. 在当前目录创建 api_key.txt 文件并写入 API 密钥
3. 实例化时直接传入 API 密钥""")
        
        self.debug = debug
        self.prompt = """请仔细分析这张微信聊天截图，严格按照以下规则识别和输出所有聊天消息：

关键规则：
1. speaker 字段只能是"我"或"TA"：
   - 绿色气泡 = "我"
   - 白色气泡 = "TA"
2. type 字段必须是以下之一：
   - text: 纯文本消息
   - image: 图片消息
   - call：语音通话/视频通话，对应的content字段为通话时长
   - voice: 语音消息
   - video: 视频消息
   - file: 文件消息
   - transfer: 转账消息(通常为橙色卡片)
   - quote: 引用消息（通常在聊天中显示为引用，在气泡下方）
   - system: 系统消息（如通知、提示等,一般居中且为灰色）
3. content 字段保存消息的具体内容
4. timestamp 字段保存时间戳，如果没有则为空字符串

按照从上到下的顺序，将消息整理为以下固定的 JSON 格式（内容仅作示范）（不要添加任何说明文字或注释）：

{
    "lines": [
        {
            "speaker": "我"或"TA",// 绿色气泡是"我"，白色气泡是"TA"
            "content": "消息内容",
            "type": "text",
            "timestamp": "13:45"
        }
    ]
}"""    
    def _clean_json_string(self, text):
        """清理和标准化 JSON 文本"""
        # 移除可能的 markdown 代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # 移除注释
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        
        # 移除文本开头和结尾的非法字符
        text = text.strip()
        
        # 如果文本以多个 } 结尾，只保留第一个
        text = re.sub(r'}[\s}]*$', '}', text)
        
        # 修复常见的 JSON 格式错误
        replacements = {
            '，': ',',
            '：': ':',
            '【': '[',
            '】': ']',
            '"': '"',
            '"': '"'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 打印清理后的文本
        print(text)
        
        return text    
    def _format_response(self, result_text):
        """格式化并规范化 API 返回结果"""
        try:
            # 清理和标准化 JSON 文本
            cleaned_text = self._clean_json_string(result_text)
            
            # 尝试解析 JSON
            try:
                result = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # 如果解析失败，尝试提取 JSON 部分
                json_match = re.search(r'{[\s\S]*}', cleaned_text)
                if json_match:
                    cleaned_text = json_match.group(0)
                    try:
                        result = json.loads(cleaned_text)
                    except:
                        raise ValueError("无法解析返回的 JSON")
                else:
                    raise ValueError("未找到有效的 JSON")

            # 初始化标准格式的结果
            standardized_result = {
                "lines": []
            }
            
            # 处理两种可能的输入格式：lines 或 messages
            source_list = result.get("lines", result.get("messages", []))
            if not isinstance(source_list, list):
                source_list = []
            
            # 标准化每个消息
            for item in source_list:
                if not isinstance(item, dict):
                    continue
                
                # 检查并规范化 speaker 字段
                speaker = str(item.get("speaker", "TA")).strip()
                if speaker not in ["我", "TA"]:
                    speaker = "TA"
                
                # 规范化消息类型
                msg_type = self._normalize_type(str(item.get("type", "text")))
                
                # 构造标准化的消息
                standardized_message = {
                    "speaker": speaker,
                    "content": str(item.get("content", "")).strip(),
                    "type": msg_type,
                    "timestamp": str(item.get("timestamp", "")).strip()
                }
                
                standardized_result["lines"].append(standardized_message)
            
            return json.dumps(standardized_result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            if self.debug:
                print(f"\n格式化响应时出错：{str(e)}")
                print(f"原始响应：{result_text}")
            return json.dumps({"error": f"处理模型返回结果时出错：{str(e)}"}, ensure_ascii=False)

    def recognize_with_speaker_from_path(self, image_path):
        if self.debug:
            print(f"\n处理图片：{image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            return json.dumps({"error": "图片读取失败"}, ensure_ascii=False)
        
        img = self._check_image(img)
        if img is None:
            return json.dumps({"error": "图片预处理失败"}, ensure_ascii=False)
        
        # 保存处理后的图片到临时文件
        temp_path = "temp_processed.jpg"
        cv2.imwrite(temp_path, img)
        
        try:
            if self.debug:
                print(f"\n开始识别处理后的图片：{temp_path}")
            
            result = self._recognize_with_speaker_core(temp_path)
            os.remove(temp_path)
            return result
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            error_msg = f"识别过程出错：{str(e)}"
            if self.debug:
                print(f"\n{error_msg}")
            return json.dumps({"error": error_msg}, ensure_ascii=False)

    def recognize_with_speaker_from_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = self._check_image(img)
        if img is None:
            return json.dumps({"error": "图片读取失败"}, ensure_ascii=False)
        
        # 保存处理后的图片到临时文件
        temp_path = "temp_processed.jpg"
        cv2.imwrite(temp_path, img)
        
        try:
            result = self._recognize_with_speaker_core(temp_path)
            os.remove(temp_path)  # 删除临时文件
            return result
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def recognize_with_speaker_from_base64(self, image_base64):
        image_bytes = base64.b64decode(image_base64)
        return self.recognize_with_speaker_from_bytes(image_bytes)

    def _recognize_with_speaker_core(self, image_path):
        abs_path = os.path.abspath(image_path)
        image_uri = f"file://{abs_path}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": image_uri,
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 28 * 28 * 8192,
                        "enable_rotate": True,
                    },
                    {
                        "text": self.prompt
                    },
                ],
            }
        ]

        try:
            if self.debug:
                print("\n正在调用模型...")
                print("\nAPI Key长度:", len(self.api_key) if self.api_key else 0)
                print("API Key前10个字符:", self.api_key[:10] if self.api_key else "无")
            print("\n正在调用 OCR 模型...")
            
            try:
                response = MultiModalConversation.call(
                    api_key=self.api_key,
                    model="qwen-vl-ocr-latest",  # 使用 OCR 模型
                    messages=messages,
                )
                print("API 调用完成")
            except Exception as api_error:
                print(f"API 调用异常: {str(api_error)}")
                raise
            
            if self.debug:
                print("\n模型返回原始响应:")
                print("-" * 50)
                print(json.dumps(response, ensure_ascii=False, indent=2))
                print("-" * 50)

            # 验证响应格式
            if not response:
                raise ValueError("模型返回为空")
                
            if not isinstance(response, dict):
                raise ValueError(f"模型返回格式错误，期望 dict 类型，得到 {type(response)}")
                
            if "output" not in response:
                raise ValueError("模型返回中没有 'output' 字段")
                
            if "choices" not in response["output"]:
                raise ValueError("模型返回中没有 'choices' 字段")
                
            if not response["output"]["choices"]:
                raise ValueError("模型返回的 'choices' 为空")
                
            if "message" not in response["output"]["choices"][0]:
                raise ValueError("模型返回中没有 'message' 字段")
                
            if "content" not in response["output"]["choices"][0]["message"]:
                raise ValueError("模型返回中没有 'content' 字段")
                
            if not response["output"]["choices"][0]["message"]["content"]:
                raise ValueError("模型返回的 'content' 为空")

            # 提取文本内容
            content = response["output"]["choices"][0]["message"]["content"]
            
            # 根据返回类型处理内容
            if isinstance(content, list):
                # 如果是列表，找到包含文本的项
                text_item = next((item for item in content if isinstance(item, dict) and "text" in item), None)
                if text_item is None:
                    raise ValueError("在返回结果中没有找到文本内容")
                result_text = text_item["text"]
            elif isinstance(content, dict):
                # 如果是字典，直接获取文本
                if "text" not in content:
                    raise ValueError("在返回结果中没有找到文本内容")
                result_text = content["text"]
            elif isinstance(content, str):
                # 如果直接是字符串
                result_text = content
            else:
                raise ValueError(f"未知的返回内容类型：{type(content)}")

            if self.debug:
                print("\n提取的文本内容:")
                print("-" * 50)
                print(result_text)
                print("-" * 50)

            return self._format_response(result_text)
            
        except Exception as e:
            error_msg = f"调用模型时发生错误：{str(e)}"
            if self.debug:
                print(f"\n{error_msg}")
                print("错误详情：")
                import traceback
                traceback.print_exc()
            return json.dumps({"error": error_msg}, ensure_ascii=False)

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

    def _normalize_type(self, msg_type: str) -> str:
        """标准化消息类型，只返回允许的类型值"""
        # 定义允许的类型和它们的别名映射
        type_mapping = {
            # 文本消息
            'text': 'text',
            'txt': 'text',
            '文本': 'text',
            '对话': 'text',
            'message': 'text',
            'quote': 'text',  # 引用也算文本消息
            
            # 图片消息
            'image': 'image',
            'img': 'image',
            'picture': 'image',
            'photo': 'image',
            '图片': 'image',
            '图像': 'image',
            
            # 语音消息
            'voice': 'voice',
            'audio': 'voice',
            '语音': 'voice',
            '录音': 'voice',
            
            # 视频消息
            'video': 'video',
            '视频': 'video',
            
            # 文件消息
            'file': 'file',
            'doc': 'file',
            'document': 'file',
            '文件': 'file',
            
            # 转账消息
            'transfer': 'transfer',
            'payment': 'transfer',
            'money': 'transfer',
            '转账': 'transfer',
            '红包': 'transfer',
            '支付': 'transfer',

            #通话记录
            '语音通话': 'call', 
            '视频通话': 'call',  

            # 系统消息
            'system': 'system',
        }
        
        # 标准化处理：转小写，去除空格
        msg_type = msg_type.lower().strip()
        
        # 返回映射的类型，如果没有匹配就返回 text
        return type_mapping.get(msg_type, 'text')

if __name__ == "__main__":
    try:
        # 创建识别器实例，默认关闭调试模式
        recognizer = QwenChatRecognizer(debug=False)
        
        print("\n微信聊天截图识别工具")
        print("=" * 40)
        print("支持的输入方式：")
        print("1. 图片文件路径")
        print("2. 图片字节流")
        print("3. Base64编码图片")
        print("-" * 40)
        
        choice = input("请选择输入方式 (1-3): ").strip()

        if not choice or choice not in ["1", "2", "3"]:
            print("\n错误：无效的选择！")
            exit(1)
            
        image_path = input("\n请输入图片路径: ").strip()
        if not image_path:
            print("\n错误：图片路径不能为空！")
            exit(1)
            
        if not os.path.exists(image_path):
            print("\n错误：图片文件不存在！")
            exit(1)

        print("\n正在处理图片...")
        try:
            if choice == "1":
                result = recognizer.recognize_with_speaker_from_path(image_path)
            elif choice == "2":
                with open(image_path, "rb") as f:
                    result = recognizer.recognize_with_speaker_from_bytes(f.read())
            else:  # choice == "3"
                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode()
                result = recognizer.recognize_with_speaker_from_base64(image_base64)
                
            
            print("\n识别结果：")
            print("-" * 40)
            print(json.dumps(json.loads(result), ensure_ascii=False, indent=2))
            print("-" * 40)
            
        except Exception as e:
            print(f"\n处理失败：{str(e)}")
            exit(1)
    
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        exit(0)
    except Exception as e:
        print(f"\n程序异常：{str(e)}")
        exit(1)

