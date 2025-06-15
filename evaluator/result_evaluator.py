import os
import json
import re
import logging
from typing import Dict, Optional, Union
from dashscope import MultiModalConversation

class ResultEvaluator:
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        """初始化评估器"""
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_key.txt')
            try:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        self.api_key = f.read().strip()
            except Exception as e:
                pass

        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置")

        self.debug = debug
        self.evaluation_prompt = """请评估以下聊天记录识别结果的质量：

原始图片内容：{original_image_context}

OCR识别结果：{recognition_result}

请直接返回以下格式的JSON（不要包含任何其他说明文字）：
{{
    "accuracy_score": 0.95,
    "completeness_score": 0.90,
    "structure_score": 0.85,
    "sequence_score": 1.0,
    "overall_score": 0.92,
    "error_analysis": {{
        "missing_messages": [],
        "wrong_speakers": [],
        "wrong_types": [],
        "text_errors": []    }},
    "suggestions": []
}}"""

    def evaluate_recognition(self, image_path, recognition_result):
        """评估识别结果"""
        try:
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
                            "text": self.evaluation_prompt.format(
                                original_image_context="[图片内容]",
                                recognition_result=recognition_result
                            )
                        },
                    ],
                }
            ]

            response = MultiModalConversation.call(
                api_key=self.api_key,
                model="qwen2.5-vl-7b-instruct",
                messages=messages,
            )

            if not response or "output" not in response:
                return json.dumps({"error": "API 返回为空"}, ensure_ascii=False)

            evaluation_text = response["output"]["choices"][0]["message"]["content"]
            if isinstance(evaluation_text, list):
                evaluation_text = next(item["text"] for item in evaluation_text if "text" in item)

            # 清理和规范化 JSON 文本
            evaluation_text = evaluation_text.strip()
            # 移除可能的 markdown 标记
            evaluation_text = re.sub(r'```json\s*', '', evaluation_text)
            evaluation_text = re.sub(r'```\s*', '', evaluation_text)

            try:
                result = json.loads(evaluation_text)
                # 确保所有必需的字段都存在
                required_fields = [
                    "accuracy_score", "completeness_score", 
                    "structure_score", "sequence_score", 
                    "overall_score"
                ]
                for field in required_fields:
                    if field not in result:
                        result[field] = 0.0
                    else:
                        # 确保分数是浮点数
                        result[field] = float(result[field])
                
                if "error_analysis" not in result:
                    result["error_analysis"] = {
                        "missing_messages": [],
                        "wrong_speakers": [],
                        "wrong_types": [],
                        "text_errors": []
                    }
                
                if "suggestions" not in result:
                    result["suggestions"] = []
                
                return json.dumps(result, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return json.dumps({
                    "error": "解析评估结果失败",
                    "raw_text": evaluation_text
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "error": f"处理评估结果时出错: {str(e)}",
                    "raw_text": evaluation_text
                }, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    from evaluator.test_evaluator import evaluate_chat_recognition, setup_logging
    
    setup_logging()
    try:
        project_root = os.path.dirname(current_dir)
        image_folder = os.path.join(project_root, "chat_screenshots")
        evaluate_chat_recognition(image_folder)
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
