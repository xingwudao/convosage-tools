import os
import json
import logging
import sys
from typing import Dict, List, Union

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # evaluator 目录
project_root = os.path.dirname(current_dir)  # 项目根目录
sys.path.append(project_root)

try:
    from src.qwen_ocr_recognizer import QwenChatRecognizer
    from evaluator.result_evaluator import ResultEvaluator
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"Python 路径:")
    for p in sys.path:
        print(f"  - {p}")
    print(f"当前目录: {current_dir}")
    print(f"项目根目录: {project_root}")
    raise

def setup_logging() -> None:
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )

def verify_message_format(message: Dict) -> bool:
    """验证消息格式是否符合规范"""
    # 检查必需字段
    required_fields = ["speaker", "content", "type", "timestamp"]
    if not all(field in message for field in required_fields):
        logging.warning(f"消息缺少必需字段: {message}")
        return False
        
    # 验证 speaker 字段
    if message["speaker"] not in ["我", "TA"]:
        logging.warning(f"无效的 speaker 值: {message['speaker']}")
        return False
        
    # 验证 type 字段
    valid_types = ["text", "image", "voice", "video", "file", "transfer", "quote","call", "system"]
    if message["type"] not in valid_types:
        logging.warning(f"无效的 type 值: {message['type']}")
        return False
        
    return True

def evaluate_single_image(
    image_path: str,
    recognizer: QwenChatRecognizer,
    evaluator: ResultEvaluator
) -> Dict[str, Union[str, Dict, float]]:
    """评估单张图片的识别结果"""
    try:
        import time
        # 记录开始时间
        start_time = time.time()
        
        # 获取识别结果
        recognition_result = recognizer.recognize_with_speaker_from_path(image_path)
        recognition_time = time.time() - start_time
        
        # 解析识别结果
        result_dict = json.loads(recognition_result)
        
        # 验证结果格式
        #if "lines" not in result_dict:
        #    raise ValueError("识别结果缺少 lines 字段")
            
        if not isinstance(result_dict["lines"], list):
            raise ValueError("lines 字段不是列表类型")
            
        # 验证每条消息的格式
        invalid_messages = []
        valid_messages = []
        for msg in result_dict["lines"]:
            if verify_message_format(msg):
                valid_messages.append(msg)
            else:
                invalid_messages.append(msg)
                
        if invalid_messages:
            logging.warning(f"发现 {len(invalid_messages)} 条格式无效的消息")
            
        # 更新识别结果只包含有效消息
        result_dict["lines"] = valid_messages
        recognition_result = json.dumps(result_dict, ensure_ascii=False)
        
        # 进行评估
        evaluation = evaluator.evaluate_recognition(image_path, recognition_result)
        
        return {
            "filename": os.path.basename(image_path),
            "recognition": result_dict,
            "evaluation": json.loads(evaluation),
            "recognition_time": round(recognition_time, 3),
            "valid_message_count": len(valid_messages),
            "invalid_message_count": len(invalid_messages)
        }
        
    except Exception as e:
        logging.error(f"处理 {os.path.basename(image_path)} 失败: {str(e)}")
        return {
            "filename": os.path.basename(image_path),
            "error": str(e)
        }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """计算平均分数和时间统计"""
    stats = {
        "accuracy": 0.0,
        "completeness": 0.0,
        "structure": 0.0,
        "sequence": 0.0,
        "overall": 0.0,
        "avg_recognition_time": 0.0,  # 平均识别时间
        "total_recognition_time": 0.0,  # 总识别时间
        "min_recognition_time": float('inf'),  # 最短识别时间
        "max_recognition_time": 0.0  # 最长识别时间
    }
    
    successful_count = 0
    total_time = 0.0

    for result in results:
        if "error" not in result and "evaluation" in result:
            try:
                # 处理识别时间统计
                recog_time = float(result.get("recognition_time", 0))
                total_time += recog_time
                stats["min_recognition_time"] = min(stats["min_recognition_time"], recog_time)
                stats["max_recognition_time"] = max(stats["max_recognition_time"], recog_time)

                # 处理评估分数
                eval_data = result["evaluation"]
                stats["accuracy"] += float(eval_data.get("accuracy_score", 0))
                stats["completeness"] += float(eval_data.get("completeness_score", 0))
                stats["structure"] += float(eval_data.get("structure_score", 0))
                stats["sequence"] += float(eval_data.get("sequence_score", 0))
                stats["overall"] += float(eval_data.get("overall_score", 0))
                
                successful_count += 1
            except Exception as e:
                logging.warning(f"处理结果统计时出错: {str(e)}")
                continue

    # 计算最终统计结果
    if successful_count > 0:
        # 计算平均值
        for key in ["accuracy", "completeness", "structure", "sequence", "overall"]:
            stats[key] = round(stats[key] / successful_count, 3)
        
        # 处理时间统计
        stats["total_recognition_time"] = round(total_time, 3)
        stats["avg_recognition_time"] = round(total_time / successful_count, 3)
        stats["min_recognition_time"] = round(stats["min_recognition_time"], 3)
        stats["max_recognition_time"] = round(stats["max_recognition_time"], 3)
    else:
        # 如果没有成功的评估，重置时间统计
        stats["min_recognition_time"] = 0.0
        stats["max_recognition_time"] = 0.0
        stats["total_recognition_time"] = 0.0
        stats["avg_recognition_time"] = 0.0

    return stats

def evaluate_chat_recognition(image_folder: str) -> None:
    """评估聊天记录识别结果"""
    try:
        recognizer = QwenChatRecognizer(debug=False)
        evaluator = ResultEvaluator(debug=False)
        
        image_files = [f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        results = []
        total = len(image_files)
        
        logging.info(f"开始处理 {total} 张图片...")
        
        for idx, image_name in enumerate(image_files, 1):
            image_path = os.path.join(image_folder, image_name)
            logging.info(f"[{idx}/{total}] 处理: {image_name}")
            
            result = evaluate_single_image(image_path, recognizer, evaluator)
            results.append(result)

        # 计算成功率
        success_count = len([r for r in results if "error" not in r])
        success_rate = success_count/total
        logging.info(f"\n处理完成! 成功率: {success_rate:.1%}")        # 计算评估统计
        try:
            avg_scores = calculate_average_scores(results)
            
            print("\n评估结果:")
            print(f"准确率: {avg_scores['accuracy']:.2%}")
            print(f"完整性: {avg_scores['completeness']:.2%}")
            print(f"结构性: {avg_scores['structure']:.2%}")
            print(f"时序性: {avg_scores['sequence']:.2%}")
            print(f"总分: {avg_scores['overall']:.2%}")
            print("\n时间统计:")
            print(f"平均识别时间: {avg_scores['avg_recognition_time']:.3f}秒")
            print(f"总识别时间: {avg_scores['total_recognition_time']:.3f}秒")
            print(f"最短识别时间: {avg_scores['min_recognition_time']:.3f}秒")
            print(f"最长识别时间: {avg_scores['max_recognition_time']:.3f}秒")

            # 保存详细结果
            with open('evaluation_results.json', 'w', encoding='utf-8') as f:
                json.dump({
                    "success_rate": success_rate,
                    "average_scores": avg_scores,
                    "detailed_results": results
                }, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logging.error(f"计算评估统计时出错: {str(e)}")
            raise
        print(f"最短识别时间: {avg_scores['min_recognition_time']:.3f}秒")
        print(f"最长识别时间: {avg_scores['max_recognition_time']:.3f}秒")# 保存详细结果
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                "success_rate": success_rate,
                "average_scores": avg_scores,
                "detailed_results": results
            }, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"评估过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    try:
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_folder = os.path.join(project_root, "chat_screenshots")
        evaluate_chat_recognition(image_folder)
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
