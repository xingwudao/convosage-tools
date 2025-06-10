import os
import base64
from chat_text_recognizer import ChatTextRecognizer

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