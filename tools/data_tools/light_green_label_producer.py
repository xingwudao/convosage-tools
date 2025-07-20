from PIL import Image, ImageDraw, ImageFont
import random
import os

# 配置参数
OUTPUT_IMAGE_DIR = "dataset/light_green_images"
OUTPUT_LABEL_DIR = "dataset/light_green_labels"
# 修改长宽比，图像更长一些（宽450，高800）
IMAGE_SIZE = (450, 800)  # 符合YOLO训练尺寸，并更长一些
CLASSES = {"white_bubble": 0, "others": 1,"green_bubble": 2}  # 添加绿色气泡类别
# 生成转账卡片的概率
TRANSFER_PROB = 0.05

# 确保输出目录存在
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def generate_realistic_chat(image_index):
    """生成一张逼真微信聊天截图并返回标注数据"""
    # 创建背景
    bg_images_dir = "dataset/light_background"
    bg_images = [os.path.join(bg_images_dir, f) for f in os.listdir(bg_images_dir) if f.endswith(('.jpg', '.png'))]
    if not bg_images:
        raise FileNotFoundError("No background images found in the 'light_background' folder.")
    bg_image_path = random.choice(bg_images)
    img = Image.open(bg_image_path).resize(IMAGE_SIZE)
    draw = ImageDraw.Draw(img)

    # 初始化标注列表
    annotations = []

    # 添加状态栏（顶部）
    status_bar_height = 40
    draw.rectangle([0, 0, IMAGE_SIZE[0], status_bar_height], fill=(255, 255, 255))
    draw.text((200, 10), "微信", fill=(0, 0, 0), font=ImageFont.truetype("simhei.ttf", 18))
    draw.text((50, 10), "10:20", fill=(0, 0, 0), font=ImageFont.truetype("simhei.ttf", 18))
    draw.text((IMAGE_SIZE[0] - 100, 10), "微信", fill=(0, 0, 0), font=ImageFont.truetype("simhei.ttf", 18))

    # 随机生成3-10条消息
    last_y = status_bar_height + 40  # 增加顶部间距
    
    # 随机生成4-10条消息
    last_y = status_bar_height + 40  # 增加顶部间距
    for _ in range(random.randint(4, 10)):
        is_sender = random.choice([True, False])  # 随机决定是对方还是自己
        
        # ===== 1. 生成头像 =====
        avatar_size = random.randint(40, 50)
        avatar_x = 20 if not is_sender else IMAGE_SIZE[0] - 20 - avatar_size
        avatar_y = last_y
        
        # 从指定目录随机挑选头像图片
        avatar_dir = "E:/Desktop/personal/recognizer_project/dataset/avatar"
        avatar_images = [os.path.join(avatar_dir, f) for f in os.listdir(avatar_dir) if f.endswith(('.jpg', '.png'))]
        if not avatar_images:
            raise FileNotFoundError("No avatar images found in the 'avatar' folder.")
        avatar_path = random.choice(avatar_images)
        avatar_img = Image.open(avatar_path).resize((avatar_size, avatar_size))
        img.paste(avatar_img, (avatar_x, avatar_y))
        
        # ===== 1.5. 随机插入转账卡片 =====
        if random.random() < TRANSFER_PROB:
            # 转账卡片尺寸及位置
            card_w, card_h = random.randint(180, 220), random.randint(70, 90)
            card_x = avatar_x + avatar_size + 10 if not is_sender else IMAGE_SIZE[0] - avatar_size - 20 - card_w
            card_y = avatar_y
            # 防止卡片越界
            card_x = max(0, min(card_x, IMAGE_SIZE[0] - card_w))
            card_y = max(status_bar_height + 10, min(card_y, IMAGE_SIZE[1] - card_h - 10))
            # 绘制卡片背景与边框
            card_color = (255, 165, 0)  # 橙色卡片
            border_color = (255, 165, 0)  # 橙色边框
            draw.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], fill=card_color, outline=border_color, width=2)
            # 绘制卡片文字
            amount = random.randint(1, 500)
            card_text = f"微信转账 {amount}"
            # 转账卡片文字字体
            font_size_card = random.randint(16, 20)
            font_t = ImageFont.truetype("simhei.ttf", font_size_card)
            txt_bbox = draw.textbbox((0, 0), card_text, font=font_t)
            txt_w = txt_bbox[2] - txt_bbox[0]
            txt_h = txt_bbox[3] - txt_bbox[1]
            txt_x = card_x + (card_w - txt_w) / 2
            txt_y = card_y + (card_h - txt_h) / 2
            draw.text((txt_x, txt_y), card_text, fill=(0, 0, 0), font=font_t)  # 字体颜色改为黑色
            # 标注转账卡片
            # 计算并归一化卡片标注坐标
            cx = (card_x + card_w / 2) / IMAGE_SIZE[0]
            cy = (card_y + card_h / 2) / IMAGE_SIZE[1]
            cw = card_w / IMAGE_SIZE[0]
            ch = card_h / IMAGE_SIZE[1]
            # 限制在 [0,1] 范围内
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            cw = min(max(cw, 0.0), 1.0)
            ch = min(max(ch, 0.0), 1.0)
            annotations.append(f"{CLASSES['others']} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}")
            # 更新下一条消息位置并跳过气泡
            last_y = card_y + card_h + random.randint(40, 60)  # 增加间距
            continue
        
        # ===== 2. 生成消息气泡 =====
        # 随机消息内容
        messages = ["你好！", "对对对好玩好玩太棒了好有",
                    "在吗？","今天天气怎么样", "什么时候方便？", 
                    "收到请回复", "资料已发", "谢谢！",
                    "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈",
                    "我喜欢晴天", "我喜欢下雨天", "我喜欢阴天",
                    "stay in the middle,like you a little",
                    "don't wanna riddle,i need u say it back",
                    "oh say it ditto",
                    "椰子水好喝，是的，非常好喝。我也觉得。你是有品位的。",
                    "不喜欢潮湿闷热的天气，舒适感很低，是的，我也觉得很闷热，但是我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我喜欢这种感觉，我", "我喜欢晴天",
                    "嗯，好的","我觉得你说得对",
                    "我也这么认为","我同意你的观点",
                    "于浩歌狂热之际中寒，于天上看见深渊，于一切眼中看见无所有，于无所希望中得救。","我很好，谢谢！",
                    "嗯","我也觉得是这样","我也这么认为",
                    "只要做了就是做了但是做了不一定做对了。做了不一定有用，也有可能现在没用，也许有一天会有用，这很难说。没关系。总比无所事事好。无所事事有时浪费时间了，但如果能休息，那也不是浪费时间。",
                    "How are you doing","I like it when you smile",
                    "I'm fine, thanks","I'm fine, thanks","我很好, 谢谢",
                    "I think I've seen this film before and i didn't like the ending.",
                    "当这世界已经准备将我遗弃，像一个伤兵被留在孤独荒野里。开始怀疑我存在有没有意义。在别人眼里我似乎变成了隐形。难道失败就永远翻不了身，谁来挽救坠落的灵魂。每次一见到你，心里好平静，好像一只蝴蝶，飞过废墟，我又能活下去，我又找回勇气，你的爱像氧气，帮忙我呼吸。每次一见到你都心存感激，你看着我的眼，没有怀疑。我能撑得下去，我会忘了过去，是你让我找回新的生命。爱我这样的人对你来说不容易，我的痛苦你也经历，你是唯一，陪我到天堂和地狱。"
                    "I'm fine, thanks",
                    "你说得对但是每过去一个小时，就意味着会有60分钟过去，每过去1分钟，就意味着会有60秒过去.yes，in my opinion, I think so.As far as I'm concerned, I think so.这是一个dark green label producer的测试消息。我要让它足够长，以便训练模型提供多样化的输入。1+1=2，i know but i can't prove it right now.",
                    "我也不知道呀","太好啦", 
                    "你们的眼睛里藏着星星般的光芒，每一次勇敢面对生活的模样都特别闪亮。别害怕孤单，老师的关怀、同学的陪伴始终围绕身旁；别怀疑自己，你们早已在风雨中学会了坚强。那些独自成长的时光，都会成为生命里最独特的勋章。记得，世界很大，总有人为你们的每一次进步欣喜鼓掌，也请你们永远相信：心有阳光，何惧路长？你们只管带着勇气奔跑，未来定有繁花绽放！",
                    "- 淘晶驰T1一般用USART1或USART2，波特率常用115200或9600，需查T1屏说明书。- 直接参考本工程的`usart_init`和`USART_Loop`，将数据格式化为T1屏支持的指令格式（如`n0.val=123\xFF\xFF\xFF`）。- 通过串口发送指令，如`USART_SendData(USART1, 0x71);`。- 通过串口接收数据，如`while(USART_GetFlagStatus(USART1, USART_FLAG_RXNE) == RESET);`。- 通过串口中断接收数据，如`void USART1_IRQHandler(void) { if(USART_GetITStatus(USART1, USART_IT_RXNE) != RESET) { USART_ClearITPendingBit(USART1, USART_IT_RXNE); } }`。- 通过串口DMA接收数据，如`void DMA1_Channel5_IRQHandler(void) { if(DMA_GetITStatus(DMA1_IT_TC5) != RESET) { DMA_ClearITPendingBit(DMA1_IT_TC5); } }`。- 通过串口DMA发送数据，如`void DMA1_Channel4_IRQHandler(void) { if(DMA_GetITStatus(DMA1_IT_TC4) != RESET) { DMA_ClearITPendingBit(DMA1_IT_TC4); } }`。- 通过串口DMA接收数据，如`void DMA1_Channel5_IRQHandler(void) { if(DMA_GetITStatus(DMA1_IT_TC5) != RESET) { DMA_ClearITPendingBit(DMA1_IT_TC5); } }`。- 建议将所有显示相关的printf封装成`Display.c/h`，便于维护。"]
        message = random.choice(messages)
        
        # 字体设置
        font_size = random.randint(18, 24)  # 字号增大
        font = ImageFont.truetype("simhei.ttf", font_size)
        
        # 计算文字尺寸
        text_bbox = draw.textbbox((0, 0), message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 气泡位置和尺寸
        bubble_padding = 20  # 增大内边距，实现气泡等比例放大
        bubble_width = text_width + bubble_padding * 2
        bubble_height = text_height + bubble_padding * 2
        bubble_x = avatar_x + avatar_size + 15 if not is_sender else IMAGE_SIZE[0] - avatar_size - 30 - bubble_width
        bubble_y = avatar_y  # 移除-5的偏移，避免可能的重叠
        
        # 防止气泡越界
        bubble_x = max(0, min(bubble_x, IMAGE_SIZE[0] - bubble_width))
        bubble_y = max(status_bar_height + 10, min(bubble_y, IMAGE_SIZE[1] - bubble_height - 10))
        
        # 绘制气泡（圆角矩形）
        # WeChat样式：对方泡白色，自己泡绿色
        bubble_color = (255, 255, 255) if not is_sender else (156, 231, 97)  # 标准微信绿色
        draw.rounded_rectangle(
            [bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height],
            radius=12, fill=bubble_color
        )
        
        # 绘制气泡尾巴以增强真实感
        tail_size = 6
        if not is_sender:
            tail = [
                (bubble_x, bubble_y + 16),
                (bubble_x - tail_size, bubble_y + 22),
                (bubble_x, bubble_y + 28)
            ]
        else:
            tail = [
                (bubble_x + bubble_width, bubble_y + 16),
                (bubble_x + bubble_width + tail_size, bubble_y + 22),
                (bubble_x + bubble_width, bubble_y + 28)
            ]
        draw.polygon(tail, fill=bubble_color)
        
        # 绘制文字
        text_x = bubble_x + bubble_padding
        text_y = bubble_y + bubble_padding
        draw.text((text_x, text_y), message, fill=(0, 0, 0), font=font)
        
        # 只对绿色气泡（用户消息）进行标注
        if  is_sender:  # 用户消息（绿色气泡）才标注
            bubble_center_x = (bubble_x + bubble_width / 2) / IMAGE_SIZE[0]
            bubble_center_y = (bubble_y + bubble_height / 2) / IMAGE_SIZE[1]
            bubble_width_norm = bubble_width / IMAGE_SIZE[0]
            bubble_height_norm = bubble_height / IMAGE_SIZE[1]
            # 限制坐标在[0,1]范围内
            bubble_center_x = min(max(bubble_center_x, 0.0), 1.0)
            bubble_center_y = min(max(bubble_center_y, 0.0), 1.0)
            bubble_width_norm = min(max(bubble_width_norm, 0.0), 1.0)
            bubble_height_norm = min(max(bubble_height_norm, 0.0), 1.0)
            annotations.append(f"{CLASSES['green_bubble']} {bubble_center_x:.6f} {bubble_center_y:.6f} {bubble_width_norm:.6f} {bubble_height_norm:.6f}")
        
        # 更新下一条消息位置
        last_y = max(avatar_y + avatar_size, bubble_y + bubble_height) + random.randint(40, 60)

        # 确保不会超出图片底部
        if last_y > IMAGE_SIZE[1] - 100:  # 预留底部空间
            break    
    # 保存图片
    img_path = os.path.join(OUTPUT_IMAGE_DIR, f"wechat_image_{image_index:03d}.jpg")
    img.save(img_path, format="JPEG")
    
    # 保存标注
    label_path = os.path.join(OUTPUT_LABEL_DIR, f"wechat_image_{image_index:03d}.txt")
    with open(label_path, 'w') as f:
        # 对所有标注进行归一化边界检查，确保在 [0,1]
        normalized_anns = []
        for ann in annotations:
            parts = ann.split()
            cls = parts[0]
            coords = list(map(float, parts[1:]))
            coords = [min(max(c, 0.0), 1.0) for c in coords]
            normalized_anns.append(f"{cls} " + " ".join(f"{c:.6f}" for c in coords))
        f.write("\n".join(normalized_anns))
    
    return img_path, label_path

# 生成1000张带标注的合成数据
for i in range(1000):
    img_path, label_path = generate_realistic_chat(i)
    if (i + 1) % 100 == 0:
        print(f"已生成 {i + 1}/1000 张图片和标注")

print(f"生成的图片已保存到文件夹 {OUTPUT_IMAGE_DIR}")
print(f"标注已保存到文件夹 {OUTPUT_LABEL_DIR}")