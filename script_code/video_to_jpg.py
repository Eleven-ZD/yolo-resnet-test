import cv2
import os

# 视频文件路径（使用原始字符串避免转义问题）
video_path = r"./no_cap.mp4"
# 输出文件夹路径
output_dir = r"./no_cap_image"

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("错误：无法打开视频文件")
    exit()

# 获取帧率（验证用）
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"视频帧率: {fps} fps")

frame_count = 0          # 当前帧序号（从0开始）
save_counter = 1         # 保存图片的序号（从1开始）

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕

    # 每隔2帧保存1帧（即帧序号为0,2,4,...）
    if frame_count % 2 == 0:
        # 构造文件名，例如 Image1.jpg
        filename = f"Image{save_counter}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, frame)
        print(f"已保存: {filename}")
        save_counter += 1

    frame_count += 1

# 释放资源
cap.release()
cv2.destroyAllWindows()

print(f"共保存了 {save_counter - 1} 张图片。")