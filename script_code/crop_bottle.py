from ultralytics import YOLO
import cv2
import os

# ===========================
# 参数配置
# ===========================

# YOLO模型
MODEL_PATH = "yolov8n.pt"

# 原始图片目录
IMAGE_DIR = "no_cap_image"

# 输出目录
OUTPUT_DIR = "no_cap_output"

# Bottle类别（COCO）
BOTTLE_CLASS = 39

# 置信度
CONF = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# 加载模型
# ===========================

model = YOLO(MODEL_PATH)

# ===========================
# 遍历图片
# ===========================

image_list = os.listdir(IMAGE_DIR)

for image_name in image_list:

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    # YOLO预测
    results = model.predict(
        image,
        conf=CONF,
        verbose=False
    )

    bottle_index = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls.item())

            # 只保留Bottle
            if cls != BOTTLE_CLASS:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # 防止越界
            h, w = image.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 裁剪整个瓶子ROI
            roi = image[y1:y2, x1:x2]

            save_name = (
                os.path.splitext(image_name)[0]
                + f"_bottle_{bottle_index}.jpg"
            )

            save_path = os.path.join(
                OUTPUT_DIR,
                save_name
            )

            cv2.imwrite(
                save_path,
                roi
            )

            bottle_index += 1

    print(f"{image_name} 完成")