from ultralytics import YOLO
import cv2
import os

# ===========================
# 参数配置
# ===========================

# YOLO模型
MODEL_PATH = "../yolo26x.pt"

# 原始图片目录
# IMAGE_DIR = "D:\\YOLO_Project\\YOLO-Resnet-test\\dataset\\2-no_cap-image"
IMAGE_DIR = "D:\\YOLO_Project\\YOLO-Resnet-test\\test-image\\yes"

# 输出目录
# OUTPUT_DIR = "D:\\YOLO_Project\\YOLO-Resnet-test\\dataset\\output\\2-no_cap-output"
OUTPUT_DIR = "D:\\YOLO_Project\\YOLO-Resnet-test\\test-image\\yes-output"

# 目标类别（COCO）
BOTTLE_CLASS = 39
OTHER_CLASSES = [i for i in range(80)]  # COCO共80类，用于fallback

# 置信度
CONF = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 记录特殊（使用非Bottle裁剪）的图片
special_images = []

# 记录未裁剪的图片
uncropped_images = []

# 记录降低置信度重试后才检测到目标的图片
retry_images = []

# ===========================
# 加载模型
# ===========================

model = YOLO(MODEL_PATH)

# ===========================
# 遍历图片
# ===========================

image_list = os.listdir(IMAGE_DIR)

box_num = 0

for image_name in image_list:

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        uncropped_images.append(image_name)
        continue

    # YOLO预测
    results = model.predict(
        image,
        conf=CONF,
        verbose=False
    )

    # ===========================
    # 收集所有检测框
    # ===========================
    all_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0]
            all_boxes.append({
                "cls": cls,
                "conf": conf,
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2)
            })

    # ===========================
    # 打印本图所有检测框（固定宽度对齐）
    # ===========================
    boxes_info = "  ".join(f"类别{b['cls']:2d} conf={b['conf']:.3f}" for b in all_boxes)
    print(f"[检测结果] {image_name:<15s}{boxes_info:<20s}", end="")

    # 分离Bottle框和其他类别的框
    bottle_boxes = [b for b in all_boxes if b["cls"] == BOTTLE_CLASS]
    other_boxes = [b for b in all_boxes if b["cls"] != BOTTLE_CLASS]

    is_special = False
    retried = False
    crop_index = 0

    if bottle_boxes:
        # ===========================
        # 有Bottle -> 取置信度最高的Bottle框
        # ===========================
        best_box = max(bottle_boxes, key=lambda b: b["conf"])
        boxes_to_crop = [best_box]
        name_tag = "bottle"
        print(f"→ 选Bottle conf={best_box['conf']:.3f}  ", end="")
    elif other_boxes:
        # ===========================
        # 无Bottle -> 取其他类别最高置信度的框
        # ===========================
        best_box = max(other_boxes, key=lambda b: b["conf"])
        boxes_to_crop = [best_box]
        name_tag = "other"
        is_special = True
        special_images.append(image_name)
        print(f"→ [特殊]类别{best_box['cls']:2d} conf={best_box['conf']:.3f}  ", end="")
    else:
        # ===========================
        # 未检测到目标 -> 降低置信度重试
        # ===========================
        current_conf = CONF
        retried = False
        while len(all_boxes) == 0 and current_conf > 0.1:
            current_conf = round(current_conf - 0.1, 1)
            retried = True
            print(f"→ 降置信度至{current_conf}重试  ", end="")
            results = model.predict(image, conf=current_conf, verbose=False)
            all_boxes = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy[0]
                    all_boxes.append({
                        "cls": cls,
                        "conf": conf,
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2)
                    })

        if len(all_boxes) == 0:
            print(f"→ 仍无目标，跳过")
            uncropped_images.append(image_name)
            continue

        # 重试成功，标记并重新分类
        retry_images.append(image_name)
        boxes_info = "  ".join(f"类别{b['cls']:2d} conf={b['conf']:.3f}" for b in all_boxes)
        print(f"检出{boxes_info:<20s}", end="")
        bottle_boxes = [b for b in all_boxes if b["cls"] == BOTTLE_CLASS]
        other_boxes = [b for b in all_boxes if b["cls"] != BOTTLE_CLASS]
        if bottle_boxes:
            best_box = max(bottle_boxes, key=lambda b: b["conf"])
            boxes_to_crop = [best_box]
            name_tag = "bottle"
            print(f"→ 选Bottle conf={best_box['conf']:.3f}  ", end="")
        else:
            best_box = max(other_boxes, key=lambda b: b["conf"])
            boxes_to_crop = [best_box]
            name_tag = "other"
            is_special = True
            special_images.append(image_name)
            print(f"→ 类别{best_box['cls']:2d} conf={best_box['conf']:.3f}  ", end="")

    h, w = image.shape[:2]

    for b in boxes_to_crop:
        x1 = max(0, b["x1"])
        y1 = max(0, b["y1"])
        x2 = min(w, b["x2"])
        y2 = min(h, b["y2"])

        roi = image[y1:y2, x1:x2]

        save_name = f"test-cap-{os.path.splitext(image_name)[0]}_{name_tag}_{crop_index}.jpg"


        save_path = os.path.join(
            OUTPUT_DIR,
            save_name
        )

        cv2.imwrite(save_path, roi)

        crop_index += 1

    tag_parts = []
    if retried:
        tag_parts.append("降置信度")
    if is_special:
        tag_parts.append("特殊")
    tag = "★" + "-" + "+".join(tag_parts) if tag_parts else ""
    print(f"✓ {name_tag}_{crop_index-1}.jpg {tag}")
    box_num += crop_index

# ===========================
# 汇总打印
# ===========================
print(f"\n{'='*50}")
print(f"处理完成！")
print(f"总共裁剪 {box_num} 个目标")
print(f"共处理 {len(image_list)} 张图片")
if special_images:
    print(f"\n特殊图片（无Bottle，使用其他类别最高置信度）共 {len(special_images)} 张：")
    for img in special_images:
        print(f"  - {img}")
else:
    print(f"\n所有图片均检测到Bottle，无特殊图片")
if retry_images:
    print(f"\n降置信度重试后检出目标的图片共 {len(retry_images)} 张：")
    for img in retry_images:
        print(f"  - {img}")
if uncropped_images:
    print(f"\n未裁剪图片共 {len(uncropped_images)} 张：")
    for img in uncropped_images:
        print(f"  - {img}")
else:
    print(f"\n所有图片均成功裁剪，无遗漏")
print(f"{'='*50}")