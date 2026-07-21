from ultralytics import YOLO
import cv2
import os
import sys

# ===========================
# 参数配置
# ===========================

# YOLO模型
MODEL_PATH = "../yolo26x.pt"

# 目标类别（COCO）
BOTTLE_CLASS = 39

# 置信度
CONF = 0.001

# 输入图片路径（直接修改这里，然后运行即可）
INPUT_IMAGE = "D:\\YOLO_Project\\YOLO-Resnet-test\\dataset\\2-cap-image\\image132.jpg"

# 输出目录
OUTPUT_DIR = "D:\\YOLO_Project\\YOLO-Resnet-test\\dataset\\output\\2-cap-output"

# ===========================
# 加载模型
# ===========================

model = YOLO(MODEL_PATH)


def process_single_image(image_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.basename(image_path)

    image = cv2.imread(image_path)

    if image is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return False

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
    # 打印本图所有检测框
    # ===========================
    print(f"[检测结果] {image_name}")
    for b in all_boxes:
        print(f"  类别 {b['cls']:2d} | 置信度 {b['conf']:.3f}")

    # 分离Bottle框和其他类别的框
    bottle_boxes = [b for b in all_boxes if b["cls"] == BOTTLE_CLASS]
    other_boxes = [b for b in all_boxes if b["cls"] != BOTTLE_CLASS]

    is_special = False
    crop_index = 0

    if bottle_boxes:
        # ===========================
        # 有Bottle -> 取置信度最高的Bottle框
        # ===========================
        best_box = max(bottle_boxes, key=lambda b: b["conf"])
        boxes_to_crop = [best_box]
        name_tag = "bottle"
        print(f"  → 选择Bottle框(conf={best_box['conf']:.3f})")
    elif other_boxes:
        # ===========================
        # 无Bottle -> 取其他类别最高置信度的框
        # ===========================
        best_box = max(other_boxes, key=lambda b: b["conf"])
        boxes_to_crop = [best_box]
        name_tag = "other"
        is_special = True
        print(f"  → [特殊]无Bottle，使用类别{best_box['cls']}(conf={best_box['conf']:.3f})")
    else:
        print(f"  → 未检测到任何目标")
        return False

    h, w = image.shape[:2]

    for b in boxes_to_crop:
        x1 = max(0, b["x1"])
        y1 = max(0, b["y1"])
        x2 = min(w, b["x2"])
        y2 = min(h, b["y2"])

        roi = image[y1:y2, x1:x2]

        save_name = f"2-cap-{os.path.splitext(image_name)[0]}_{name_tag}_{crop_index}.jpg"

        save_path = os.path.join(
            output_dir,
            save_name
        )

        cv2.imwrite(save_path, roi)

        crop_index += 1

    tag = " [特殊]" if is_special else ""
    print(f"  ✓ 已保存为 {save_name}{tag}")
    return True


# ===========================
# 主入口
# ===========================

if __name__ == "__main__":
    # 优先使用命令行参数，否则使用上方 INPUT_IMAGE
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_IMAGE
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR

    print(f"输入: {input_path}")
    print(f"输出: {output_dir}")
    process_single_image(input_path, output_dir)
