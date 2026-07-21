# ===========================
# 实时检测 — YOLO找瓶子 + ResNet判瓶盖
# ===========================

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

from model import create_model
from dataset import val_transform

# ===========================
# 配置
# ===========================
YOLO_MODEL_PATH   = "../yolo11m.pt"
RESNET_MODEL_PATH = "../models/run_003/bottle_cap_resnet18.pth"
# VIDEO_SOURCE      = 0
VIDEO_SOURCE      = "../videos-test/video-cap.mp4"                     # 0=摄像头，或视频文件路径
# VIDEO_SOURCE      = "../videos-test/video-cap.mp4"
BOTTLE_CLASS      = 39                    # YOLO 中 bottle 的类别 ID
YOLO_CONF         = 0.3                   # YOLO 置信度阈值
RESIZE_DISPLAY    = (540, 960)            # 显示缩放（宽, 高），None 为原始大小

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# 类别名称
# ===========================
CLASS_NAMES = ["cap", "no-cap"]           # ResNet 输出：0=cap, 1=no-cap
COLORS = {
    "cap":    (0, 255, 0),                # 绿色 = 有瓶盖
    "no-cap": (0, 0, 255),                # 红色 = 无瓶盖
}

# ===========================
# 加载模型
# ===========================
print(f"设备: {DEVICE}")
print("加载 YOLO...")
yolo = YOLO(YOLO_MODEL_PATH)

print("加载 ResNet...")
resnet = create_model(num_classes=2)
resnet.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))
resnet.to(DEVICE)
resnet.eval()

# ===========================
# 辅助函数
# ===========================

def classify_roi(roi_bgr):
    """将裁剪的 BGR 图像送给 ResNet，返回 (label_str, confidence)"""
    # BGR → RGB → PIL
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = resnet(tensor)
        probs  = torch.softmax(output, dim=1)
        pred   = torch.argmax(probs, dim=1).item()
        conf   = probs[0, pred].item()

    return CLASS_NAMES[pred], conf


def draw_box(frame, x1, y1, x2, y2, label, det_conf, cls_conf):
    """在画面上绘制检测框和标签"""
    color = COLORS.get(label, (255, 255, 255))

    # ---- 框 ----
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # ---- 标签文字 ----
    text = f"{label.upper()}  det:{det_conf:.2f}  cls:{cls_conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 标签背景（留更多边距）
    pad = 8
    label_y1 = max(y1 - th - pad * 2, 0)
    label_y2 = y1
    label_x2 = x1 + tw + pad * 2

    # 背景半透明叠加
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, label_y1), (min(label_x2, frame.shape[1]), label_y2), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 文字（白色加黑色描边，更清晰）
    text_x = x1 + pad
    text_y = y1 - pad
    cv2.putText(frame, text, (text_x, text_y),
                font, font_scale, (0, 0, 0), thickness + 2)       # 黑色描边
    cv2.putText(frame, text, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness)     # 白色文字


# ===========================
# 视频循环
# ===========================
print(f"\n视频源: {VIDEO_SOURCE}")
print("按 Q 退出 | 按 S 截图保存\n")

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("无法打开视频源！")
    exit(1)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("视频结束或读取失败")
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # ---- 第一阶段：YOLO 检测瓶子 ----
    results = yolo.predict(frame, conf=YOLO_CONF, verbose=False)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls != BOTTLE_CLASS:
                continue

            det_conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # ---- 第二阶段：裁剪 ROI 并分类 ----
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            label, cls_conf = classify_roi(roi)

            # ---- 绘制结果 ----
            draw_box(frame, x1, y1, x2, y2, label, det_conf, cls_conf)

    # ---- 画面左上角信息 ----
    info_text = f"Frame: {frame_count}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    # ---- 缩放显示 ----
    if RESIZE_DISPLAY:
        display = cv2.resize(frame, RESIZE_DISPLAY)
    else:
        display = frame

    cv2.imshow("Bottle Cap Detector", display)

    # ---- 按键 ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite(f"screenshot_{frame_count:04d}.jpg", frame)
        print(f"已截图: screenshot_{frame_count:04d}.jpg")

cap.release()
cv2.destroyAllWindows()
print(f"\n处理完成，共 {frame_count} 帧")
