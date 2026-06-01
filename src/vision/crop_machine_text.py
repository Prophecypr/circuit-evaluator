"""裁剪机画数值图片 → OCR训练数据。追加模式，不会覆盖已有裁剪。
输出: data/ocr_machine/crops/ (裁剪图) + labels_auto.txt (标注文件)
"""
import sys, os, cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO
from src.vision.train_ocr import load_trained_model, predict as crnn_predict

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "data" / "ocr_machine" / "crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)

text_model = YOLO(str(ROOT / "runs" / "detect" / "circuit_text" / "weights" / "best.pt"))
ocr_model, ocr_chars, c2i, i2c, img_h = load_trained_model()

# 从已有数量继续
existing = list(OUT_DIR.glob("machine_*.png"))
crop_idx = len(existing)
all_labels = []

img_files = sorted(ROOT.glob("数值*.jpg"))
# 只处理新图片（编号>7的是Times New Roman）
new_files = [f for f in img_files if int(f.stem.replace("数值", "")) > 7]
print(f"处理 {len(new_files)} 张 Times New Roman 图片 (已有 {crop_idx} 张 Arial 裁剪)")

for img_path in new_files:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = text_model(str(img_path))[0]

    for box in (results.boxes or []):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        raw = crnn_predict(ocr_model, crop, ocr_chars, i2c, img_h)
        crop_name = f"machine_{crop_idx:04d}.png"
        cv2.imwrite(str(OUT_DIR / crop_name), crop)
        all_labels.append(f"data/ocr_machine/crops/{crop_name}\t{raw}")
        crop_idx += 1

# 追加写 labels
label_file = ROOT / "data" / "ocr_machine" / "labels_auto.txt"
with open(label_file, "a", encoding="utf-8") as f:
    for line in all_labels:
        f.write(line + "\n")

print(f"新裁剪: {len(all_labels)} 张 → 总计 {crop_idx} 张")
print(f"标注追加到: {label_file}")
print(f"打开 labels_auto.txt 核对新增的 {len(all_labels)} 行")
