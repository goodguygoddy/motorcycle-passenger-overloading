from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import cv2
import torch

TARGET_CLASSES = [0, 3]  # COCO: person, motorcycle
CLASS_NAMES = {0: "person", 3: "motorcycle"}

def main():
    model = YOLO("models/yolo11x.pt")
    image_dir = Path("data/images")
    out_dir = Path("runs/detect/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    print(f"📷 Found {len(image_paths)} images for detection.")

    rows = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠️ Skipping invalid image: {img_path}")
            continue

        # Optional: resize large images
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Predict for one image
        result = model.predict(
            source=frame,
            conf=0.25,
            device=0,
            save=False,
            stream=False,
            verbose=False
        )[0]

        for box in result.boxes:
            cls = int(box.cls)
            if cls not in TARGET_CLASSES:
                continue

            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            rows.append((img_path.name, cls, conf, x1, y1, x2, y2))

            label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save annotated image
        cv2.imwrite(str(out_dir / img_path.name), frame)

        # ✅ Release memory after each image
        torch.cuda.empty_cache()

    pd.DataFrame(rows, columns=["file", "class", "conf", "x1", "y1", "x2", "y2"]).to_csv(
        "annotations/detect_boxes.csv", index=False
    )

    print("✅ Saved filtered detections and images with only person/motorcycle.")

if __name__ == "__main__":
    main()
