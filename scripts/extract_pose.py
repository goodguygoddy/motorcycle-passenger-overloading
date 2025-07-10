import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

# COCO-style skeleton for the 17 keypoints
SKELETON = [
    (0,1),(1,3),(0,2),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(11,13),(13,15),(12,14),(14,16)
]

KP_CONF_THR = 0.3  # Confidence threshold

def main():
    model = YOLO("models/yolo11x-pose.pt")  # Replace with yolo11m-pose.pt if needed
    image_dir = Path("data/images")
    out_dir = Path("runs/pose/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"📷 Found {len(image_paths)} images")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        results = model.predict(str(img_path), conf=0.25, device=0 if torch.cuda.is_available() else "cpu")[0]
        keypoints_xy = results.keypoints.xy.cpu().numpy()       # (N, 17, 2)
        keypoints_conf = results.keypoints.conf.cpu().numpy()   # (N, 17)

        for person_id, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
            # Draw keypoints
            for (x, y), c in zip(xy, conf):
                if c >= KP_CONF_THR:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Draw skeleton
            for a, b in SKELETON:
                xa, ya, ca = xy[a][0], xy[a][1], conf[a]
                xb, yb, cb = xy[b][0], xy[b][1], conf[b]
                if ca >= KP_CONF_THR and cb >= KP_CONF_THR:
                    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

            flat = np.hstack([xy, conf.reshape(-1, 1)]).reshape(-1).tolist()
            rows.append([img_path.name, person_id] + flat)

        cv2.imwrite(str(out_dir / img_path.name), img)
        torch.cuda.empty_cache()  # ✅ Prevent VRAM spikes

    # Save keypoints to CSV
    cols = ["file", "person_id"] + [f"kpt{i}_{axis}" for i in range(17) for axis in ("x", "y", "v")]
    pd.DataFrame(rows, columns=cols).to_csv("annotations/keypoints.csv", index=False)

    print(f"✅ Saved visualizations → {out_dir}")
    print("✅ Keypoints CSV → annotations/keypoints.csv")

if __name__ == "__main__":
    main()
