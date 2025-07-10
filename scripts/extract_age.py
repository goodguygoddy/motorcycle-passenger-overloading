import pandas as pd
from insightface.app import FaceAnalysis
from pathlib import Path
import cv2
import os
from tqdm import tqdm

# Paths
DETECT_PATH = "annotations/detect_boxes.csv"
IMAGE_DIR = "runs/detect/exp"
OUTPUT_CSV = "annotations/age.csv"

# Load detection boxes
detect_df = pd.read_csv(DETECT_PATH)
detect_df = detect_df[detect_df["class"] == 0].copy()
detect_df["person_id"] = detect_df.groupby("file").cumcount()

# Initialize InsightFace
print("📦 Initializing InsightFace...")
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0)

# Store results
results = []

# Group by image
for file, group in tqdm(detect_df.groupby("file"), desc="🔍 Predicting age"):
    image_path = Path(IMAGE_DIR) / file
    if not image_path.exists():
        print(f"⚠️ Image not found: {image_path}")
        continue

    img = cv2.imread(str(image_path))
    h, w, _ = img.shape

    faces = app.get(img)

    for _, row in group.iterrows():
        # Person bounding box
        px1, py1, px2, py2 = row["x1"], row["y1"], row["x2"], row["y2"]
        person_box = [px1, py1, px2, py2]

        # Match the closest face inside person bbox
        best_age = None
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox
            face_cx, face_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
                best_age = face.age
                break  # pick first matching face

        results.append({
            "file": file,
            "person_id": row["person_id"],
            "age": best_age if best_age is not None else -1  # fallback
        })

# Save results
age_df = pd.DataFrame(results)
age_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved InsightFace age predictions to {OUTPUT_CSV}")
