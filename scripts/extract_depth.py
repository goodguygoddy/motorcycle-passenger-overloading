import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load MiDaS model
print("📦 Loading MiDaS depth model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Keypoints CSV
kp_path = Path("annotations/keypoints.csv")
kps = pd.read_csv(kp_path)
grouped = kps.groupby("file")

# Directories
image_dir = Path("data/images")
vis_dir = Path("runs/depth/exp")
vis_dir.mkdir(parents=True, exist_ok=True)

out_rows = []

print("📸 Processing images for depth estimation...")
for file, group in tqdm(grouped):
    # Get full image path
    img_path = next(image_dir.rglob(file), None)
    if not img_path:
        print(f"❌ Image not found: {file}")
        continue

    # Load image and transform
    img = Image.open(img_path).convert("RGB")
    # Convert PIL Image to numpy array for MiDaS transform
    img_array = np.array(img)
    input_batch = transform(img_array)
    input_image = input_batch.to(device)

    with torch.no_grad():
        prediction = midas(input_image)  # input_image already has batch dimension
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Save visualization
    plt.imsave(vis_dir / file, depth_map, cmap='inferno')

    # Extract average depth for each person
    for _, row in group.iterrows():
        person_id = row["person_id"]
        xs = np.array([row[f"kpt{i}_x"] for i in range(17)])
        ys = np.array([row[f"kpt{i}_y"] for i in range(17)])
        confs = np.array([row[f"kpt{i}_v"] for i in range(17)])
        mask = confs > 0.3

        if not mask.any():
            avg_depth = -1
        else:
            coords = np.stack([ys[mask], xs[mask]], axis=1).astype(int)
            valid_depths = [depth_map[y, x]
                            for y, x in coords
                            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]]
            avg_depth = float(np.mean(valid_depths)) if valid_depths else -1

        out_rows.append({"file": file, "person_id": person_id, "avg_depth": avg_depth})

# Save CSV
out_df = pd.DataFrame(out_rows)
out_df.to_csv("annotations/depth.csv", index=False)
print("✅ Saved per-person depth to annotations/depth.csv")
print(f"🖼️  Saved visualizations to {vis_dir}/")
