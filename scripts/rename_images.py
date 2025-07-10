# scripts/rename_images.py

import os
from pathlib import Path

def rename_images(root_dir: str):
    """
    Renames images in each subfolder of `root_dir` to a consistent pattern:
      <FOLDERNAME>_<4-digit-index>.<ext>
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory '{root_dir}' does not exist.")
    
    for class_dir in sorted(root.iterdir()):
        if class_dir.is_dir():
            # Collect image files with common extensions
            images = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
            if not images:
                continue
            for idx, img_path in enumerate(images, start=1):
                new_name = f"{class_dir.name}_{idx:04d}{img_path.suffix.lower()}"
                new_path = class_dir / new_name
                # Avoid overwriting if already named
                if img_path.name != new_name:
                    img_path.rename(new_path)
            print(f"Renamed {len(images)} images in '{class_dir.name}'")

if __name__ == "__main__":
    rename_images("data/images")
