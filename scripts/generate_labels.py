# scripts/generate_labels.py

import csv
from pathlib import Path

ALL_LABELS = ["OVERLOADING","SIDE_SADDLE","REVERSE_SIDE_SADDLE","CHILD_IN_FRONT"]

def main(root="data/images", out="annotations/labels.csv"):
    root = Path(root)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file"] + ALL_LABELS)
        for cls in sorted(root.iterdir()):
            if not cls.is_dir(): continue
            labels = set(cls.name.split("+"))
            for img in sorted(cls.iterdir()):
                if img.suffix.lower() not in (".jpg",".jpeg",".png"): continue
                row = [ img.name ]
                row += [1 if L in labels else 0 for L in ALL_LABELS]
                writer.writerow(row)
    print(f"Wrote {out}")

if __name__=="__main__":
    main()
