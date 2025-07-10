#!/usr/bin/env python3
import sys
import os
import math
import torch
import numpy as np
import pandas as pd
import cv2
import joblib

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis

ALL_LABELS = ["OVERLOADING", "SIDE_SADDLE", "REVERSE_SIDE_SADDLE", "CHILD_IN_FRONT"]

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter/(areaA+areaB-inter+1e-6)

def angle3d(a, b, c):
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    cosang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def main():
    if len(sys.argv)!=2:
        print("Usage: test.py <image.jpg>")
        sys.exit(1)
    img_path = sys.argv[1]

    DET_WEIGHTS   = "models/yolo11x.pt"
    POSE_WEIGHTS  = "models/yolo11x-pose.pt"
    CLF_PATH      = "models/violation_classifier_balanced.pkl"

    for p in (img_path, DET_WEIGHTS, POSE_WEIGHTS, CLF_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    det_model  = YOLO(DET_WEIGHTS)
    pose_model = YOLO(POSE_WEIGHTS)
    clf        = joblib.load(CLF_PATH)

    # depth
    midas      = torch.hub.load("intel-isl/MiDaS","DPT_Large").to(device).eval()
    tfs        = torch.hub.load("intel-isl/MiDaS","transforms")
    midas_tfm  = tfs.dpt_transform

    # age
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    face_app.prepare(ctx_id=0)

    # read image
    pil = Image.open(img_path).convert("RGB")
    img = np.array(pil)
    H, W = img.shape[:2]

    # detect persons & bikes
    det = det_model(img, conf=0.3)[0]
    boxes, classes = det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy().astype(int)
    persons     = [b for b,c in zip(boxes,classes) if c==0]
    motorcycles = [b for b,c in zip(boxes,classes) if c==3]

    # full-image depth
    tm = midas_tfm(np.array(pil))
    im_t = tm["image"] if isinstance(tm,dict) else tm
    if im_t.ndim==3: im_t = im_t.unsqueeze(0)
    with torch.no_grad():
        pred = midas(im_t.to(device))
    depth_map = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H,W), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()

    # assign bike & count riders
    p_df = pd.DataFrame(persons, columns=["x1","y1","x2","y2"])
    p_df["pid"] = p_df.index
    def assign(r):
        for bid,mb in enumerate(motorcycles):
            if compute_iou(r[["x1","y1","x2","y2"]].tolist(), mb)>0.3:
                return bid
        return -1
    p_df["bike_id"] = p_df.apply(assign, axis=1)
    p_df["persons_on_bike"] = p_df.groupby("bike_id")["pid"].transform("count")

    # extract raw features
    raw = []
    for i,(x1,y1,x2,y2) in enumerate(persons):
        x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
        crop = img[y1:y2, x1:x2]
        if crop.size==0: continue

        pr = pose_model(crop, conf=0.4)[0]
        if pr.keypoints.xy.shape[0]==0: continue

        kp2d  = pr.keypoints.xy[0].cpu().numpy()
        conf2 = pr.keypoints.conf[0].cpu().numpy()

        r = {}
        # 2D + score
        for j in range(17):
            r[f"kpt{j}_x"] = float(kp2d[j,0]+x1)
            r[f"kpt{j}_y"] = float(kp2d[j,1]+y1)
            r[f"kpt{j}_v"] = float(conf2[j])

        # bbox dims
        bw, bh = x2-x1, y2-y1
        r["bbox_width"]  = float(bw)
        r["bbox_height"] = float(bh)
        r["bbox_area"]   = float(bw*bh)

        # avg_depth
        pts = kp2d[conf2>0.3].astype(int)
        ds  = [depth_map[y1+py, x1+px] for px,py in pts
               if 0<=x1+px<W and 0<=y1+py<H]
        r["avg_depth"] = float(np.mean(ds)) if ds else -1.0

        # age
        faces = face_app.get(crop[...,::-1])
        r["age"] = float(faces[0].age) if faces else -1.0

        # bike
        r["persons_on_bike"]  = float(p_df.loc[i,"persons_on_bike"])
        r["is_on_motorcycle"] = float(p_df.loc[i,"bike_id"]!=-1)

        # pseudo-3D
        for j in range(17):
            r[f"kpt{j}_z"] = r["avg_depth"]

        # left/right hip & torso angles
        r["left_hip_angle"]   = angle3d(
            (r["kpt5_x"],r["kpt5_y"],r["kpt5_z"]),
            (r["kpt11_x"],r["kpt11_y"],r["kpt11_z"]),
            (r["kpt13_x"],r["kpt13_y"],r["kpt13_z"])
        )
        r["left_torso_lean"]  = angle3d(
            (r["kpt5_x"],r["kpt5_y"],r["kpt5_z"]),
            (r["kpt11_x"],r["kpt11_y"],r["kpt11_z"]),
            (r["kpt11_x"],r["kpt11_y"]-1,r["kpt11_z"])
        )
        r["right_hip_angle"]  = angle3d(
            (r["kpt6_x"],r["kpt6_y"],r["kpt6_z"]),
            (r["kpt12_x"],r["kpt12_y"],r["kpt12_z"]),
            (r["kpt14_x"],r["kpt14_y"],r["kpt14_z"])
        )
        r["right_torso_lean"] = angle3d(
            (r["kpt6_x"],r["kpt6_y"],r["kpt6_z"]),
            (r["kpt12_x"],r["kpt12_y"],r["kpt12_z"]),
            (r["kpt12_x"],r["kpt12_y"]-1,r["kpt12_z"])
        )

        raw.append(r)

    df = pd.DataFrame(raw)
    if df.empty:
        print("No valid persons found.")
        return

    # now normalize & build exactly the same columns used at train time
    root_x = df["kpt11_x"]
    root_y = df["kpt11_y"]
    scale  = df["bbox_height"] + 1e-6

    # normalized 2D keypoints + v
    for j in range(17):
        df[f"kpt{j}_xn"] = (df[f"kpt{j}_x"] - root_x)/scale
        df[f"kpt{j}_yn"] = (df[f"kpt{j}_y"] - root_y)/scale
        df[f"kpt{j}_vn"] = df[f"kpt{j}_v"]

    # extras
    df["shoulder_diff"]     = (df.kpt5_y - df.kpt6_y).abs()/scale
    df["hip_diff"]          = (df.kpt11_y - df.kpt12_y).abs()/scale
    df["head_seat_ratio"]   = (df.kpt0_y  - df.kpt11_y)/scale

    # front-half overlap
    def front_overlap(r):
        bikes = motorcycles if len(motorcycles)>0 else []
        bid   = p_df.loc[r.name,"bike_id"]
        if bid<0 or bid>=len(bikes):
            return 0.0
        mb = bikes[bid]
        midx = (mb[0]+mb[2])/2
        ix1, iy1 = max(r["kpt11_x"], mb[0]), max(r["kpt11_y"], mb[1])
        ix2, iy2 = min(r["kpt11_x"]+r["bbox_width"], midx), min(r["kpt11_y"]+r["bbox_height"], mb[3])
        inter = max(0, ix2-ix1)*max(0, iy2-iy1)
        return inter/((r["bbox_width"]*r["bbox_height"])+1e-6)

    df["front_half_overlap"] = df.apply(front_overlap, axis=1)
    df["helmet"]             = df["age"].lt(0).astype(int)

    # assemble final X in the same order
    kpt2d_norm = [f"kpt{j}_{s}" for j in range(17) for s in ("xn","yn","vn")]
    kpt3d      = [f"kpt{j}_z" for j in range(17)]
    extras     = [
        "bbox_width","bbox_height","bbox_area",
        "avg_depth","age",
        "persons_on_bike","is_on_motorcycle",
        "left_hip_angle","left_torso_lean",
        "right_hip_angle","right_torso_lean",
        "shoulder_diff","hip_diff",
        "head_seat_ratio","front_half_overlap","helmet"
    ]

    feat_cols = kpt2d_norm + kpt3d + extras
    X = df[feat_cols].fillna(-1.0)

    # predict
    ypred = clf.predict(X)
    print(f"\nResults on {img_path}")
    for i,row in enumerate(ypred):
        flags = [ALL_LABELS[j] for j,v in enumerate(row) if v==1]
        print(f" • person {i}: {', '.join(flags) or 'NO VIOLATION'}")

    # visualize...
    vis = img.copy()
    COLORS = {
      "OVERLOADING":        (0,0,255),
      "SIDE_SADDLE":        (0,255,255),
      "REVERSE_SIDE_SADDLE":(255,0,255),
      "CHILD_IN_FRONT":     (0,255,0),
    }
    SKE = [(0,1),(1,3),(0,2),(2,4),
           (5,6),(5,7),(7,9),(6,8),(8,10),
           (11,12),(11,13),(13,15),(12,14),(14,16)]

    for i,(x1,y1,x2,y2) in enumerate(persons):
        x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
        cv2.rectangle(vis,(x1,y1),(x2,y2),(200,200,100),2)
        for a,b in SKE:
            if df.loc[i,f"kpt{a}_v"]>0.3 and df.loc[i,f"kpt{b}_v"]>0.3:
                cv2.line(vis,
                         (int(df.loc[i,f"kpt{a}_x"]), int(df.loc[i,f"kpt{a}_y"])),
                         (int(df.loc[i,f"kpt{b}_x"]), int(df.loc[i,f"kpt{b}_y"])),
                         (180,180,180),1)
        ty, drew = y1-10, False
        for j,lbl in enumerate(ALL_LABELS):
            if ypred[i,j]:
                cv2.putText(vis, lbl, (x1,ty),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[lbl],2)
                ty -= 20; drew = True
        if not drew:
            cv2.putText(vis, "NO VIOLATION", (x1,ty),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1)

    out = Path(img_path).with_name(Path(img_path).stem + "_out.png")
    cv2.imwrite(str(out), vis)
    print("Visualization saved to", out)


if __name__=="__main__":
    main()
