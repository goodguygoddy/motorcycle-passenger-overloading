#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

ALL_LABELS = ["OVERLOADING", "SIDE_SADDLE", "REVERSE_SIDE_SADDLE", "CHILD_IN_FRONT"]

def compute_iou(boxA, boxB):
    xA,yA = max(boxA[0],boxB[0]), max(boxA[1],boxB[1])
    xB,yB = min(boxA[2],boxB[2]), min(boxA[3],boxB[1])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter==0: return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def angle3d(a,b,c):
    v1 = np.array(a)-np.array(b)
    v2 = np.array(c)-np.array(b)
    cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def main():
    # ── load annotation CSVs
    kps    = pd.read_csv("annotations/keypoints.csv")
    det    = pd.read_csv("annotations/detect_boxes.csv")
    depth  = pd.read_csv("annotations/depth.csv")
    age_df = pd.read_csv("annotations/age.csv")

    # ── split people vs bikes
    persons = det[det["class"]==0].reset_index(drop=True)
    motors  = det[det["class"]==3].reset_index(drop=True)
    persons["person_id"] = persons.groupby("file").cumcount()

    # ── merge into one DF
    df = (
        kps
        .merge(persons, on=["file","person_id"], suffixes=("_kp","_box"))
        .merge(depth,   on=["file","person_id"], how="left")
        .merge(age_df,  on=["file","person_id"], how="left")
    )

    # ── bbox features
    df["bbox_width"]  = df["x2"] - df["x1"]
    df["bbox_height"] = df["y2"] - df["y1"]
    df["bbox_area"]   = df["bbox_width"] * df["bbox_height"]

    # ── assign bike_id
    def assign_bike(r):
        bikes = motors[motors.file==r.file].reset_index(drop=True)
        for bid, mb in bikes.iterrows():
            if compute_iou([r.x1,r.y1,r.x2,r.y2],
                           [mb.x1,mb.y1,mb.x2,mb.y2])>0.3:
                return bid
        return -1
    df["bike_id"] = df.apply(assign_bike, axis=1)

    # ── riders per bike & flag
    df["persons_on_bike"]  = df.groupby(["file","bike_id"])["person_id"].transform("count")
    df["is_on_motorcycle"] = (df["bike_id"]!=-1).astype(int)

    # ── pseudo-3D depths
    for j in range(17):
        df[f"kpt{j}_z"] = df["avg_depth"]

    # ── side-saddle + reverse-saddle angles
    df["left_hip_angle"]   = df.apply(lambda r: angle3d(
        (r.kpt5_x,r.kpt5_y,r.kpt5_z),
        (r.kpt11_x,r.kpt11_y,r.kpt11_z),
        (r.kpt13_x,r.kpt13_y,r.kpt13_z)
    ),axis=1)
    df["left_torso_lean"]  = df.apply(lambda r: angle3d(
        (r.kpt5_x,r.kpt5_y,r.kpt5_z),
        (r.kpt11_x,r.kpt11_y,r.kpt11_z),
        (r.kpt11_x,r.kpt11_y-1,r.kpt11_z)
    ),axis=1)
    df["right_hip_angle"]  = df.apply(lambda r: angle3d(
        (r.kpt6_x,r.kpt6_y,r.kpt6_z),
        (r.kpt12_x,r.kpt12_y,r.kpt12_z),
        (r.kpt14_x,r.kpt14_y,r.kpt14_z)
    ),axis=1)
    df["right_torso_lean"] = df.apply(lambda r: angle3d(
        (r.kpt6_x,r.kpt6_y,r.kpt6_z),
        (r.kpt12_x,r.kpt12_y,r.kpt12_z),
        (r.kpt12_x,r.kpt12_y-1,r.kpt12_z)
    ),axis=1)

    # ── normalize all 2D keypoints around kpt11, scale by bbox_height
    root_x = df["kpt11_x"]
    root_y = df["kpt11_y"]
    scale  = df["bbox_height"] + 1e-6

    for j in range(17):
        df[f"kpt{j}_xn"] = (df[f"kpt{j}_x"] - root_x) / scale
        df[f"kpt{j}_yn"] = (df[f"kpt{j}_y"] - root_y) / scale
        df[f"kpt{j}_vn"] = df[f"kpt{j}_v"]

    # ── extras
    df["shoulder_diff"]   = (df.kpt5_y - df.kpt6_y).abs() / scale
    df["hip_diff"]        = (df.kpt11_y - df.kpt12_y).abs() / scale
    df["head_seat_ratio"] = (df.kpt0_y - df.kpt11_y) / scale

    def front_overlap(r):
        bikes = motors[motors.file==r.file].reset_index(drop=True)
        bid   = r.bike_id
        if bid<0 or bid>=len(bikes):
            return 0.0
        mb    = bikes.loc[bid]
        midx  = (mb.x1+mb.x2)/2.0
        ix1,iy1 = max(r.x1,mb.x1), max(r.y1,mb.y1)
        ix2,iy2 = min(r.x2,midx),   min(r.y2,mb.y2)
        inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
        area    = (r.x2-r.x1)*(r.y2-r.y1) + 1e-6
        return inter/area

    df["front_half_overlap"] = df.apply(front_overlap, axis=1)
    df["helmet"]             = df["age"].fillna(-1).lt(0).astype(int)

    # ── assemble feature matrix exactly as train used
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
    X = df[kpt2d_norm + kpt3d + extras].fillna(-1)

    # ── load your trained multi‐output HGB
    clf = joblib.load("models/violation_classifier_balanced.pkl")

    # ── predict all four at once
    ypred = clf.predict(X)

    # ── write out predictions.csv
    out = pd.DataFrame({
        "file":      df["file"],
        "person_id": df["person_id"],
        **{lbl: ypred[:,i] for i,lbl in enumerate(ALL_LABELS)}
    })
    Path("annotations").mkdir(exist_ok=True)
    out.to_csv("annotations/predictions.csv", index=False)
    print("✅  Saved → annotations/predictions.csv")

if __name__=="__main__":
    main()
