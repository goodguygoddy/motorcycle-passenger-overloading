#!/usr/bin/env python3
"""
scripts/test.py

End-to-end passenger overloading violation detection on images or directories.
Usage:
    python scripts/test.py <image.jpg | image_folder>

This script processes images, predicts violations, and outputs annotated images with bounding boxes, labels, and pose skeletons.
"""
import sys
import os
from pathlib import Path
import numpy as np
import joblib
import cv2
from ultralytics import YOLO

# Thresholds
THR_SIDE_SADDLE = 0.7
THR_CHILD_FRONT = 0.8

# COCO pose skeleton connections (17 keypoints)
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
]

# Helper: IoU between two boxes
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# Rule-based validation
def is_side_saddle(r):
    if r['leg_vis'] >= 1:
        return (r['theta_L'] > 120 and r['leg_asym'] > 25 and r['foot_drop'] > 30)
    else:
        return (r['torso_pitch'] < 20 and r['torso_disp'] < 0.4)

# Compute 14-dim feature vector
def compute_feature_vector(kp, bbox_p, bbox_b):
    import numpy as _np
    def _angle(a, b, c):
        BA = _np.array(a) - _np.array(b)
        BC = _np.array(c) - _np.array(b)
        cosang = _np.dot(BA, BC) / (_np.linalg.norm(BA) * _np.linalg.norm(BC) + 1e-6)
        return _np.degrees(_np.arccos(_np.clip(cosang, -1, 1)))

    θ_R = _angle(kp[11][:2], kp[13][:2], kp[15][:2])
    θ_L = _angle(kp[12][:2], kp[14][:2], kp[16][:2])
    pel, sh = _np.array(kp[11][:2]), _np.array(kp[5][:2])
    vert = _np.array([0, -1])
    φ   = _np.degrees(_np.arccos(_np.dot(sh-pel, vert) / 
                        (_np.linalg.norm(sh-pel)*_np.linalg.norm(vert)+1e-6)))

    sh_w = _np.linalg.norm(_np.array(kp[6][:2]) - _np.array(kp[5][:2])) + 1e-6
    hip_w = _np.linalg.norm(_np.array(kp[11][:2]) - _np.array(kp[12][:2])) / sh_w
    foot_drop   = (kp[15][1] - kp[11][1]) / sh_w
    torso_disp  = (kp[5][1]  - kp[11][1]) / sh_w
    x_hip_norm  = ((kp[11][0] + kp[12][0]) / 2) / (bbox_p[2] + 1e-6)

    d1 = _np.linalg.norm(_np.array(kp[7][:2])  - _np.array(kp[11][:2]))
    d2 = _np.linalg.norm(_np.array(kp[4][:2])  - _np.array(kp[12][:2]))
    arm_torso   = min(d1, d2) / sh_w

    leg_vis           = int((kp[13][2]>0.25 and kp[15][2]>0.25) + \
                            (kp[14][2]>0.25 and kp[16][2]>0.25))
    torso_leg_ratio   = torso_disp / (foot_drop + 1e-6)
    shoulder_hip_ratio= sh_w / (hip_w * sh_w + 1e-6)
    leg_asym          = abs(θ_L - θ_R)
    head_w            = _np.linalg.norm(_np.array(kp[2][:2]) - _np.array(kp[1][:2])) / sh_w
    hand_torso        = min(d1, d2) / sh_w

    return {
        'theta_R': θ_R,
        'theta_L': θ_L,
        'torso_pitch': φ,
        'hip_width': hip_w,
        'foot_drop': foot_drop,
        'x_hip_norm': x_hip_norm,
        'torso_disp': torso_disp,
        'arm_torso': arm_torso,
        'leg_vis': leg_vis,
        'torso_leg_ratio': torso_leg_ratio,
        'shoulder_hip_ratio': shoulder_hip_ratio,
        'leg_asym': leg_asym,
        'head_w': head_w,
        'hand_torso': hand_torso
    }

# Process a single image
def process_image(img_path, det_model, pose_model, clf):
    print(f"\n🖼 Processing {img_path}")
    ori = cv2.imread(img_path)
    img = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)

    det = det_model(img, conf=0.3)[0]
    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy().astype(int)
    persons = [b for b,c in zip(boxes,classes) if c==0]
    bikes   = [b for b,c in zip(boxes,classes) if c==3]

    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    for i,pbox in enumerate(persons):
        y1, y2 = int(pbox[1]), int(pbox[3])
        x1, x2 = int(pbox[0]), int(pbox[2])
        crop = img[y1:y2, x1:x2]

        # Pose estimation
        res = pose_model(crop, conf=0.25)[0]
        kp_xy = res.keypoints.xy[0].cpu().numpy() + [x1, y1]
        kp_cf = res.keypoints.conf[0].cpu().numpy()
        kp    = {idx: (float(kp_xy[idx][0]), float(kp_xy[idx][1]), float(kp_cf[idx]))  
                 for idx in range(len(kp_cf))}

        # Draw pose skeleton
        for a,b in SKELETON:
            xa,ya,ca = kp[a]
            xb,yb,cb = kp[b]
            if ca>0.25 and cb>0.25:
                cv2.line(ori, (int(xa),int(ya)), (int(xb),int(yb)), (255,0,0), 2)
        for idx,(x,y,c) in kp.items():
            if c>0.25:
                cv2.circle(ori, (int(x),int(y)), 3, (0,0,255), -1)

        # Associate to best bike
        best_iou, best_b = 0.0, None
        for b in bikes:
            iou = compute_iou(pbox,b)
            if iou>best_iou:
                best_iou, best_b = iou, b
        riders_on_bike = sum(compute_iou(q,best_b)>0.3 for q in persons) if best_b is not None else 1
        bike_bbox = best_b.tolist() if best_b is not None else [0,0,0,0]

        # Features & classification
        feats = compute_feature_vector(kp, pbox.tolist(), bike_bbox)
        x     = np.array([list(feats.values())])
        probs = clf.predict_proba(x)
        p_side  = probs[0][0][1]
        p_child = probs[1][0][1]
        side_pred  = p_side  > THR_SIDE_SADDLE
        child_pred = p_child > THR_CHILD_FRONT
        side_final  = side_pred  and is_side_saddle(feats)
        child_final = child_pred
        overload    = riders_on_bike >= 3

        print(f"➡️ Person {i}:")
        print(f"  SIDE_SADDLE: prob={p_side:.2f}, pred={side_pred}, final={side_final}")
        print(f"  CHILD_FRONT: prob={p_child:.2f}, pred={child_pred}, final={child_final}")
        print(f"  OVERLOAD (count={riders_on_bike}): {overload}")

        # Draw bounding box & label
        cv2.rectangle(ori, (x1,y1), (x2,y2), (0,255,0), 2)
        lbl = []
        if overload:    lbl.append('OVERLOAD')
        if side_final:  lbl.append('SIDE_SADDLE')
        if child_final: lbl.append('CHILD_FRONT')
        text = ','.join(lbl) if lbl else 'NO VIOLATION'
        cv2.putText(ori, text, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, ori)
    print(f"Annotated image saved to {out_path}")

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Usage: python scripts/test.py <image.jpg|image_folder>")
        sys.exit(1)
    target = sys.argv[1]

    det_model  = YOLO('models/yolo11x.pt')
    pose_model = YOLO('models/yolo11x-pose.pt')
    clf        = joblib.load('models/violation_classifier_balanced.pkl')

    if os.path.isdir(target):
        for img in sorted(Path(target).glob('*')):
            if img.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']:
                process_image(str(img), det_model, pose_model, clf)
    else:
        process_image(target, det_model, pose_model, clf)
