#!/usr/bin/env python3
"""
scripts/train_classifier.py

Train a multi-output violation classifier using pre-extracted annotations:
- load detect_boxes.csv, keypoints.csv, depth.csv, labels.csv
- merge and compute the 14-dimensional geometric feature vector per person
- train a HistGradientBoostingClassifier wrapped in MultiOutputClassifier
- save the model to models/violation_classifier_balanced.pkl
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------- Configuration ---------------
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ANNOT_DIR    = os.path.join(PROJECT_ROOT, 'annotations')
MODEL_OUT    = os.path.join(PROJECT_ROOT, 'models', 'violation_classifier_balanced.pkl')
TEST_SIZE    = 0.3
RANDOM_SEED  = 42

# ------------- Load annotation CSVs ----------
print("📥 Loading CSVs from", ANNOT_DIR)
detect_df = pd.read_csv(os.path.join(ANNOT_DIR, 'detect_boxes.csv'))
kp_df     = pd.read_csv(os.path.join(ANNOT_DIR, 'keypoints.csv'))
depth_df  = pd.read_csv(os.path.join(ANNOT_DIR, 'depth.csv'))
labels_df = pd.read_csv(os.path.join(ANNOT_DIR, 'labels.csv'))

# ------------- Prepare person & bike IDs -------
persons = detect_df[detect_df['class'] == 0].copy()
persons['person_id'] = persons.groupby('file').cumcount()
bikes   = detect_df[detect_df['class'] == 3].copy()
bikes['bike_id']    = bikes.groupby('file').cumcount()

# ------------- Normalize labels ---------------
# Merge SIDE_SADDLE and REVERSE_SIDE_SADDLE
labels_df['SIDE_SADDLE'] = (
    (labels_df['SIDE_SADDLE'] == 1) |
    (labels_df['REVERSE_SIDE_SADDLE'] == 1)
).astype(int)
labels_df['PASSENGER_OVERLOAD'] = labels_df['OVERLOADING']
labels_df['CHILD_FRONT']         = labels_df['CHILD_IN_FRONT']
labels_df = labels_df[['file','PASSENGER_OVERLOAD','SIDE_SADDLE','CHILD_FRONT']]

# ------------- Merge all annotations ----------
print("🔗 Merging annotations into one DataFrame...")
df = kp_df.merge(
    persons[['file','person_id','x1','y1','x2','y2']],
    on=['file','person_id'], how='inner'
).merge(
    bikes[['file','x1','y1','x2','y2']].rename(
        columns={'x1':'x1_bike','y1':'y1_bike','x2':'x2_bike','y2':'y2_bike'}
    ),
    on='file', how='left'
).merge(
    depth_df, on=['file','person_id'], how='left'
).merge(
    labels_df, on='file', how='left'
)

# ------------- Feature computation -------------
def compute_feature_vector(bbox_bike, bbox_person, kp, depth):
    import numpy as _np
    def _angle(a, b, c):
        BA = _np.array(a) - _np.array(b)
        BC = _np.array(c) - _np.array(b)
        cosang = _np.dot(BA, BC) / (_np.linalg.norm(BA) * _np.linalg.norm(BC) + 1e-6)
        return _np.degrees(_np.arccos(_np.clip(cosang, -1, 1)))

    θ_R = _angle(kp[11][:2], kp[13][:2], kp[15][:2])
    θ_L = _angle(kp[12][:2], kp[14][:2], kp[16][:2])
    pel = _np.array(kp[11][:2]); sh = _np.array(kp[5][:2]); vert = _np.array([0, -1])
    φ   = _np.degrees(_np.arccos(_np.dot(sh - pel, vert) / (
        _np.linalg.norm(sh - pel) * _np.linalg.norm(vert) + 1e-6
    )))

    sh_w    = _np.linalg.norm(_np.array(kp[6][:2]) - _np.array(kp[5][:2])) + 1e-6
    hip_w   = _np.linalg.norm(_np.array(kp[11][:2]) - _np.array(kp[12][:2])) / sh_w
    foot_drop  = (kp[15][1] - kp[11][1]) / sh_w
    torso_disp = (kp[5][1]  - kp[11][1]) / sh_w
    x_hip_norm = ((kp[11][0] + kp[12][0]) / 2) / (bbox_person[2] + 1e-6)

    d1 = _np.linalg.norm(_np.array(kp[7][:2]) - _np.array(kp[11][:2]))
    d2 = _np.linalg.norm(_np.array(kp[4][:2]) - _np.array(kp[12][:2]))
    arm_torso = min(d1, d2) / sh_w

    leg_vis          = int((kp[13][2] > 0.25 and kp[15][2] > 0.25) +
                           (kp[14][2] > 0.25 and kp[16][2] > 0.25))
    λ_torso          = torso_disp / (foot_drop + 1e-6)
    sh_hip_ratio     = sh_w / (hip_w * sh_w + 1e-6)
    Δθ               = abs(θ_L - θ_R)
    head_w           = _np.linalg.norm(_np.array(kp[2][:2]) - _np.array(kp[1][:2])) / sh_w
    hand_torso       = min(d1, d2) / sh_w

    return [
        θ_R, θ_L, φ, hip_w, foot_drop, x_hip_norm,
        torso_disp, arm_torso, leg_vis, λ_torso,
        sh_hip_ratio, Δθ, head_w, hand_torso
    ]

print("⚙️ Computing features & labels...")
X, Y = [], []
for _, row in df.iterrows():
    kp       = {i: (row[f'kpt{i}_x'], row[f'kpt{i}_y'], row[f'kpt{i}_v']) for i in range(17)}
    bbox_p   = [row['x1'], row['y1'], row['x2'], row['y2']]
    bbox_b   = [row['x1_bike'], row['y1_bike'], row['x2_bike'], row['y2_bike']]
    depth    = row.get('avg_depth', -1)

    X.append(compute_feature_vector(bbox_b, bbox_p, kp, depth))
    Y.append([row['PASSENGER_OVERLOAD'], row['SIDE_SADDLE'], row['CHILD_FRONT']])

X = np.array(X)
Y = np.array(Y)

# ------------- Split, Train, Evaluate -------------
print(f"Split {len(X)} samples → train/test {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%}")
Xtr, Xte, ytr, yte = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED,
)

print("🤖 Training multi-output HistGradientBoostingClassifier...")
clf = MultiOutputClassifier(
    HistGradientBoostingClassifier(random_state=RANDOM_SEED)
)
clf.fit(Xtr, ytr)

print("\n📊 Classification report on test set:")
ypred = clf.predict(Xte)
print(classification_report(yte, ypred,
      target_names=['OVERLOAD','SIDE_SADDLE','CHILD_FRONT'],
      zero_division=0))
print("Overall accuracy:", accuracy_score(yte, ypred))

# ------------- Save Model ------------------------
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(clf, MODEL_OUT)
print(f"✅ Saved trained classifier to {MODEL_OUT}")
