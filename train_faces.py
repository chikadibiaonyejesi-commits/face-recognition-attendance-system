import cv2
import pickle
import os
import numpy as np

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx", "")

features = []
ids = []

print("🧠 Training...\n")

for folder_name in sorted(os.listdir("dataset")):
    folder_path = os.path.join("dataset", folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder_name}")

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)
            if img is None:
                continue

            h, w = img.shape[:2]
            scale = 320 / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            detector.setInputSize((resized.shape[1], resized.shape[0]))
            _, faces = detector.detect(resized)

            if faces is None or len(faces) == 0:
                print(f"   ⚠️ No face in {file}")
                continue

            face = max(faces, key=lambda f: f[4] * f[5])
            face_align = recognizer.alignCrop(resized, face)
            feat = recognizer.feature(face_align)

            features.append(feat)
            ids.append(folder_name)
            print(f"   ✅ {file} → {folder_name}")

# Save
with open("features.pkl", "wb") as f:
    pickle.dump({"features": features, "ids": ids}, f)

print(f"\n✅ Done! Trained {len(set(ids))} student(s) with {len(features)} photos")
print(f"   IDs in order: {ids}")