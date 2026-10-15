import cv2
import pickle
import os
import numpy as np

with open("features.pkl", "rb") as f:
    data = pickle.load(f)
features = data["features"]
ids = data["ids"]

print("Features stored in order:")
for i, (feat, id_) in enumerate(zip(features, ids)):
    print(f"  [{i}] → {id_}")

detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

print("\nTesting each image:\n")
correct = 0
total = 0

for folder in sorted(os.listdir("dataset")):
    folder_path = os.path.join("dataset", folder)
    if not os.path.isdir(folder_path): continue
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, file)
            img = cv2.imread(path)
            if img is None: continue
            h, w = img.shape[:2]
            scale = 320 / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            detector.setInputSize((resized.shape[1], resized.shape[0]))
            _, faces = detector.detect(resized)
            if faces is None or len(faces) == 0: continue
            face_align = recognizer.alignCrop(resized, faces[0])
            feat = recognizer.feature(face_align)

            best_dist = float('inf')
            best_id = None
            for i, kf in enumerate(features):
                d = recognizer.match(feat, kf, 0)
                if d < best_dist:
                    best_dist = d
                    best_id = ids[i]

            total += 1
            if best_id == folder:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            print(f"{status} {path} → {best_id} (dist: {best_dist:.4f}) TRUE: {folder}")

print(f"\n📊 Accuracy: {correct}/{total} = {100 * correct / total:.1f}%")

