import urllib.request
import os

os.makedirs("models", exist_ok=True)

print("Downloading face detection model...")
url1 = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
urllib.request.urlretrieve(url1, "models/face_detection_yunet_2023mar.onnx")

print("Downloading face recognition model...")
url2 = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
urllib.request.urlretrieve(url2, "models/face_recognition_sface_2021dec.onnx")

print("✅ Models downloaded!")
