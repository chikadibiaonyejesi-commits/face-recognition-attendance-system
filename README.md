# Face Recognition Attendance System
A web-based attendance system that recognizes students' faces through a webcam and records attendance automatically.

Tools: Python, Flask, OpenCV (YuNet face detection, SFace face recognition), NumPy, Pandas

Features
- Detects and recognizes faces in live webcam video
- Saves attendance with date and time to a CSV file
- Reads student details (name, reg number, department) from a CSV database
- Accessible from a phone or another device on the same network

How it works
Face features are extracted from training images and saved. During attendance, each face from the camera is compared to the saved features, and the closest match is marked present. Unmatched faces show as UNKNOWN.

Run it
pip install flask opencv-python numpy pandas
python app.py

Built with: AI-assisted development
Note: Runs on a local network, so it has no live link. Demo available on request.
