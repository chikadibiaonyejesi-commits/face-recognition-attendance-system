from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
import socket

app = Flask(__name__)

STUDENT_DB = "students.csv"
ATTENDANCE_DIR = "attendance"
MODEL_DIR = "models"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

current_course = None
current_date = datetime.now().strftime('%Y-%m-%d')
course_state = {}
current_person = None
person_is_unknown = False

students_df = pd.read_csv(STUDENT_DB, dtype=str)
students_df['reg_number'] = students_df['reg_number'].str.strip()
all_students = {}
for _, row in students_df.iterrows():
    all_students[row['reg_number']] = {
        'name': row['name'],
        'department': row.get('department', 'N/A')
    }
print(f"📚 Loaded {len(all_students)} students")

with open("features.pkl", "rb") as f:
    data = pickle.load(f)
known_features = data["features"]
known_ids = data["ids"]
print(f"🧠 Loaded {len(set(known_ids))} trained faces")

detector = cv2.FaceDetectorYN.create(f"{MODEL_DIR}/face_detection_yunet_2023mar.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(f"{MODEL_DIR}/face_recognition_sface_2021dec.onnx", "")
print("✅ Models loaded")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(3, 640)
camera.set(4, 480)
print("✅ Camera ready")


def get_course_state(course):
    if course not in course_state:
        course_state[course] = {'marked': set(), 'alerted': set(), 'log': []}
    return course_state[course]


def get_attendance_file(course):
    safe_course = course.replace(" ", "_").replace("/", "_")
    return os.path.join(ATTENDANCE_DIR, f"{safe_course}_{current_date}.csv")


def mark_student(reg_number):
    info = all_students.get(reg_number, {'name': 'Unknown', 'department': 'N/A'})
    name = info['name']
    dept = info['department']
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')

    state = get_course_state(current_course)
    if reg_number in state['marked']:
        return False

    file = get_attendance_file(current_course)
    if os.path.exists(file):
        df = pd.read_csv(file, dtype=str)
    else:
        df = pd.DataFrame(columns=['reg_number', 'name', 'department', 'time'])

    new_row = pd.DataFrame([{'reg_number': reg_number, 'name': name, 'department': dept, 'time': time_str}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file, index=False)

    state['marked'].add(reg_number)
    state['log'].append({"reg_number": reg_number, "name": name, "department": dept, "time": time_str})
    print(f"✅ {name} ({dept}) marked at {time_str}")
    return True


def get_present():
    if not current_course:
        return []
    state = get_course_state(current_course)
    return [{"reg_number": e['reg_number'], "name": e['name'], "department": e['department'], "time": e['time']} for e
            in state['log']]


def get_absent():
    if not current_course:
        return []
    state = get_course_state(current_course)
    return [{"reg_number": r, "name": info['name'], "department": info['department']} for r, info in
            all_students.items() if r not in state['marked']]


def generate_frames():
    global current_person, person_is_unknown
    frame_count = 0
    unknown_count = 0

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 6 == 0 and current_course:
            h, w = frame.shape[:2]
            scale = 320 / max(h, w)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            detector.setInputSize((resized.shape[1], resized.shape[0]))
            _, faces = detector.detect(resized)

            if faces is not None and len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[4] * f[5], reverse=True)
                face_align = recognizer.alignCrop(resized, faces[0])
                feat = recognizer.feature(face_align)
                best_dist = float('inf')
                best_idx = -1
                for i, kf in enumerate(known_features):
                    d = recognizer.match(feat, kf, 0)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                if best_dist < 0.25:
                    # Known face detected
                    detected_id = known_ids[best_idx]
                    person_is_unknown = False
                    unknown_count = 0

                    if detected_id != current_person:
                        state = get_course_state(current_course)
                        if detected_id not in state['marked']:
                            mark_student(detected_id)
                        elif detected_id not in state['alerted']:
                            info = all_students.get(detected_id, {'name': 'Unknown'})
                            print(f"⏭️ {info['name']} already marked")
                            state['alerted'].add(detected_id)
                        current_person = detected_id
                else:
                    # Unknown face
                    unknown_count += 1
                    if unknown_count >= 3:
                        person_is_unknown = True
                        current_person = None
            else:
                # No face at all
                unknown_count = 0
                person_is_unknown = False
                current_person = None

        h, w = frame.shape[:2]
        scale = 320 / max(h, w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = detector.detect(resized)
        if faces is not None:
            for face in faces:
                x, y, fw, fh = face[:4].astype(int)
                x = int(x * w / resized.shape[1])
                y = int(y * h / resized.shape[0])
                fw = int(fw * w / resized.shape[1])
                fh = int(fh * h / resized.shape[0])

                if person_is_unknown:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif current_person:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                    info = all_students.get(current_person, {'name': 'Unknown'})
                    cv2.putText(frame, info['name'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def home():
    if not current_course:
        return redirect(url_for('select_course'))
    return redirect(url_for('dashboard'))


@app.route('/select-course', methods=['GET', 'POST'])
def select_course():
    global current_course, current_person, person_is_unknown
    if request.method == 'POST':
        course = request.form.get('course_name', '').strip().upper()
        if course:
            current_course = course
            current_person = None
            person_is_unknown = False
            get_course_state(course)
            return redirect(url_for('dashboard'))
    return render_template('select_course.html')


@app.route('/dashboard')
def dashboard():
    if not current_course:
        return redirect(url_for('select_course'))
    return render_template('dashboard.html',
                           course=current_course,
                           present=len(get_present()),
                           absent=len(get_absent()),
                           total=len(all_students),
                           present_students=get_present(),
                           absent_students=get_absent())


@app.route('/mark')
def mark():
    if not current_course:
        return redirect(url_for('select_course'))
    return render_template('mark.html', course=current_course)


@app.route('/present')
def present():
    if not current_course:
        return redirect(url_for('select_course'))
    return render_template('present.html', course=current_course, students=get_present())


@app.route('/absent')
def absent():
    if not current_course:
        return redirect(url_for('select_course'))
    return render_template('absent.html', course=current_course, students=get_absent())


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    if not current_course:
        return jsonify(
            {'message': 'No course selected', 'marked': 0, 'total': 0, 'all_marked': False, 'present': 0, 'absent': 0,
             'log': []})

    state = get_course_state(current_course)
    all_marked = len(state['marked']) >= len(all_students)

    if all_marked:
        msg = "🎉 All checked in!"
    elif current_person:
        info = all_students.get(current_person, {'name': 'Unknown'})
        if current_person in state['marked']:
            msg = f"{info['name']} already checked in."
        else:
            msg = f"Recognized: {info['name']}"
    elif person_is_unknown:
        msg = "❌ Unknown person detected"
    else:
        msg = "⏳ Waiting for face..."

    return jsonify({
        'message': msg,
        'marked': len(state['marked']),
        'total': len(all_students),
        'all_marked': all_marked,
        'present': len(get_present()),
        'absent': len(get_absent()),
        'log': state['log'][-10:]
    })


@app.route('/api/present')
def api_present():
    if not current_course:
        return jsonify([])
    return jsonify(get_present())


@app.route('/api/absent')
def api_absent():
    if not current_course:
        return jsonify([])
    return jsonify(get_absent())


@app.route('/shutdown')
def shutdown():
    camera.release()
    cv2.destroyAllWindows()
    return "Server stopped."


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'


local_ip = get_local_ip()

print("\n" + "=" * 55)
print("  👁️  FaceAttendance System v1.0")
print("=" * 55)
print(f"  💻 Local:    http://127.0.0.1:5000")
print(f"  📱 Network:  http://{local_ip}:5000")
print("=" * 55 + "\n")

app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)