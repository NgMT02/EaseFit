from flask import Flask, Response, render_template, request, redirect, url_for, send_file
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

JOINTS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def extract_joint_angles(landmarks):
    joint_angles = {}

    for joint_name, joint_index in JOINTS.items():
        # Get the coordinates for the joints required for angle calculation
        if joint_name.endswith('elbow'):
            shoulder = landmarks[JOINTS['left_shoulder'] if 'left' in joint_name else JOINTS['right_shoulder']]
            elbow = landmarks[joint_index]
            wrist = landmarks[JOINTS['left_wrist'] if 'left' in joint_name else JOINTS['right_wrist']]
            joint_angles[joint_name] = calculate_angle([shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y])
        elif joint_name.endswith('shoulder'):
            hip = landmarks[JOINTS['left_hip'] if 'left' in joint_name else JOINTS['right_hip']]
            shoulder = landmarks[joint_index]
            elbow = landmarks[JOINTS['left_elbow'] if 'left' in joint_name else JOINTS['right_elbow']]
            joint_angles[joint_name] = calculate_angle([hip.x, hip.y], [shoulder.x, shoulder.y], [elbow.x, elbow.y])
        elif joint_name.endswith('hip'):
            shoulder = landmarks[JOINTS['left_shoulder'] if 'left' in joint_name else JOINTS['right_shoulder']]
            hip = landmarks[joint_index]
            knee = landmarks[JOINTS['left_knee'] if 'left' in joint_name else JOINTS['right_knee']]
            joint_angles[joint_name] = calculate_angle([shoulder.x, shoulder.y], [hip.x, hip.y], [knee.x, knee.y])
        elif joint_name.endswith('knee'):
            hip = landmarks[JOINTS['left_hip'] if 'left' in joint_name else JOINTS['right_hip']]
            knee = landmarks[joint_index]
            ankle = landmarks[JOINTS['left_ankle'] if 'left' in joint_name else JOINTS['right_ankle']]
            joint_angles[joint_name] = calculate_angle([hip.x, hip.y], [knee.x, knee.y], [ankle.x, ankle.y])

    return joint_angles

def process_image(image_path, output_image_path, output_csv='pose_angles_reference.csv'):
    if not os.path.exists(image_path):
        return None, None

    image = cv2.imread(image_path)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            joint_angles = extract_joint_angles(landmarks)

            # Draw landmarks on the image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save the annotated image
            cv2.imwrite(output_image_path, annotated_image)

            # Save joint angles to a CSV
            with open(output_csv, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                header = ['joint', 'angle']
                csv_writer.writerow(header)

                for joint, angle in joint_angles.items():
                    csv_writer.writerow([joint, angle])

            return joint_angles, output_image_path

    return None, None


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    image_path = os.path.join('uploads', file.filename)
    processed_image_path = os.path.join('static/uploads', 'processed_' + file.filename)
    file.save(image_path)

    # Process the uploaded image to get joint angles and the annotated image
    joint_angles, _ = process_image(image_path, processed_image_path)
    if joint_angles is None:
        return "No pose landmarks detected in the image."

    # Prepare the response to show joint angles
    response_message = "<h3>Joint Angles:</h3><ul>"
    for joint, angle in joint_angles.items():
        response_message += f"<li>{joint}: {angle:.2f} degrees</li>"
    response_message += "</ul>"

    # Display the processed image with landmarks drawn
    response_message += f'<img src="/static/uploads/{os.path.basename(processed_image_path)}" alt="Processed Image"/>'
    response_message += '<br><a href="/live_feed"><button>Compare with Live Video</button></a>'

    return response_message


@app.route('/live_feed')
def live_feed():
    return render_template('live_feed.html')

def resize_frame(frame, target_width):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (target_width, new_height))


def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam feed

    # Load reference joint angles from CSV
    reference_pose = {}
    with open('pose_angles_reference.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            joint, angle = row
            reference_pose[joint] = float(angle)

    threshold = 10  # Set the threshold for angle matching

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Resize the frame to make it bigger (change width and height as desired)
            frame = resize_frame(frame, 1280)  # Resize frame to 1280px wide

            # Process the frame for pose detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract joint angles for the live video
                live_joint_angles = extract_joint_angles(landmarks)

                # Overlay joint angles and compare with reference angles
                for joint, live_angle in live_joint_angles.items():
                    if joint in reference_pose:
                        ref_angle = reference_pose[joint]
                        angle_diff = abs(ref_angle - live_angle)

                        # Get the coordinates of the joint on the frame
                        joint_index = JOINTS[joint]
                        joint_coords = (int(landmarks[joint_index].x * frame.shape[1]), 
                                        int(landmarks[joint_index].y * frame.shape[0]))

                        # Check if the live angle is within the threshold of the reference angle
                        if angle_diff <= threshold:
                            match_text = "MATCH"
                            text_color = (0, 255, 0)  # Green for match
                        else:
                            match_text = "UNMATCH"
                            text_color = (0, 0, 255)  # Red for unmatch

                        # Display the live angle, angle difference, and match/unmatch text on the video feed
                        display_text = f"{int(live_angle)}deg ({int(angle_diff)}deg off) - {match_text}"
                        cv2.putText(frame, display_text, joint_coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to be served
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
