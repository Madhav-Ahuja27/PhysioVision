import argparse
import cv2
import mediapipe as mp
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
client = None
if groq_api_key and groq_api_key.strip():
    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        print(f"⚠️ Groq client initialization failed: {e}")
        client = None
else:
    print("⚠️ GROQ_API_KEY not found. Groq feedback will be disabled.")

# Parse image path option
parser = argparse.ArgumentParser(description="Run pose inference on a local image file.")
parser.add_argument("--image", "-i", default="pose3.jpg", help="Path to the local image file")
args = parser.parse_args()

# Load the image
image_path = args.image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Calculate joint angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Process image with Pose Model
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract keypoints
        keypoints = {
            "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
            "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
            "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
            "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
            "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
            "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
        }

        coords = {name: [landmarks[val.value].x, landmarks[val.value].y] for name, val in keypoints.items()}

        # Calculate joint angles
        angles = {
            "Left Elbow": calculate_angle(coords["left_shoulder"], coords["left_elbow"], coords["left_wrist"]),
            "Right Elbow": calculate_angle(coords["right_shoulder"], coords["right_elbow"], coords["right_wrist"]),
            "Left Shoulder": calculate_angle(coords["right_shoulder"], coords["left_shoulder"], coords["left_elbow"]),
            "Right Shoulder": calculate_angle(coords["right_elbow"], coords["right_shoulder"], coords["left_shoulder"]),
            "Left Knee": calculate_angle(coords["left_hip"], coords["left_knee"], coords["left_ankle"]),
            "Right Knee": calculate_angle(coords["right_hip"], coords["right_knee"], coords["right_ankle"]),
            "Left Hip": calculate_angle(coords["left_shoulder"], coords["left_hip"], coords["left_knee"]),
            "Right Hip": calculate_angle(coords["right_shoulder"], coords["right_hip"], coords["right_knee"]),
        }

        # Print detected angles
        for name, angle in angles.items():
            print(f"{name} Angle: {angle:.2f}°")

        # Define the desired pose
        desired_pose = {
            "Left Knee": 90,
            "Right Knee": 90,
            "Hip 90°": 90,
            "Hip 135-180°": (135, 180),
        }

        # Identify hip angles
        detected_hip_angles = [angles["Left Hip"], angles["Right Hip"]]
        hip_90 = None
        hip_135_180 = None

        for hip_angle in detected_hip_angles:
            if 88 <= hip_angle <= 92:
                hip_90 = hip_angle
            elif 135 <= hip_angle <= 180:
                hip_135_180 = hip_angle

        # Generate feedback
        if angles["Left Knee"] == 90 and angles["Right Knee"] == 90 and hip_90 and hip_135_180:
            chatbot_feedback = "The person is performing the desired pose correctly!"
        else:
            chatbot_feedback = "The person is not in the correct pose. Detected angles:\n"
            for key, angle in angles.items():
                chatbot_feedback += f"- {key}: {int(angle)}°\n"

        if client is not None:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an experienced physiotherapist with over 10 years of practice. "
                                "Provide precise, direct corrective feedback. "
                                "Focus on actionable adjustments based on detected angles. "
                                "Keep response under 100 words."
                            )
                        },
                        {
                            "role": "user",
                            "content": chatbot_feedback
                        }
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    max_tokens=500
                )
                
                response_text = chat_completion.choices[0].message.content
                print("\n🔹 AI Physiotherapist Feedback:\n")
                print(response_text)
                
            except Exception as e:
                print(f"❌ Groq API Error: {e}")
                print("\n⚠️ Falling back to local pose evaluation:\n")
                print(chatbot_feedback)
        else:
            print("\n⚠️ Groq unavailable. Falling back to local pose evaluation:\n")
            print(chatbot_feedback)

        # Track used positions
        used_positions = []

        # Draw angles
        for (name, angle), (point, color) in zip(angles.items(), [
            (coords["left_elbow"], (0, 255, 0)),
            (coords["right_elbow"], (0, 255, 0)),
            (coords["left_shoulder"], (255, 0, 0)),
            (coords["right_shoulder"], (255, 0, 0)),
            (coords["left_knee"], (255, 255, 0)),
            (coords["right_knee"], (255, 255, 0)),
            (coords["left_hip"], (0, 0, 255)),
            (coords["right_hip"], (0, 0, 255)),
        ]):
            position = np.multiply(point, [image.shape[1], image.shape[0]]).astype(int)

            # Avoid collisions
            shift_y = 0
            for used_pos in used_positions:
                if abs(used_pos[0] - position[0]) < 20 and abs(used_pos[1] - position[1]) < 20:
                    shift_y += 15

            position = (position[0], position[1] + shift_y)
            used_positions.append(position)

            cv2.putText(image, f"{int(angle)}°", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Display
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
