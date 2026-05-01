import cv2
import mediapipe as mp
from groq import Groq
import os
from dotenv import load_dotenv
import threading
import time

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
    print("⚠️ GROQ_API_KEY not found. Live AI feedback will be disabled.")
    last_response = "⚠️ Groq feedback disabled. Add GROQ_API_KEY to enable live AI feedback."

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

latest_landmarks = None
processing = False
lock = threading.Lock()
last_response = ""
last_processing_time = 0

# Landmark names mapping
LANDMARK_NAMES = {
    mp_pose.PoseLandmark.NOSE: "Nose",
    mp_pose.PoseLandmark.LEFT_SHOULDER: "Left Shoulder",
    mp_pose.PoseLandmark.RIGHT_SHOULDER: "Right Shoulder",
    mp_pose.PoseLandmark.LEFT_ELBOW: "Left Elbow",
    mp_pose.PoseLandmark.RIGHT_ELBOW: "Right Elbow",
    mp_pose.PoseLandmark.LEFT_WRIST: "Left Wrist",
    mp_pose.PoseLandmark.RIGHT_WRIST: "Right Wrist",
    mp_pose.PoseLandmark.LEFT_HIP: "Left Hip",
    mp_pose.PoseLandmark.RIGHT_HIP: "Right Hip",
    mp_pose.PoseLandmark.LEFT_KNEE: "Left Knee",
    mp_pose.PoseLandmark.RIGHT_KNEE: "Right Knee",
    mp_pose.PoseLandmark.LEFT_ANKLE: "Left Ankle",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "Right Ankle",
}

def send_to_groq(landmarks_dict):
    """Send pose landmarks to Groq and get response."""
    global processing, last_response
    
    # Format landmarks for prompt
    landmarks_text = "\n".join([f"{name}: x={data['x']:.3f}, y={data['y']:.3f}" 
                                for name, data in landmarks_dict.items()])

    if client is None:
        last_response = "⚠️ Groq feedback disabled. Add GROQ_API_KEY to enable live AI feedback."
        processing = False
        return
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI fitness coach analyzing body poses. Provide brief, actionable feedback in 2-3 sentences."
                },
                {
                    "role": "user",
                    "content": f"Analyze this human pose:\n{landmarks_text}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=200
        )
        
        response = chat_completion.choices[0].message.content
        last_response = response
        print("\n[AI FEEDBACK]:", response)
        
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        last_response = "⚠️ Groq feedback failed. Please check your API key and network connection."
    
    processing = False

def llm_worker():
    """Process latest frame when ready (rate limited to avoid API spam)."""
    global latest_landmarks, processing, last_processing_time
    
    while True:
        current_time = time.time()
        
        # Rate limit: process every 5 seconds
        if latest_landmarks and not processing and (current_time - last_processing_time > 5):
            with lock:
                processing = True
                landmarks_dict = latest_landmarks.copy()
                last_processing_time = current_time
            
            print("\n[INFO] Analyzing pose...")
            send_to_groq(landmarks_dict)
        
        time.sleep(0.5)

# Start background thread only if Groq is available
if client is not None:
    threading.Thread(target=llm_worker, daemon=True).start()
else:
    print("⚠️ Live AI feedback thread will not start because Groq is unavailable.")

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Extract landmarks
        landmarks_dict = {
            LANDMARK_NAMES[landmark]: {
                "x": round(results.pose_landmarks.landmark[landmark].x, 4),
                "y": round(results.pose_landmarks.landmark[landmark].y, 4)
            }
            for landmark in LANDMARK_NAMES.keys() 
            if results.pose_landmarks.landmark[landmark].visibility > 0.5
        }

        with lock:
            latest_landmarks = landmarks_dict
    
    # Display last AI response on frame
    if last_response:
        y0, dy = 30, 25
        for i, line in enumerate(last_response.split('\n')[:3]):  # Max 3 lines
            y = y0 + i * dy
            cv2.putText(frame, line[:50], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow("Pose Detection - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
