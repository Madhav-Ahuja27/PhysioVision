# AI Fitness Pose Analyzer

Computer Vision capstone project using MediaPipe pose estimation with AI-powered feedback via Groq API.

## Overview

Real-time pose detection system that:
- Detects body pose using MediaPipe
- Calculates joint angles
- Provides AI-powered corrective feedback
- Works on static images or webcam

## Features

- **Pose Detection**: 33 body landmarks tracked in real-time
- **Angle Calculation**: Automatic joint angle measurement
- **AI Feedback**: Groq LLM provides physiotherapy-style corrections
- **Visual Output**: Annotated images/video with angle overlays
- **No Local Dependencies**: Uses free Groq API (no LM Studio needed)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**NOTE BY ME -> GROQ IS JUST FOR THE TEXT FEEDBACK, ACTUAL LANDMARKS AND DETECTION IS DONE BY MEDIAPIPE
**
### 2. Get Groq API Key (Free)
1. Visit https://console.groq.com
2. Sign up for free account
3. Create API key
4. Copy `.env.example` to `.env`
5. Add your key:
```bash
GROQ_API_KEY=your_actual_key_here
```

## Usage

### Static Image Analysis
```bash
python inference.py
```
Analyzes `pose3.jpg` by default.

To use a different local image:
```bash
python inference.py --image pose4.jpg
```

### Webcam Real-time Analysis
```bash
python inference_webcam.py
```
Press 'q' to quit.

## ## Sample Images

Test images included:
- `pose2.jpg` - `pose6.jpg`: Various exercise poses

## How It Works

1. **Pose Detection**: MediaPipe extracts 33 body landmarks
2. **Angle Calculation**: Computes joint angles (knees, hips, elbows, shoulders)
3. **Analysis**: Compares detected angles to desired pose
4. **AI Feedback**: Groq API generates corrective instructions
5. **Visualization**: Displays annotated image with angles and feedback

## Technical Details

- **Framework**: OpenCV + MediaPipe
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Language**: Python 3.8+
- **License**: Educational use

## Project Structure

```
├── inference.py              # Main static image script
├── inference_webcam.py       # Webcam real-time script
├── requirements.txt          # Dependencies
├── .env.example              # API key template
├── pose2.jpg                 # Sample pose image
├── pose3.jpg                 # Sample pose image
├── pose4.jpg                 # Sample pose image
├── pose5.jpg                 # Sample pose image
├── pose6.jpg                 # Sample pose image
└── README.md                 # This file
```

## API Rate Limits

Groq free tier:
- 30 requests/minute
- 14,400 requests/day

Webcam script rate-limited to 1 request/5 seconds.

## Troubleshooting

**"No API key found"**
- Check `.env` file exists and contains `GROQ_API_KEY`

**"Webcam not opening"**
- Check camera permissions
- Try different camera index: `cv2.VideoCapture(1)`

**"Module not found"**
- Run: `pip install -r requirements.txt`

## Future Improvements

- [ ] Custom pose definitions
- [ ] Exercise rep counter
- [ ] Form score calculation
- [ ] Multiple person tracking
- [ ] Mobile app version

## Credits

Built for Computer Vision & Deep Learning capstone project.
