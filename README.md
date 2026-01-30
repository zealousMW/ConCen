# ConCen - Concentration Monitor

A real-time attention and focus monitoring application using computer vision and face detection. ConCen helps you maintain concentration during work sessions by detecting when you're distracted and providing gentle interruptions to bring you back on task.

## 📋 Description

ConCen (Concentration Center) is an AI-powered focus assistant that uses your webcam to monitor your attention during work sessions. It employs MediaPipe's face landmark detection to track:

- **Eye movements** - Detects when you're looking away from the screen
- **Head orientation** - Monitors head roll, pitch, and yaw to determine focus
- **Eye closure** - Tracks if your eyes are closed (drowsiness/fatigue)
- **Presence detection** - Identifies when you leave your workstation

The application provides:
- Real-time focus tracking with visual feedback
- Distraction interruptions (video/audio alerts when focus is lost)
- Session management with Pomodoro-style goal setting (default: 25 minutes)
- Task tracking with customizable categories
- Focus statistics and performance metrics
- Resume challenges to regain focus after distractions

## ✨ Features

### Core Features
- **Real-time Face Detection**: Uses MediaPipe Face Landmarker for accurate face tracking
- **Smart State Management**: Three attention states:
  - `FOCUSED`: Actively looking at screen with good posture
  - `UNCERTAIN`: Slightly distracted (looking down, minor head tilt)
  - `DISTRACTED`: Completely off-task (absent, eyes closed, looking away)

### Session Management
- Task title and categorization (Deep Work, Learning, Coding, etc.)
- Configurable session goals (default: 25-minute Pomodoro sessions)
- Real-time focus timer tracking only actual focused time
- Session statistics including focus percentage

### Distraction Management
- Automatic video/audio interruptions when distracted
- Configurable grace periods before triggering distractions
- Resume screen after prolonged distraction (30 seconds)
- Two resume challenge options:
  1. Gaze challenge (look at screen for 2 seconds)
  2. Typing challenge (type task keyword)

### Visual Feedback
- Live webcam feed with status overlay
- Color-coded state indicators:
  - Green border: FOCUSED
  - Yellow/Orange border: UNCERTAIN
  - Red border: DISTRACTED
- Real-time metrics display
- Session progress tracking

## 🔧 Requirements

### System Requirements
- Python 3.7 or higher
- Webcam (for face detection)
- Operating System: Linux, macOS, or Windows

### Python Dependencies
- opencv-python (cv2)
- mediapipe
- numpy
- ffpyplayer
- collections (standard library)
- time (standard library)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/zealousMW/ConCen.git
cd ConCen
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python mediapipe numpy ffpyplayer
```

### 3. Required Files
The repository includes:
- `main.py` - Main application file
- `face_landmarker.task` - MediaPipe face landmark model (required)
- Sample video file for interruptions

## 🚀 Usage

### Starting a Session

1. **Run the application**:
```bash
python main.py
```

2. **Task Setup Screen**:
   - Enter your task title (e.g., "Write project proposal")
   - Select task type using number keys (1-9):
     1. Deep Work
     2. Learning/Study
     3. Writing
     4. Coding
     5. Reading
     6. Admin/Email
     7. Creative Work
     8. Meeting Prep
     9. Review/Editing
   - Press `ENTER` to start the session
   - Press `ESC` to cancel

3. **During the Session**:
   - Stay focused on your screen
   - The application monitors your attention in real-time
   - Visual feedback shows your current state
   - Timer tracks only actual focused time

4. **Keyboard Controls**:
   - `Q`: Quit the session (triggers resume screen)
   - `E`: Abandon session (when on resume screen)
   - `1`: Select gaze challenge (on resume screen)
   - `2`: Select typing challenge (on resume screen)
   - `ESC`: Force quit (bypasses resume screen)

### Resume Challenges

When you become distracted for an extended period (30 seconds), the resume screen appears:

**Option 1 - Gaze Challenge**:
- Press `1` to select
- Look at the screen continuously for 2 seconds
- Progress bar shows your advancement

**Option 2 - Typing Challenge**:
- Press `2` to select
- Type the first word of your task
- Press `ENTER` to submit

## ⚙️ Configuration

Edit the configuration section at the top of `main.py`:

```python
# Timing Thresholds
MAX_ABSENCE_TIME = 3.0          # Seconds before marking as distracted
UNCERTAIN_GRACE_TIME = 7.0      # Seconds of uncertain state before distraction
UNCERTAIN_TOTAL_CAP = 20.0      # Total uncertain time before forced distraction

# Detection Thresholds
EYE_CLOSED_EAR = 0.20           # Eye Aspect Ratio (lower = more sensitive)
ROLL_THRESHOLD = 15             # Head tilt tolerance (degrees)
YAW_THRESHOLD = 10              # Head turn tolerance (units)

# Resume Screen
RESUME_SCREEN_THRESHOLD = 30.0  # Seconds in distracted state before resume screen
RESUME_GAZE_DURATION = 2.0      # Seconds of focus to pass gaze challenge

# Session Settings
SESSION_GOAL_MINUTES = 25       # Default session length (Pomodoro)
```

### Customization Options

**Add Custom Task Types**:
Edit the `TASK_TYPES` list in `main.py`:
```python
TASK_TYPES = [
    "Deep Work",
    "Learning/Study",
    # Add your custom types here
    "My Custom Task"
]
```

**Change Interruption Video**:
Replace the video file or update the path:
```python
MONKEY_VIDEO = "your_video_file.mp4"
```

## 📊 Understanding the Metrics

### On-Screen Display
- **State**: Current attention state (FOCUSED/UNCERTAIN/DISTRACTED)
- **Focus Timer**: Total time spent in FOCUSED state
- **Current Streak**: Continuous focused duration
- **Session Time**: Total elapsed time
- **Goal**: Target session duration
- **Task**: Your current task title and type

### Session Statistics
At the end of a session:
- Total focused time
- Total session duration
- Focus percentage
- Session outcome (COMPLETED/ABANDONED)

### Pose Metrics (Debug Info)
- **Roll**: Head tilt angle (degrees)
- **Yaw**: Head turn displacement
- **EAR**: Eye Aspect Ratio (blink/closure detection)

## 🐛 Troubleshooting

### Camera Not Opening
- Check that your webcam is connected and not in use by another application
- Try a different camera index if you have multiple cameras:
  ```python
  cap = cv2.VideoCapture(1)  # Change 0 to 1, 2, etc.
  ```

### Model File Not Found
- Ensure `face_landmarker.task` is in the same directory as `main.py`
- Download from MediaPipe if missing

### Performance Issues
- Close other applications using the webcam
- Reduce video resolution or frame rate
- Ensure adequate lighting for better face detection

### Detection Too Sensitive/Insensitive
- Adjust threshold values in the configuration section
- Fine-tune based on your working position and lighting

### Video/Audio Interruption Not Working
- Ensure the interruption video file exists
- Check ffpyplayer installation
- Verify video codec compatibility

## 🎯 Best Practices

1. **Positioning**: Sit directly facing the camera at a comfortable distance
2. **Lighting**: Ensure good lighting on your face for accurate detection
3. **Calibration**: Adjust thresholds based on your natural working posture
4. **Breaks**: Use the 25-minute Pomodoro default and take regular breaks
5. **Environment**: Minimize background movement and distractions

## 📝 Task Categories

Choose from these pre-defined task types:
- **Deep Work**: Intensive, focused work requiring maximum concentration
- **Learning/Study**: Educational activities, courses, tutorials
- **Writing**: Documents, articles, creative writing
- **Coding**: Programming, debugging, development
- **Reading**: Documentation, articles, books
- **Admin/Email**: Administrative tasks, email management
- **Creative Work**: Design, brainstorming, creative tasks
- **Meeting Prep**: Preparation for meetings or presentations
- **Review/Editing**: Proofreading, code review, editing

## 🔒 Privacy

ConCen processes all video locally on your machine. No video data is transmitted or stored externally. The application only uses your webcam for real-time face landmark detection.

## 📄 License

This project is provided as-is for personal use. Please respect the licenses of the underlying libraries (OpenCV, MediaPipe, etc.).

## 🙏 Credits

Built using:
- [MediaPipe](https://google.github.io/mediapipe/) - Google's face landmark detection
- [OpenCV](https://opencv.org/) - Computer vision library
- [FFPyPlayer](https://github.com/matham/ffpyplayer) - Media player for Python

---

**Note**: This tool is designed to help improve focus and productivity. Use it as a supportive aid, not a replacement for good time management habits and healthy work practices.
