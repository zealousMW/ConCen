import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ffpyplayer.player import MediaPlayer

# ---------------- CONFIG ----------------
MODEL_PATH = "face_landmarker.task"
MONKEY_VIDEO = "parama padi da....✌️.mp4"  # Interruption video

MAX_ABSENCE_TIME = 3.0      # seconds before distraction
UNCERTAIN_GRACE_TIME = 7.0  # seconds of looking down before distraction
UNCERTAIN_TOTAL_CAP = 20.0  # total seconds of UNCERTAIN before forced DISTRACTED

EYE_CLOSED_EAR = 0.20       # Eye Aspect Ratio threshold
ROLL_THRESHOLD = 15         # degrees (head tilt tolerance)
YAW_THRESHOLD = 10          # proxy units (nose displacement)

# Resume Screen Settings
RESUME_SCREEN_THRESHOLD = 30.0   # seconds in DISTRACTED before resume screen
RESUME_GAZE_DURATION = 2.0       # seconds of focus required to resume

# Session Configuration
SESSION_GOAL_MINUTES = 25        # Pomodoro default
SESSION_GOAL_SECONDS = SESSION_GOAL_MINUTES * 60

# Task Types (Fixed List)
TASK_TYPES = [
    "Deep Work",
    "Learning/Study",
    "Writing",
    "Coding",
    "Reading",
    "Admin/Email",
    "Creative Work",
    "Meeting Prep",
    "Review/Editing"
]

# Session Lifecycle States
SESSION_ACTIVE = "ACTIVE"
SESSION_COMPLETED = "COMPLETED"
SESSION_ABANDONED = "ABANDONED"
# ---------------------------------------

# State Definitions
FOCUSED = "FOCUSED"
DISTRACTED = "DISTRACTED"
UNCERTAIN = "UNCERTAIN"

# Global Variables
state = FOCUSED
last_present_time = time.time()
uncertain_start_time = None
uncertain_total_time = 0.0  # Accumulated UNCERTAIN time
last_uncertain_update = None
current_pose = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "ear": 0.0}

# Focus Timer State (The Truth Machine)
timer_state = {
    "total_focused_seconds": 0.0,      # Lifetime focus time (only FOCUSED)
    "current_streak_seconds": 0.0,      # Current continuous focus duration
    "session_start_time": time.time(),  # Overall session start
    "last_update_time": None,           # Last timer update timestamp
}

# Session Lifecycle State
session_state = {
    "lifecycle": SESSION_ACTIVE,        # Current session status
    "goal_seconds": SESSION_GOAL_SECONDS,
    "is_goal_reached": False,
    "completion_time": None,
    "task_title": "",                  # User-entered task
    "task_type": "",                   # Selected category
    "task_keyword": "",                # First word for typing challenge
}

# Interruption state
interruption_active = False
interruption_cap = None

# Resume Screen State
resume_screen_active = False
resume_screen_trigger_time = None
resume_screen_type = None  # "distraction" | "quit_attempt"
distraction_start_time = None  # Track when DISTRACTED began

# Resume Challenge
resume_challenge = {
    "gaze_start_time": None,      # When user started looking
    "gaze_progress": 0.0,          # Current gaze duration
    "typed_text": "",              # User's typed input
    "challenge_type": "gaze"       # "gaze" | "typing"
}

# Temporal smoothing to prevent single-frame jitter
presence_window = deque(maxlen=5)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Timestamp tracking for frame ordering
latest_ts = 0

def format_time(seconds):
    """Convert seconds to MM:SS or HH:MM:SS format"""
    if seconds < 3600:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

def show_task_input_screen():
    """Capture task details before session starts. Returns (title, type, keyword) or None if cancelled."""
    # Create dummy VideoCapture for screen rendering
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera for task input")
        return None
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    h, w = frame.shape[:2]
    
    task_title = ""
    selected_type_idx = 0  # Default to first type
    
    while True:
        # Black background
        screen = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Header
        cv2.putText(screen, "NEW SESSION SETUP", (w//2 - 280, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 3)
        
        # Instructions
        cv2.putText(screen, "Enter your task details to begin:", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        
        # Task Title Input
        cv2.putText(screen, "Task Title:", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        title_display = task_title + "_" if len(task_title) < 50 else task_title
        cv2.putText(screen, title_display, (50, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Task Type Selection
        cv2.putText(screen, "Task Type (press number to select):", (50, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_pos = 320
        for i, task_type in enumerate(TASK_TYPES):
            if i == selected_type_idx:
                color = (0, 255, 255)  # Cyan for selected
                prefix = ">"
            else:
                color = (150, 150, 150)  # Gray for unselected
                prefix = " "
            
            cv2.putText(screen, f"{prefix} [{i+1}] {task_type}", (50, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 35
        
        # Bottom instructions
        cv2.line(screen, (50, h - 100), (w - 50, h - 100), (100, 100, 100), 2)
        cv2.putText(screen, "[ENTER] Start Session  |  [ESC] Cancel", (50, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.imshow("Attention Monitor", screen)
        
        key = cv2.waitKey(28) & 0xFF
        
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None
        
        elif key == 13:  # ENTER
            if task_title.strip():  # Only proceed if title is not empty
                # Extract first word as keyword
                words = task_title.strip().split()
                task_keyword = words[0].lower() if words else "focus"
                
                cap.release()
                return (task_title.strip(), TASK_TYPES[selected_type_idx], task_keyword)
        
        elif key >= ord('1') and key <= ord('9'):  # Number keys
            idx = key - ord('1')
            if idx < len(TASK_TYPES):
                selected_type_idx = idx
        
        elif key == 8:  # Backspace
            if task_title:
                task_title = task_title[:-1]
        
        elif 32 <= key <= 126:  # Printable characters
            if len(task_title) < 50:  # Limit title length
                task_title += chr(key)

def activate_resume_screen(trigger_type):
    """Activate the resume screen with specified trigger"""
    global resume_screen_active, resume_screen_trigger_time, resume_screen_type
    global resume_challenge, interruption_active, interruption_cap
    
    resume_screen_active = True
    resume_screen_trigger_time = time.time()
    resume_screen_type = trigger_type
    
    # Stop interruption video if playing
    if interruption_active:
        interruption_active = False
        if interruption_cap:
            interruption_cap.release()
            interruption_cap = None
    
    # Reset challenge state
    resume_challenge["gaze_start_time"] = None
    resume_challenge["gaze_progress"] = 0.0
    resume_challenge["typed_text"] = ""
    resume_challenge["challenge_type"] = "gaze"
    
    print(f"[RESUME SCREEN] Activated - Reason: {trigger_type}")

def deactivate_resume_screen(reason):
    """Deactivate the resume screen"""
    global resume_screen_active, distraction_start_time, state
    
    resume_screen_active = False
    distraction_start_time = None
    
    # Force state to FOCUSED if challenge completed
    if reason in ["gaze_complete", "typing_complete"]:
        state = FOCUSED
        timer_state["last_update_time"] = time.time()
        timer_state["current_streak_seconds"] = 0.0  # Start fresh streak
    
    print(f"[RESUME SCREEN] Deactivated - Reason: {reason}")

def render_completion_screen(frame):
    """Draw success screen when session goal is reached"""
    h, w = frame.shape[:2]
    
    # Green celebration background
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :] = (0, 40, 0)  # Dark green tint
    
    # Celebration header
    cv2.putText(overlay, "SESSION COMPLETE!", (w//2 - 350, 100),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 255, 0), 4)
    
    cv2.putText(overlay, "Congratulations!", (w//2 - 220, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    # Task completed
    cv2.putText(overlay, f"Task: {session_state['task_title']}", (100, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay, f"Type: {session_state['task_type']}", (100, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    
    # Stats Display
    actual_time = session_state["completion_time"] - timer_state["session_start_time"]
    efficiency = (timer_state["total_focused_seconds"] / actual_time * 100) if actual_time > 0 else 0
    
    y_pos = 300
    cv2.putText(overlay, f"Goal Time:       {format_time(session_state['goal_seconds'])}", (100, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    y_pos += 60
    cv2.putText(overlay, f"Focused Time:    {format_time(timer_state['total_focused_seconds'])}", (100, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    y_pos += 60
    cv2.putText(overlay, f"Real Time:       {format_time(actual_time)}", (100, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
    y_pos += 60
    cv2.putText(overlay, f"Efficiency:      {efficiency:.1f}%", (100, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
    
    # Actions
    y_pos = h - 100
    cv2.line(overlay, (50, y_pos - 20), (w - 50, y_pos - 20), (100, 255, 100), 2)
    cv2.putText(overlay, "[Q] Quit  |  Press ESC to exit", (100, y_pos + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return overlay

def render_resume_screen(frame):
    """Draw fullscreen resume screen overlay"""
    h, w = frame.shape[:2]
    
    # Black background
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Header
    cv2.putText(overlay, "RESUME SESSION?", (w//2 - 280, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 0), 3)
    
    # Task Info
    cv2.putText(overlay, f"Task: {session_state['task_title']}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(overlay, f"Type: {session_state['task_type']}", (50, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    
    # Stats Display
    now = time.time()
    session_elapsed = now - timer_state["session_start_time"]
    focus_pct = (timer_state["total_focused_seconds"] / session_elapsed * 100) if session_elapsed > 0 else 0
    
    y_pos = 210
    cv2.putText(overlay, f"Session Time:   {format_time(session_elapsed)}", (50, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    y_pos += 45
    cv2.putText(overlay, f"Focused Time:   {format_time(timer_state['total_focused_seconds'])}", (50, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_pos += 45
    cv2.putText(overlay, f"Focus Rate:     {focus_pct:.1f}%", (50, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    
    # Separator line
    cv2.line(overlay, (50, y_pos + 35), (w - 50, y_pos + 35), (100, 100, 100), 2)
    
    # Challenge Options
    y_pos += 75
    cv2.putText(overlay, "Choose one to resume:", (50, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Option 1: Gaze Challenge
    y_pos += 55
    option1_color = (0, 255, 255) if resume_challenge["challenge_type"] == "gaze" else (100, 100, 100)
    cv2.putText(overlay, "[1] Look at screen for 2 seconds", (50, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, option1_color, 2)
    
    # Progress bar for gaze
    if resume_challenge["challenge_type"] == "gaze":
        progress = min(resume_challenge["gaze_progress"] / RESUME_GAZE_DURATION, 1.0)
        bar_width = 400
        bar_filled = int(bar_width * progress)
        
        y_pos += 35
        cv2.rectangle(overlay, (50, y_pos), (50 + bar_width, y_pos + 20), (50, 50, 50), -1)
        cv2.rectangle(overlay, (50, y_pos), (50 + bar_filled, y_pos + 20), (0, 255, 0), -1)
        cv2.putText(overlay, f"{resume_challenge['gaze_progress']:.1f}s / {RESUME_GAZE_DURATION}s",
                    (460, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Option 2: Typing Challenge
    y_pos += 55
    option2_color = (0, 255, 255) if resume_challenge["challenge_type"] == "typing" else (100, 100, 100)
    cv2.putText(overlay, f"[2] Type task keyword: '{session_state['task_keyword']}' + ENTER", (50, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, option2_color, 2)
    
    if resume_challenge["challenge_type"] == "typing":
        y_pos += 35
        input_text = resume_challenge["typed_text"] + "_"
        cv2.putText(overlay, f"Input: {input_text}", (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Action Buttons at bottom
    y_pos = h - 80
    cv2.line(overlay, (50, y_pos - 20), (w - 50, y_pos - 20), (100, 100, 100), 2)
    cv2.putText(overlay, "[E] Abandon Session (will count as failure)", (50, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2)
    
    return overlay

def eye_aspect_ratio(landmarks, eye_ids):
    """Calculate Eye Aspect Ratio for blink/closed eye detection"""
    p = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_ids])
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)

def get_face_pose(landmarks):
    """Calculate Roll, Pitch, Yaw for head orientation"""
    # Calculate Roll (head tilt)
    left_eye_xy = np.array([landmarks[33].x, landmarks[33].y])
    right_eye_xy = np.array([landmarks[263].x, landmarks[263].y])
    
    dx = right_eye_xy[0] - left_eye_xy[0]
    dy = right_eye_xy[1] - left_eye_xy[1]
    roll = np.degrees(np.arctan2(dy, dx))

    # Calculate Yaw (head turn) - stable horizontal displacement method
    nose_x = landmarks[1].x
    eye_mid_x = (landmarks[33].x + landmarks[263].x) / 2.0
    yaw = (nose_x - eye_mid_x) * 100  # scaled proxy

    # Pitch for diagnostics
    top_z = landmarks[10].z
    chin_z = landmarks[152].z
    top_y = landmarks[10].y
    chin_y = landmarks[152].y
    height = chin_y - top_y
    pitch = np.degrees(np.arctan2(top_z - chin_z, height))

    return roll, pitch, yaw

def result_callback(result, output_image, timestamp_ms):
    """MediaPipe callback with timestamp tracking to prevent out-of-order frames"""
    global state, last_present_time, uncertain_start_time, current_pose, presence_window
    global uncertain_total_time, last_uncertain_update, latest_ts
    
    # Reject out-of-order frames to maintain causality
    if timestamp_ms <= latest_ts:
        return
    latest_ts = timestamp_ms
    
    now = time.time()

    if not result.face_landmarks:
        if now - last_present_time > MAX_ABSENCE_TIME:
            new_state = DISTRACTED
            if new_state != state:
                print(f"{state} → {new_state} @ {now:.1f}")
                state = new_state
        return

    landmarks = result.face_landmarks[0]

    # Eye Aspect Ratio detection
    ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    ear = (ear_left + ear_right) / 2.0

    # Get Roll, Pitch, Yaw
    roll, pitch, yaw = get_face_pose(landmarks)

    # Store for diagnostics (thread-safe)
    current_pose["roll"] = roll
    current_pose["pitch"] = pitch
    current_pose["yaw"] = yaw
    current_pose["ear"] = ear

    # Raw presence detection
    # Eyes open, face roughly centered (no extreme yaw/roll)
    raw_presence = (
        ear > EYE_CLOSED_EAR and 
        abs(yaw) < YAW_THRESHOLD and 
        abs(roll) < ROLL_THRESHOLD
    )

    # Temporal smoothing - require majority vote from last 5 frames
    presence_window.append(raw_presence)
    is_looking_forward = sum(presence_window) >= 3

    # Adaptive nose-below-eyes detection
    nose_y = landmarks[1].y
    eyes_y = (landmarks[33].y + landmarks[263].y) / 2.0
    face_height = landmarks[152].y - landmarks[10].y
    nose_below_eyes = nose_y > eyes_y + 0.15 * face_height

    is_looking_down = (
        (ear < EYE_CLOSED_EAR or nose_below_eyes) and
        abs(yaw) < YAW_THRESHOLD and
        abs(roll) < ROLL_THRESHOLD
    )

    # State machine with transition logging
    new_state = state

    if is_looking_forward:
        last_present_time = now
        uncertain_start_time = None
        last_uncertain_update = None
        uncertain_total_time = 0.0  # Reset total UNCERTAIN time on FOCUSED
        new_state = FOCUSED

    elif is_looking_down:
        if uncertain_start_time is None:
            uncertain_start_time = now
        
        # UNCERTAIN freezes absence decay to prevent harsh penalties
        last_present_time = now

        # Check both continuous and total UNCERTAIN time
        continuous_uncertain_time = now - uncertain_start_time
        
        # Accumulate UNCERTAIN time
        if state == UNCERTAIN and last_uncertain_update is not None:
            uncertain_total_time += now - last_uncertain_update
        last_uncertain_update = now

        if continuous_uncertain_time >= UNCERTAIN_GRACE_TIME:
            new_state = DISTRACTED
        elif uncertain_total_time >= UNCERTAIN_TOTAL_CAP:
            # Hard cap: too much total UNCERTAIN time = forced DISTRACTED
            new_state = DISTRACTED
        else:
            new_state = UNCERTAIN

    else:
        last_uncertain_update = None
        if now - last_present_time > MAX_ABSENCE_TIME:
            new_state = DISTRACTED

    # Resume Screen Gaze Challenge Tracking
    if resume_screen_active and resume_challenge["challenge_type"] == "gaze":
        if new_state == FOCUSED or (state == FOCUSED and is_looking_forward):
            if resume_challenge["gaze_start_time"] is None:
                resume_challenge["gaze_start_time"] = now
            else:
                resume_challenge["gaze_progress"] = now - resume_challenge["gaze_start_time"]
                
                # Challenge complete!
                if resume_challenge["gaze_progress"] >= RESUME_GAZE_DURATION:
                    deactivate_resume_screen("gaze_complete")
        else:
            # Lost focus, reset
            resume_challenge["gaze_start_time"] = None
            resume_challenge["gaze_progress"] = 0.0
    
    # Timer State Management
    if new_state != state:
        # Transitioning FROM FOCUSED - save accumulated time
        if state == FOCUSED and timer_state["last_update_time"] is not None:
            elapsed = now - timer_state["last_update_time"]
            timer_state["total_focused_seconds"] += elapsed
            timer_state["current_streak_seconds"] += elapsed
        
        # Transitioning TO FOCUSED - start/resume timer
        if new_state == FOCUSED:
            timer_state["last_update_time"] = now
            if state == DISTRACTED:
                # Reset streak on return from distraction
                timer_state["current_streak_seconds"] = 0.0
        
        # Transitioning TO DISTRACTED - finalize and reset streak
        if new_state == DISTRACTED:
            timer_state["current_streak_seconds"] = 0.0
            timer_state["last_update_time"] = None
        
        # Transitioning TO UNCERTAIN - freeze timer
        if new_state == UNCERTAIN:
            if state == FOCUSED and timer_state["last_update_time"] is not None:
                elapsed = now - timer_state["last_update_time"]
                timer_state["total_focused_seconds"] += elapsed
                timer_state["current_streak_seconds"] += elapsed
            timer_state["last_update_time"] = None
        
        print(f"{state} → {new_state} @ {now:.1f}")
        state = new_state

# MediaPipe Setup - recreate options with callback now that result_callback is defined
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    result_callback=result_callback,
)

def main():
    global interruption_active, interruption_cap, resume_screen_active
    global distraction_start_time, state
    
    # Show task input screen BEFORE starting session
    print('--- TASK SETUP ---')
    task_input = show_task_input_screen()
    
    if task_input is None:
        print('Session cancelled by user')
        return
    
    task_title, task_type, task_keyword = task_input
    session_state["task_title"] = task_title
    session_state["task_type"] = task_type
    session_state["task_keyword"] = task_keyword
    
    print(f'Task: {task_title}')
    print(f'Type: {task_type}')
    print(f'Keyword: {task_keyword}')
    
    video_cap = None
    audio_player = None
    
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            print('--- ATTENTION MONITOR STARTED ---')
            cap = cv2.VideoCapture(0)
            print('Camera opened:', cap.isOpened())
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Track distraction duration for resume screen trigger
                if state == DISTRACTED and not resume_screen_active:
                    if distraction_start_time is None:
                        distraction_start_time = time.time()
                    elif time.time() - distraction_start_time > RESUME_SCREEN_THRESHOLD:
                        activate_resume_screen("distraction")
                elif state != DISTRACTED:
                    distraction_start_time = None

                # If DISTRACTED and not already interrupting, start video + audio
                if state == DISTRACTED and not interruption_active:
                    interruption_active = True
                    video_cap = cv2.VideoCapture(MONKEY_VIDEO)
                    audio_player = MediaPlayer(MONKEY_VIDEO)
                    print(f"[INTERRUPTION] Starting @ {time.time():.1f}")
                
                # If FOCUSED/UNCERTAIN and interruption is active, stop it
                if state != DISTRACTED and interruption_active:
                    interruption_active = False
                    
                    if video_cap is not None:
                        video_cap.release()
                        video_cap = None
                    
                    if audio_player is not None:
                        audio_player.close_player()
                        audio_player = None
                    
                    print(f"[INTERRUPTION] Stopped @ {time.time():.1f}")
                
                # Display fullscreen monkey video with audio
                if interruption_active and video_cap is not None:
                    ret_video, video_frame = video_cap.read()
                    audio_frame, val = audio_player.get_frame()
                    
                    if ret_video:
                        # Get screen size and resize to fullscreen
                        h, w = frame.shape[:2]
                        video_frame_resized = cv2.resize(video_frame, (w, h))
                        
                        # Continue processing webcam in background for state detection
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )
                        frame_timestamp_ms = int(time.time() * 1000)
                        landmarker.detect_async(mp_image, frame_timestamp_ms)
                        
                        # Show video frame instead of webcam
                        frame = video_frame_resized
                        
                        # Audio sync happens automatically via get_frame()
                        if val != 'eof' and audio_frame is not None:
                            img, t = audio_frame
                    else:
                        # Loop video
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        audio_player.seek(0)
                else:
                    # Normal webcam processing
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    frame_timestamp_ms = int(time.time() * 1000)
                    landmarker.detect_async(mp_image, frame_timestamp_ms)

                # Update timer continuously while FOCUSED
                now = time.time()
                if state == FOCUSED and timer_state["last_update_time"] is not None:
                    delta = now - timer_state["last_update_time"]
                    timer_state["total_focused_seconds"] += delta
                    timer_state["current_streak_seconds"] += delta
                    timer_state["last_update_time"] = now
                    
                    # Check if goal reached
                    if (not session_state["is_goal_reached"] and 
                        timer_state["total_focused_seconds"] >= session_state["goal_seconds"]):
                        
                        session_state["is_goal_reached"] = True
                        session_state["completion_time"] = now
                        session_state["lifecycle"] = SESSION_COMPLETED
                        print(f"\n🎉 SESSION GOAL REACHED! 🎉")
                
                # Completion Screen Handling
                if session_state["lifecycle"] == SESSION_COMPLETED:
                    frame = render_completion_screen(frame)
                    cv2.imshow("Attention Monitor", frame)
                    
                    key = cv2.waitKey(28) & 0xFF
                    if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                        break
                    
                    continue  # Skip normal rendering
                
                # Resume Screen Handling
                if resume_screen_active:
                    frame = render_resume_screen(frame)
                    cv2.imshow("Attention Monitor", frame)
                    
                    key = cv2.waitKey(28) & 0xFF
                    
                    if key == ord('1'):
                        resume_challenge["challenge_type"] = "gaze"
                        resume_challenge["typed_text"] = ""
                    elif key == ord('2'):
                        resume_challenge["challenge_type"] = "typing"
                        resume_challenge["gaze_start_time"] = None
                        resume_challenge["gaze_progress"] = 0.0
                    elif key == ord('e') or key == ord('E'):
                        # Abandon session
                        session_state["lifecycle"] = SESSION_ABANDONED
                        break
                    elif resume_challenge["challenge_type"] == "typing":
                        if key == 13:  # Enter
                            if resume_challenge["typed_text"].lower() == session_state["task_keyword"].lower():
                                deactivate_resume_screen("typing_complete")
                            else:
                                resume_challenge["typed_text"] = ""  # Wrong keyword, reset
                        elif key == 8:  # Backspace
                            if resume_challenge["typed_text"]:
                                resume_challenge["typed_text"] = resume_challenge["typed_text"][:-1]
                        elif 32 <= key <= 126:  # Printable characters
                            resume_challenge["typed_text"] += chr(key)
                    
                    continue  # Skip normal rendering
                
                # Show debug overlay (only when not interrupting)
                if not interruption_active:
                    if state == FOCUSED:
                        color = (0, 255, 0)  # Green
                    elif state == UNCERTAIN:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                    
                    # State indicator
                    cv2.putText(frame, state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    # Task type badge (top-right)
                    h, w = frame.shape[:2]
                    badge_text = f"{session_state['task_type']}"
                    cv2.putText(frame, badge_text, (w - 220, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
                    
                    # Pose debug info
                    cv2.putText(
                        frame,
                        f"Roll: {current_pose['roll']:.1f} | Yaw: {current_pose['yaw']:.1f} | EAR: {current_pose['ear']:.2f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    # Focus Timer Display (The Truth Machine)
                    session_elapsed = now - timer_state["session_start_time"]
                    
                    cv2.putText(
                        frame,
                        f"SESSION: {format_time(session_elapsed)}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 180, 180),
                        2
                    )
                    
                    cv2.putText(
                        frame,
                        f"FOCUSED: {format_time(timer_state['total_focused_seconds'])}",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.putText(
                        frame,
                        f"STREAK:  {format_time(timer_state['current_streak_seconds'])}",
                        (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 165, 0),
                        2
                    )
                    
                    # Goal Progress Display
                    goal_progress = min(timer_state["total_focused_seconds"] / session_state["goal_seconds"], 1.0)
                    goal_remaining = max(session_state["goal_seconds"] - timer_state["total_focused_seconds"], 0)
                    
                    # Goal info
                    cv2.putText(
                        frame,
                        f"GOAL:    {format_time(session_state['goal_seconds'])}",
                        (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (100, 200, 255),
                        2
                    )
                    
                    cv2.putText(
                        frame,
                        f"REMAIN:  {format_time(goal_remaining)}",
                        (20, 265),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 100),
                        2
                    )
                    
                    # Progress bar
                    bar_width = 300
                    bar_filled = int(bar_width * goal_progress)
                    bar_y = 280
                    
                    cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + 20), (50, 50, 50), -1)
                    cv2.rectangle(frame, (20, bar_y), (20 + bar_filled, bar_y + 20), (0, 255, 0), -1)
                    
                    cv2.putText(
                        frame,
                        f"{goal_progress * 100:.1f}%",
                        (330, bar_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                cv2.imshow("Attention Monitor", frame)

                key = cv2.waitKey(28) & 0xFF
                if key == 27:  # ESC
                    # If session has started, show resume screen instead of exiting
                    if timer_state["total_focused_seconds"] > 0 and not resume_screen_active:
                        activate_resume_screen("quit_attempt")
                    else:
                        # No focus time yet, allow immediate exit
                        break

        # Cleanup
        if video_cap is not None:
            video_cap.release()
        if audio_player is not None:
            audio_player.close_player()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print Final Session Summary
        final_session = time.time() - timer_state["session_start_time"]
        print("\n" + "="*50)
        
        if session_state["lifecycle"] == SESSION_COMPLETED:
            print("🎉 SESSION COMPLETED SUCCESSFULLY! 🎉")
        elif session_state["lifecycle"] == SESSION_ABANDONED:
            print("❌ SESSION ABANDONED")
            completion_pct = (timer_state["total_focused_seconds"] / session_state["goal_seconds"] * 100)
            print(f"Goal Progress: {completion_pct:.1f}%")
        else:
            print("SESSION SUMMARY - THE TRUTH")
        
        print("="*50)
        print(f"Task: {session_state['task_title']}")
        print(f"Type: {session_state['task_type']}")
        print("="*50)
        print(f"Goal Target:           {format_time(session_state['goal_seconds'])}")
        print(f"Total Session Time:    {format_time(final_session)}")
        print(f"Actual Focused Time:   {format_time(timer_state['total_focused_seconds'])}")
        print(f"Focus Percentage:      {(timer_state['total_focused_seconds'] / final_session * 100):.1f}%")
        
        if session_state["lifecycle"] == SESSION_COMPLETED:
            efficiency = (timer_state["total_focused_seconds"] / final_session * 100)
            print(f"Session Efficiency:    {efficiency:.1f}%")
        
        print("="*50)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('Error occurred:', e)
        raise

if __name__ == "__main__":
    main()