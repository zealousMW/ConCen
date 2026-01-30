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

# Interruption state
interruption_active = False
interruption_cap = None

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
    global interruption_active, interruption_cap
    
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

                cv2.imshow("Attention Monitor", frame)

                if cv2.waitKey(28) & 0xFF == 27:  # ESC to exit, ~28ms = 30fps
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
        print("SESSION SUMMARY - THE TRUTH")
        print("="*50)
        print(f"Total Session Time:    {format_time(final_session)}")
        print(f"Actual Focused Time:   {format_time(timer_state['total_focused_seconds'])}")
        print(f"Focus Percentage:      {(timer_state['total_focused_seconds'] / final_session * 100):.1f}%")
        print("="*50)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('Error occurred:', e)
        raise

if __name__ == "__main__":
    main()