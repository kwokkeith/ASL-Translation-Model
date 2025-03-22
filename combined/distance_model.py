import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.fft import fft, fftfreq

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Store last 60 frames of hand landmark positions
prev_hand_positions = {}
motion_history = {}
# frame_window = 30  # Number of frames to analyze FFT
fps = 15  # Assume 15 FPS for FFT calculation

def detect_motion_status(frame, frame_window):
    """Detects if the motion is Static or Dynamic."""
    global prev_hand_positions, motion_history
    
    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Track hand movements
    current_hand_positions = {}
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Extract (x, y) positions of hand landmarks
            hand_positions = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            
            # Determine Left or Right Hand
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            hand_id = f"{hand_label} Hand"  # Unique ID per hand
            
            # Store current positions
            current_hand_positions[hand_id] = hand_positions
            
            # Initialize motion history if not exists
            if hand_id not in motion_history:
                motion_history[hand_id] = []
            
            # Compute movement magnitude if we have a previous position
            if hand_id in prev_hand_positions:
                prev_positions = prev_hand_positions[hand_id]
                movement = np.linalg.norm(hand_positions - prev_positions, axis=1)
                hand_movement_magnitude = np.mean(movement)
                
                # Store movement data for FFT analysis
                motion_history[hand_id].append(hand_movement_magnitude)
                
                # Keep only the last 60 frames
                if len(motion_history[hand_id]) > frame_window:
                    motion_history[hand_id].pop(0)
                
                # Apply FFT every 60 frames
                if len(motion_history[hand_id]) == frame_window:
                    movement_series = np.array(motion_history[hand_id])
                    fft_values = np.abs(fft(movement_series))[:frame_window // 2]
                    freqs = fftfreq(frame_window, 1 / fps)[:frame_window // 2]
                    
                    # Compute mean frequency
                    mean_freq = np.sum(freqs * fft_values) / np.sum(fft_values)
                    
                    # Classify motion
                    threshold = 1  # Tune based on testing
                    motion_status = "Dynamic" if mean_freq > threshold else "Static"
                    return motion_status
    
    # Update previous hand positions
    prev_hand_positions = current_hand_positions.copy()
    
    return "Static"  # Default to static if no hand detected