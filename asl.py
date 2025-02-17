import mediapipe as mp
import cv2
import numpy as np
import glob

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# img_files = glob.glob('asl_data/train/*.jpg')

# Load image
img = cv2.imread('asl_data/train/J19_jpg.rf.e48b6808342f54aa56d3d776cf0065bf.jpg')

# Check if image was loaded successfully
if img is None:
    print("Error: Unable to load image.")
    exit()

# Convert BGR to RGB for MediaPipe
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width, _ = img.shape

# Process the image with MediaPipe
results = hands.process(img_rgb)
points = {}

# Extract landmark points
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        print(hand_landmarks)
        for i, landmark in enumerate(hand_landmarks.landmark):
            points[i] = (int(landmark.x * width), int(landmark.y * height))   # to get point coordinates

# Create blank white image
output_img = np.ones((height, width, 3), dtype=np.uint8) * 255

# Draw hand landmarks and connections
if len(points) == 21:  # Ensure a full hand is detected
    for hc in mp_hands.HAND_CONNECTIONS:
        cv2.line(img, points[hc[0]], points[hc[1]], (0, 0, 255), 4)
        cv2.line(output_img, points[hc[0]], points[hc[1]], (0, 0, 255), 4)


# Display original image with landmarks
for point in points.values():
    cv2.circle(img, point, 5, (255, 0, 0), -1)
    cv2.circle(output_img, point, 5, (255, 0, 0), -1)

# Show the images
cv2.imshow('Original Image', img)
cv2.imshow('Hand Tracking', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()