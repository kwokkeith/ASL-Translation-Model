from roboflow import Roboflow
import supervision as sv
import cv2

cap = cv2.VideoCapture(0)

rf = Roboflow(api_key="3njVsAWJotNGOdz9GK1B")
project = rf.workspace().project("american-sign-language-letters")
model = project.version(6).model

label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert frame to RGB (MediaPipe requires RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = model.predict(image, confidence=40, overlap=30).json()

    labels = [item["class"] for item in result["predictions"]]

    detections = sv.Detections.from_inference(result)

    # image = cv2.imread(frame)

    annotated_image = box_annotator.annotate(frame, detections=detections)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections, labels=labels)

    # sv.plot_image(image=annotated_image, size=(4, 4))

    # Display the output
    cv2.imshow("Hand Tracking", annotated_image)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()