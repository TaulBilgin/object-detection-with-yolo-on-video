import torch
from ultralytics import YOLO
import cv2
import time

def rescale_frame(frame, scale):   
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

model = YOLO('C:\\Users\\bilgi\\OneDrive\\Masaüstü\\code\\AI\go br\\AI v3\\weights\\best.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
modell = model.to(device)

cap = cv2.VideoCapture(0)

fps_start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read() # Capture a frame from the video stream
    if not ret:
        break

    results = modell(frame, conf=0.5)
    result = results[0]
    boxs = result.boxes

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    total = 0

    for box in boxs:
        if box.cls == 0:
            print("100_tl")
            text = "100_tl"
            value = 100
        elif box.cls == 1:
            print("10_tl")
            text = "10_tl"
            value = 10
        elif box.cls == 2:
            print("200_tl")
            text = "200_tl"
            value = 200
        elif box.cls == 3:
            print("20_tl")
            text = "20_tl"
            value = 20
        elif box.cls == 4:
            print("50_tl")
            text = "50_tl"
            value = 50
        elif box.cls == 5:
            print("5_tl")
            text = "5_tl"
            value = 5
        total = total + value
        xmin, ymin, xmax, ymax = box.xyxy[0]
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        org = (int(xmin), int(ymin))

        frame = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        print("-----------------------------------------")

    frame = rescale_frame(frame, 1)

    fps_end_time = time.time()
    fps_diff_time = fps_end_time - fps_start_time
    fps = 1 / fps_diff_time
    fps_start_time = fps_end_time
    fps_text = "FPS: {:.2f}".format(fps)
    frame = cv2.putText(frame, fps_text, (5, 30), font, fontScale, color, thickness, cv2.LINE_AA)

    print(total)

    # Display the image
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()