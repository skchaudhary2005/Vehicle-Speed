import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("traffic.mp4")

line_A = 250
line_B = 350

entry_time = {}
prev_y = {}
counted = set()   # ✅ already processed vehicles

distance_m = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in [2,3,5,7] and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, 'vehicle'))

    tracks = tracker.update_tracks(detections, frame=frame)

    cv2.line(frame, (0, line_A), (640, line_A), (255, 0, 0), 2)
    cv2.line(frame, (0, line_B), (640, line_B), (0, 0, 255), 2)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        cx = int(l + w/2)
        cy = int(t + h/2)

        if track_id not in prev_y:
            prev_y[track_id] = cy

        cv2.rectangle(frame, (l, t), (l+w, t+h), (0,255,0), 2)

        # 🚀 LINE A CROSS
        if (prev_y[track_id] < line_A and cy >= line_A 
            and track_id not in counted):
            entry_time[track_id] = time.time()

        # 🚀 LINE B CROSS
        if (prev_y[track_id] < line_B and cy >= line_B 
            and track_id in entry_time 
            and track_id not in counted):

            time_taken = time.time() - entry_time[track_id]

            # 🔥 STRONG FILTER
            if 0.5 < time_taken < 5:   # realistic time only
                speed = (distance_m / time_taken) * 3.6

                # 🔥 HARD LIMIT
                if speed < 150:
                    cv2.putText(frame, f"{int(speed)} km/h",
                                (l, t-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255,255,0), 2)

                    if speed > 60:
                        cv2.putText(frame, "FINE",
                                    (l, t-30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0,0,255), 2)

                    counted.add(track_id)  # ✅ mark as done

            # cleanup
            if track_id in entry_time:
                del entry_time[track_id]

        prev_y[track_id] = cy

    cv2.imshow("FINAL SPEED SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()