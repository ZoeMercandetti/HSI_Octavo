import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Clase para gestionar múltiples Kalman Filters
class Track:
    def __init__(self, id, bbox, box):
        self.id = id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([bbox[0], bbox[1], 0, 0])  # x, y, dx, dy
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000.
        self.kf.R = np.array([[5, 0],
                              [0, 5]])
        self.kf.Q = np.eye(4)
        self.time_since_update = 0
        self.last_box = box  # para dibujar la bounding box

    def predict(self):
        self.kf.predict,()
        self.time_since_update += 1

    def update(self, bbox, box):
        self.kf.update(np.array([bbox[0], bbox[1]]))
        self.time_since_update = 0
        self.last_box = box

    def get_position(self):
        return self.kf.x[0], self.kf.x[1]

# Calcular IoU entre dos cajas [x1, y1, x2, y2]
def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Inicializa modelo YOLO
model = YOLO('yolov8n.pt')

# Video
cap = cv2.VideoCapture('loro inteligente todo loro animales inteligentes funny.mp4')

# Clases de interés (COCO dataset)
ANIMALES = {14: 'Bird', 15: 'Cat', 16: 'Dog', 17: 'Horse'}

# Variables de seguimiento
tracks = []
next_id = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección
    results = model.predict(frame, conf=0.3)[0]
    detections = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    animal_detections = []

    for det, cls in zip(detections, classes):
        if int(cls) in ANIMALES:
            x1, y1, x2, y2 = det[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            animal_detections.append(((cx, cy, int(cls)), (x1, y1, x2, y2)))

    # Predict todos los tracks existentes
    for track in tracks:
        track.predict()

    # Asignación detecciones a tracks usando IoU
    assigned_tracks = []
    assigned_detections = []

    for i, ((cx, cy, cls_id), box) in enumerate(animal_detections):
        best_track = None
        max_iou = 0.1  # umbral mínimo de intersección para asignar

        for track in tracks:
            if track.id in assigned_tracks:
                continue
            iou = bbox_iou(track.last_box, box)
            if iou > max_iou:
                max_iou = iou
                best_track = track

        if best_track:
            best_track.update((cx, cy), box)
            assigned_tracks.append(best_track.id)
            assigned_detections.append(i)

    # Crear nuevos tracks para las detecciones no asignadas
    for i, ((cx, cy, cls_id), box) in enumerate(animal_detections):
        if i not in assigned_detections:
            new_track = Track(next_id, (cx, cy), box)
            new_track.cls_id = cls_id  # guardar clase
            tracks.append(new_track)
            next_id += 1

    # Dibujar
    for track in tracks:
        x, y = track.get_position()
        label = f"{ANIMALES.get(track.cls_id, 'Animal')} {track.id}"
        x1, y1, x2, y2 = track.last_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Remover tracks viejos
    tracks = [t for t in tracks if t.time_since_update < 30]

    cv2.imshow("Seguimiento Animal 🐦", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()