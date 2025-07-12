import cv2
import time
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("runs/detect/train8/weights/best.pt")  # o tu ruta personalizada

# Inicializar cámara (0 = webcam por defecto)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Bucle principal de captura
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break

    # Inferencia con el modelo
    results = model(frame)[0]

    ball_box = None
    zone_box = None

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        cls = int(cls)
        label = model.names[cls]

        # Dibujar cajas con colores diferentes
        color = (0, 255, 0) if label == "ball" else (255, 0, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if label == "ball":
            ball_box = [x1, y1, x2, y2]
        elif label == "strike_zone":
            zone_box = [x1, y1, x2, y2]

    # Determinar si es bola o strike
    if ball_box and zone_box:
        bx = (ball_box[0] + ball_box[2]) / 2
        by = (ball_box[1] + ball_box[3]) / 2
        zx1, zy1, zx2, zy2 = zone_box
        if zx1 <= bx <= zx2 and zy1 <= by <= zy2:
            cv2.putText(frame, "STRIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 4)
        else:
            cv2.putText(frame, "BALL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 4)

    # Mostrar FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar frame en pantalla
    cv2.imshow("Transmisión - Bola o Strike", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
