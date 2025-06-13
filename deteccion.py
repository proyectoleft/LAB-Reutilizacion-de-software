import cv2
import numpy as np
from ultralytics import YOLO
import time
from screeninfo import get_monitors

# === OBTENER RESOLUCIÓN DE PANTALLA ===
try:
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height
except Exception:
    print("⚠️ Error al obtener resolución de pantalla. Usando valores por defecto.")
    screen_w, screen_h = 1920, 1080

# === CONFIGURACIÓN ===
MODEL_PATH = r"C:\Users\maste\Desktop\proyecto2\best.pt"  # Ruta al modelo YOLO
CONFIDENCE_THRESHOLD = 0.25  # Umbral de confianza reducido para detectar más piezas
INPUT_RESOLUTION = (640, 480)  # Resolución de entrada para YOLO
DETECTION_INTERVAL = 1.5  # Intervalo de detección (segundos)

def resize_to_fit_screen(frame, screen_width, screen_height):
    """Redimensiona el frame para que quepa en la pantalla, manteniendo la relación de aspecto."""
    h, w = frame.shape[:2]
    scale = min(screen_width * 0.9 / w, screen_height * 0.9 / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_frame, scale

def map_piece_label(label, square):
    """Convierte etiquetas de YOLO a iniciales estándar, deduciendo color por posición."""
    label = label.lower()
    # Determinar color según la fila (1-2: blancas, 7-8: negras)
    row = int(square[1])
    is_white = row in [1, 2]
    
    piece_map = {
        "peón": "P" if is_white else "p",
        "pawn": "P" if is_white else "p",
        "torre": "R" if is_white else "r",
        "rook": "R" if is_white else "r",
        "caballo": "N" if is_white else "n",
        "knight": "N" if is_white else "n",
        "alfil": "B" if is_white else "b",
        "bishop": "B" if is_white else "b",
        "reina": "Q" if is_white else "q",
        "queen": "Q" if is_white else "q",
        "rey": "K" if is_white else "k",
        "king": "K" if is_white else "k"
    }
    
    for key in piece_map:
        if key in label:
            return piece_map[key]
    return ""  # Etiqueta desconocida

def process_yolo_detections(frame, model, matrix, grid, board_size, video_w, video_h):
    """Procesa detecciones de YOLO y las mapea a casillas del tablero."""
    # Redimensionar frame para YOLO
    frame_resized = cv2.resize(frame, INPUT_RESOLUTION, interpolation=cv2.INTER_AREA)
    results = model.predict(frame_resized, conf=CONFIDENCE_THRESHOLD)
    
    # Escalar coordenadas al tamaño original
    scale_x = video_w / INPUT_RESOLUTION[0]
    scale_y = video_h / INPUT_RESOLUTION[1]
    
    positions = {}
    detections = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        
        # Mapear al tablero corregido
        center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
        center_warped = cv2.perspectiveTransform(center[None, :, :], matrix)[0][0]
        
        # Depuración: verificar coordenadas transformadas
        print(f"Coordenadas transformadas: {center_warped}, Tablero: {board_size}x{board_size}")
        
        # Verificar si está dentro del tablero
        if 0 <= center_warped[0] < board_size and 0 <= center_warped[1] < board_size:
            col = int(center_warped[0] * 8 / board_size)
            row = int(center_warped[1] * 8 / board_size)
            square = f"{chr(97 + col)}{8 - row}"
            piece = map_piece_label(label, square)
            if piece:
                positions[square] = piece
                detections.append((x1, y1, x2, y2, conf, label, square))
    
    return positions, detections

def main_standalone(video_path, matrix, grid, board_size, corners):
    """Prueba standalone para procesar detecciones en el video."""
    # Cargar modelo
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modelo YOLO cargado correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        exit()
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        exit()
    
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución del video: {video_w}x{video_h}")
    
    last_processed_time = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video finalizado.")
                break
            
            # Escalar frame para visualización
            resized_frame, scale = resize_to_fit_screen(frame, screen_w, screen_h)
            
            current_time = time.time()
            if current_time - last_processed_time >= DETECTION_INTERVAL:
                last_processed_time = current_time
                positions, detections = process_yolo_detections(frame, model, matrix, grid, board_size, video_w, video_h)
                
                # Imprimir detecciones para depuración
                print(f"\nDetecciones en frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}:")
                for det in detections:
                    x1, y1, x2, y2, conf, label, square = det
                    print(f"Pieza: {label}, Confianza: {conf:.2f}, Casilla: {square}")
                print(f"Posiciones: {positions}")
            
            # Dibujar detecciones y esquinas
            for det in detections:
                x1, y1, x2, y2, conf, label, square = det
                vis_x1, vis_y1 = int(x1 * scale), int(y1 * scale)
                vis_x2, vis_y2 = int(x2 * scale), int(y2 * scale)
                cv2.rectangle(resized_frame, (vis_x1, vis_y1), (vis_x2, vis_y2), (0, 0, 255), 2)
                cv2.putText(resized_frame, f"{label} ({square})", (vis_x1, vis_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Dibujar esquinas del tablero
            for corner in corners:
                vis_x = int(corner[0] * scale)
                vis_y = int(corner[1] * scale)
                cv2.circle(resized_frame, (vis_x, vis_y), 8, (0, 255, 0), -1)
            
            cv2.imshow("Detecciones YOLO", resized_frame)
            cv2.resizeWindow("Detecciones YOLO", resized_frame.shape[1], resized_frame.shape[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Interrumpido por el usuario.")
                break
    
    except KeyboardInterrupt:
        print("⏹️ Interrumpido por el usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    from board_segmentation import segment_board
    video_path = r"C:\Users\maste\Desktop\proyecto2\video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        exit()
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ No se pudo leer el primer frame.")
        exit()
    
    matrix, grid, corners = segment_board(frame, screen_w=1920, screen_h=1080)
    main_standalone(video_path, matrix, grid, board_size=400, corners=corners)