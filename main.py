import cv2
import numpy as np
import time
import chess
import chess.svg
from screeninfo import get_monitors
from cairosvg import svg2png
from PIL import Image
import io
from ultralytics import YOLO
import json
import os

# === CONFIGURACIÓN AJUSTABLE ===
CONFIG = {
    'MODEL_PATH': r"C:\Users\maste\Desktop\proyecto2\best.pt",  # Ruta al modelo YOLO
    'VIDEO_PATH': r"C:\Users\maste\Desktop\proyecto2\video.mp4",  # Ruta al video
    'CORNERS_FILE': r"C:\Users\maste\Desktop\proyecto2\board_corners.json",  # Archivo de esquinas
    'CONFIDENCE_THRESHOLD': 0.3,  # Umbral de confianza para YOLO (0.2: más detecciones, 0.4: más precisas)
                                  # Opciones: 0.2, 0.25, 0.3, 0.4
    'DETECTION_INTERVAL': 0.5,  # Intervalo entre detecciones en segundos (1.0: equilibrado, 0.5: más frecuente)
                                # Opciones: 0.5, 1.0, 1.5, 2.0
    'INPUT_RESOLUTION': (2560, 1440),  # Resolución para YOLO (320x320: rápido, 640x480: preciso)
                                     # Opciones: (320, 320), (416, 416), (640, 480)
    'BOARD_SIZE': 400,  # Tamaño del tablero digital en píxeles (400: equilibrado)
                        # Opciones: 300, 400, 500
    'FRAME_SUBSAMPLE': 1.0,  # Procesar cada N frames (1: todos, 2: cada segundo frame)
                           # Opciones: 1, 2, 3
    'STABILITY_THRESHOLD': 2,  # Detecciones consecutivas para confirmar una pieza (3: estable, 2: más rápido)
                               # Opciones: 2, 3, 4
}

# === OBTENER RESOLUCIÓN DE PANTALLA ===
try:
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height
except Exception:
    print("Error al obtener resolución de pantalla. Usando valores por defecto.")
    screen_w, screen_h = 1920, 1080

# === SEGMENTACIÓN DEL TABLERO ===
def select_corners(event, x, y, flags, param):
    corners, frame, scale = param
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x / scale, y / scale))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Seleccionar Esquinas", frame)

def segment_board(frame):
    corners = []
    if os.path.exists(CONFIG['CORNERS_FILE']):
        with open(CONFIG['CORNERS_FILE'], 'r') as f:
            corners = json.load(f)
        print("✅ Esquinas cargadas desde", CONFIG['CORNERS_FILE'])
    else:
        frame_copy = frame.copy()
        resized_frame, scale = resize_to_fit_screen(frame_copy)
        cv2.namedWindow("Seleccionar Esquinas")
        cv2.setMouseCallback("Seleccionar Esquinas", select_corners, [corners, resized_frame, scale])
        print("Seleccione las 4 esquinas en orden: superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda")
        while len(corners) < 4:
            cv2.imshow("Seleccionar Esquinas", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt("Selección cancelada.")
        cv2.destroyWindow("Seleccionar Esquinas")
        with open(CONFIG['CORNERS_FILE'], 'w') as f:
            json.dump(corners, f)
        print("✅ Esquinas guardadas en", CONFIG['CORNERS_FILE'])
    
    src_pts = np.float32(corners)
    board_size = CONFIG['BOARD_SIZE']
    dst_pts = np.float32([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return matrix, corners

# === PROCESAMIENTO DE DETECCIONES YOLO ===
def map_piece_label(label, square):
    piece_map = {
        "white-pawn": "P", "black-pawn": "p",
        "white-rook": "R", "black-rook": "r",
        "white-knight": "N", "black-knight": "n",
        "white-bishop": "B", "black-bishop": "b",
        "white-queen": "Q", "black-queen": "q",
        "white-king": "K", "black-king": "k"
    }
    return piece_map.get(label.lower(), "")

def process_yolo_detections(frame, model, matrix, video_w, video_h):
    frame_resized = cv2.resize(frame, CONFIG['INPUT_RESOLUTION'], interpolation=cv2.INTER_NEAREST)
    results = model.predict(frame_resized, conf=CONFIG['CONFIDENCE_THRESHOLD'])
    
    scale_x = video_w / CONFIG['INPUT_RESOLUTION'][0]
    scale_y = video_h / CONFIG['INPUT_RESOLUTION'][1]
    board_size = CONFIG['BOARD_SIZE']
    positions = {}
    detections = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        conf = float(box.conf[0])
        label = model.names[int(box.cls[0])]
        
        center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
        center_warped = cv2.perspectiveTransform(center[None, :, :], matrix)[0][0]
        print(f"Etiqueta: {label}, Coordenadas: {center_warped}")
        
        if 0 <= center_warped[0] < board_size and 0 <= center_warped[1] < board_size:
            col = int(center_warped[0] * 8 / board_size)
            row = int(center_warped[1] * 8 / board_size)
            square = f"{chr(97 + col)}{8 - row}"
            piece = map_piece_label(label, square)
            if piece:
                positions[square] = piece
                detections.append((x1, y1, x2, y2, conf, label, square))
    
    return positions, detections

# === VISUALIZACIÓN CON PYTHON-CHESS ===
def positions_to_board(positions):
    board = chess.Board(None)
    for square, piece in positions.items():
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        square_idx = row * 8 + col
        piece_map = {'P': chess.PAWN, 'p': chess.PAWN, 'R': chess.ROOK, 'r': chess.ROOK,
                     'N': chess.KNIGHT, 'n': chess.KNIGHT, 'B': chess.BISHOP, 'b': chess.BISHOP,
                     'Q': chess.QUEEN, 'q': chess.QUEEN, 'K': chess.KING, 'k': chess.KING}
        if piece in piece_map:
            color = chess.WHITE if piece.isupper() else chess.BLACK
            board.set_piece_at(square_idx, chess.Piece(piece_map[piece], color))
    return board

def draw_digital_board(positions, last_positions, last_board_img):
    positions_str = str(sorted(positions.items()))
    last_positions_str = str(sorted(last_positions.items()))
    if positions_str == last_positions_str and last_board_img is not None:
        return last_board_img, positions
    
    board = positions_to_board(positions)
    svg_data = chess.svg.board(board, size=CONFIG['BOARD_SIZE'])
    png_buffer = io.BytesIO()
    svg2png(bytestring=svg_data, write_to=png_buffer)
    img = Image.open(png_buffer)
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_array, positions

# === FILTRO DE ESTABILIDAD ===
def stabilize_positions(new_positions, prev_positions, counter):
    stabilized = {}
    for square, piece in new_positions.items():
        if square in prev_positions and prev_positions[square] == piece:
            counter[square] = counter.get(square, 0) + 1
        else:
            counter[square] = 1
        if counter.get(square, 0) >= CONFIG['STABILITY_THRESHOLD']:
            stabilized[square] = piece
    return stabilized, counter

# === UTILIDADES ===
def resize_to_fit_screen(frame):
    h, w = frame.shape[:2]
    scale = min(screen_w * 0.9 / w, screen_h * 0.9 / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_frame, scale

# === PROGRAMA PRINCIPAL ===
def main():
    try:
        model = YOLO(CONFIG['MODEL_PATH'])
        print("✅ Modelo YOLO cargado.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        exit()
    
    cap = cv2.VideoCapture(CONFIG['VIDEO_PATH'])
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {CONFIG['VIDEO_PATH']}")
        exit()
    
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución del video: {video_w}x{video_h}")
    
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el primer frame.")
        cap.release()
        exit()
    
    matrix, corners = segment_board(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    last_processed_time = 0
    prev_positions = {}
    counter = {}
    last_board_img = None
    last_positions = {}
    fps_start_time = time.time()
    frame_count = 0
    frame_idx = 0
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video finalizado.")
                break
            
            frame_idx += 1
            if frame_idx % CONFIG['FRAME_SUBSAMPLE'] != 0:
                continue
            
            frame_count += 1
            if time.time() - fps_start_time >= 1:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()
            
            resized_frame, scale = resize_to_fit_screen(frame)
            current_time = time.time()
            
            positions = prev_positions
            detections = []
            if current_time - last_processed_time >= CONFIG['DETECTION_INTERVAL']:
                last_processed_time = current_time
                positions, detections = process_yolo_detections(frame, model, matrix, video_w, video_h)
                stabilized_positions, counter = stabilize_positions(positions, prev_positions, counter)
                prev_positions = positions
                positions = stabilized_positions
                print(f"\nFrame {cap.get(cv2.CAP_PROP_POS_FRAMES)} - Posiciones: {positions}")
            
            for det in detections:
                x1, y1, x2, y2, _, label, square = det
                vis_x1, vis_y1 = int(x1 * scale), int(y1 * scale)
                vis_x2, vis_y2 = int(x2 * scale), int(y2 * scale)
                cv2.rectangle(resized_frame, (vis_x1, vis_y1), (vis_x2, vis_y2), (0, 0, 255), 2)
                cv2.putText(resized_frame, f"{label} ({square})", (vis_x1, vis_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for corner in corners:
                vis_x = int(corner[0] * scale)
                vis_y = int(corner[1] * scale)
                cv2.circle(resized_frame, (vis_x, vis_y), 8, (0, 255, 0), -1)
            
            digital_board, last_positions = draw_digital_board(positions, last_positions, last_board_img)
            last_board_img = digital_board
            
            cv2.putText(resized_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Detecciones YOLO", resized_frame)
            cv2.resizeWindow("Detecciones YOLO", resized_frame.shape[1], resized_frame.shape[0])
            cv2.imshow("Tablero Digital", digital_board)
            cv2.resizeWindow("Tablero Digital", digital_board.shape[1], digital_board.shape[0])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Interrumpido por el usuario.")
                break
    
    except KeyboardInterrupt:
        print("⏹️ Interrumpido por el usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()