import cv2
import numpy as np
import json
import os
from screeninfo import get_monitors

# === OBTENER RESOLUCIÓN DE PANTALLA ===
try:
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height
except Exception:
    print("⚠️ Error al obtener resolución de pantalla. Usando valores por defecto.")
    screen_w, screen_h = 1920, 1080

def resize_to_fit_screen(frame, screen_width, screen_height):
    """Redimensiona el frame para que quepa en la pantalla, manteniendo la relación de aspecto."""
    h, w = frame.shape[:2]
    scale = min(screen_width * 0.9 / w, screen_height * 0.9 / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_frame, scale

def load_corners(json_path):
    """Carga las esquinas desde un archivo JSON si existe."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            corners = json.load(f)
        return np.array(corners, dtype=np.float32)
    return None

def save_corners(corners, json_path):
    """Guarda las esquinas en un archivo JSON."""
    with open(json_path, 'w') as f:
        json.dump(corners.tolist(), f)
    print(f"✅ Esquinas guardadas en {json_path}")

def select_corners_manually(frame, screen_w, screen_h):
    """Permite seleccionar manualmente las 4 esquinas del tablero."""
    global board_corners, click_count
    board_corners = []
    click_count = 0
    
    # Redimensionar frame para visualización
    resized_frame, scale = resize_to_fit_screen(frame, screen_w, screen_h)
    
    def mouse_callback(event, x, y, flags, param):
        global board_corners, click_count
        if event == cv2.EVENT_LBUTTONDOWN and click_count < 4:
            # Escalar coordenadas al tamaño original
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            board_corners.append((orig_x, orig_y))
            click_count += 1
            print(f"Esquina {click_count} seleccionada: ({orig_x}, {orig_y})")
    
    cv2.namedWindow("Selecciona Esquinas", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Selecciona Esquinas", mouse_callback)
    
    while click_count < 4:
        temp_frame = resized_frame.copy()
        for i, pt in enumerate(board_corners):
            # Escalar coordenadas para visualización
            vis_x = int(pt[0] * scale)
            vis_y = int(pt[1] * scale)
            cv2.circle(temp_frame, (vis_x, vis_y), 8, (0, 255, 0), -1)
            cv2.putText(temp_frame, f"Esquina {i+1}", (vis_x + 15, vis_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Selecciona Esquinas", temp_frame)
        cv2.resizeWindow("Selecciona Esquinas", temp_frame.shape[1], temp_frame.shape[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Interrumpido por el usuario.")
    
    cv2.destroyWindow("Selecciona Esquinas")
    return np.array(board_corners, dtype=np.float32)

def get_perspective_transform(corners, board_size=400):
    """Calcula la transformación de perspectiva para obtener un tablero cuadrado."""
    dst_pts = np.float32([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
    matrix = cv2.getPerspectiveTransform(corners, dst_pts)
    return matrix, board_size

def get_grid_coordinates(board_size):
    """Genera coordenadas de la cuadrícula 8x8 y notación de casillas."""
    cell_size = board_size / 8
    grid = {}
    for i in range(8):
        for j in range(8):
            square = f"{chr(97 + j)}{8 - i}"
            x_min = j * cell_size
            y_min = i * cell_size
            x_max = (j + 1) * cell_size
            y_max = (i + 1) * cell_size
            grid[square] = (x_min, y_min, x_max, y_max)
    return grid

def segment_board(frame, screen_w, screen_h, json_path="board_corners.json"):
    """Segmenta el tablero y devuelve la matriz de perspectiva, la cuadrícula y las esquinas."""
    # Intentar cargar esquinas desde JSON
    corners = load_corners(json_path)
    
    if corners is None or len(corners) != 4:
        print("ℹ️ Selecciona las 4 esquinas del tablero (superior-izq, superior-der, inferior-der, inferior-izq).")
        corners = select_corners_manually(frame, screen_w, screen_h)
        save_corners(corners, json_path)
    
    matrix, board_size = get_perspective_transform(corners)
    grid = get_grid_coordinates(board_size)
    return matrix, grid, corners

if __name__ == "__main__":
    # Prueba standalone
    cap = cv2.VideoCapture(r"C:\Users\maste\Desktop\proyecto2\video.mp4")
    if not cap.isOpened():
        print("❌ No se pudo abrir el video.")
        exit()
    
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución del video: {video_w}x{video_h}")
    
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el primer frame.")
        cap.release()
        exit()
    
    try:
        json_path = r"C:\Users\maste\Desktop\proyecto2\board_corners.json"
        matrix, grid, corners = segment_board(frame, screen_w, screen_h, json_path)
        print("✅ Tablero segmentado correctamente.")
        print("Esquinas:", corners)
        print("Cuadrícula:", list(grid.keys())[:5], "...")
        
        # Mostrar resultado para depuración
        debug_frame = frame.copy()
        resized_debug, scale = resize_to_fit_screen(debug_frame, screen_w, screen_h)
        for corner in corners:
            vis_x = int(corner[0] * scale)
            vis_y = int(corner[1] * scale)
            cv2.circle(resized_debug, (vis_x, vis_y), 8, (0, 255, 0), -1)
            cv2.putText(resized_debug, "Corner", (vis_x + 15, vis_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Tablero Detectado", resized_debug)
        cv2.resizeWindow("Tablero Detectado", resized_debug.shape[1], resized_debug.shape[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("⏹️ Interrumpido por el usuario.")
    finally:
        cap.release()