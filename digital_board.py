import cv2
import numpy as np
import time
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

def draw_digital_board(positions, board_size=400):
    """Dibuja un tablero digital 8x8 con las piezas en las casillas."""
    # Crear imagen en blanco
    board = np.ones((board_size + 50, board_size + 50, 3), dtype=np.uint8) * 255  # Fondo blanco
    cell_size = board_size // 8
    
    # Colores (RGB)
    light_square = (240, 217, 181)  # Marrón claro
    dark_square = (181, 136, 99)   # Marrón oscuro
    text_color = (0, 0, 0)         # Negro para texto
    
    # Dibujar casillas
    for row in range(8):
        for col in range(8):
            color = light_square if (row + col) % 2 == 0 else dark_square
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = (col + 1) * cell_size
            y2 = (row + 1) * cell_size
            cv2.rectangle(board, (x1, y1), (x2, y2), color, -1)
            
            # Dibujar pieza si existe
            square = f"{chr(97 + col)}{8 - row}"
            if square in positions:
                piece = positions[square]
                cv2.putText(board, piece, (x1 + cell_size//3, y1 + 2*cell_size//3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    # Dibujar etiquetas de filas (1-8) y columnas (a-h)
    for i in range(8):
        # Filas (1-8)
        cv2.putText(board, str(8 - i), (board_size + 10, i * cell_size + cell_size//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        # Columnas (a-h)
        cv2.putText(board, chr(97 + i), (i * cell_size + cell_size//2, board_size + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    return board

def main_standalone(video_path, model_path, board_size=400):
    """Prueba standalone para mostrar el tablero digital con detecciones."""
    from board_segmentation import segment_board
    from yolo_processing import process_yolo_detections, YOLO
    
    # Cargar modelo YOLO
    try:
        model = YOLO(model_path)
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
    
    # Obtener matriz de perspectiva y cuadrícula
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el primer frame.")
        cap.release()
        exit()
    
    matrix, grid, corners = segment_board(frame, screen_w, screen_h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video
    
    last_processed_time = 0
    detection_interval = 0.5  # Intervalo de detección (segundos)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video finalizado.")
                break
            
            # Escalar frame para visualización
            resized_frame, scale = resize_to_fit_screen(frame, screen_w, screen_h)
            
            # Procesar detecciones cada DETECTION_INTERVAL
            current_time = time.time()
            if current_time - last_processed_time >= detection_interval:
                last_processed_time = current_time
                positions, detections = process_yolo_detections(frame, model, matrix, grid, board_size, video_w, video_h)
                
                # Imprimir posiciones para depuración
                print(f"\nFrame {cap.get(cv2.CAP_PROP_POS_FRAMES)} - Posiciones: {positions}")
            
            # Dibujar detecciones y esquinas en el video
            for det in detections:
                x1, y1, x2, y2, conf, label, square = det
                vis_x1, vis_y1 = int(x1 * scale), int(y1 * scale)
                vis_x2, vis_y2 = int(x2 * scale), int(y2 * scale)
                cv2.rectangle(resized_frame, (vis_x1, vis_y1), (vis_x2, vis_y2), (0, 0, 255), 2)
                cv2.putText(resized_frame, f"{label} ({square})", (vis_x1, vis_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for corner in corners:
                vis_x = int(corner[0] * scale)
                vis_y = int(corner[1] * scale)
                cv2.circle(resized_frame, (vis_x, vis_y), 8, (0, 255, 0), -1)
            
            # Dibujar tablero digital
            digital_board = draw_digital_board(positions, board_size)
            
            # Mostrar ventanas
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
    video_path = r"C:\Users\maste\Desktop\proyecto2\video.mp4"
    model_path = r"C:\Users\maste\Desktop\proyecto2\best.pt"
    main_standalone(video_path, model_path, board_size=400)