import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import os
import sys
from collections import defaultdict

def calculate_iou(box1, box2):
    """
    Calcula el Intersection over Union (IoU) entre dos bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calcular área de intersección
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calcular área de unión
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

class ObjectTracker:
    """
    Clase para rastrear objetos únicos a través de frames.
    """
    def __init__(self, iou_threshold=0.3, max_frames_missing=30):
        self.tracked_objects = {}  # {track_id: {'class': str, 'box': [x1,y1,x2,y2], 'frames_missing': int}}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.object_counts = defaultdict(int)  # Conteo de objetos únicos por clase
        
    def update(self, detections):
        """
        Actualiza el tracker con las detecciones del frame actual.
        
        Args:
            detections: lista de diccionarios [{'class': str, 'box': [x1,y1,x2,y2], 'confidence': float}, ...]
        """
        # Marcar todos los objetos rastreados como no vistos en este frame
        for track_id in self.tracked_objects:
            self.tracked_objects[track_id]['frames_missing'] += 1
        
        # Intentar asociar cada detección con objetos rastreados existentes
        matched_tracks = set()
        
        for detection in detections:
            best_iou = 0
            best_track_id = None
            
            # Buscar el objeto rastreado más similar
            for track_id, tracked_obj in self.tracked_objects.items():
                if tracked_obj['class'] == detection['class']:
                    iou = calculate_iou(detection['box'], tracked_obj['box'])
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Actualizar objeto existente
                self.tracked_objects[best_track_id]['box'] = detection['box']
                self.tracked_objects[best_track_id]['frames_missing'] = 0
                matched_tracks.add(best_track_id)
            else:
                # Crear nuevo objeto rastreado
                self.tracked_objects[self.next_id] = {
                    'class': detection['class'],
                    'box': detection['box'],
                    'frames_missing': 0
                }
                self.object_counts[detection['class']] += 1
                self.next_id += 1
        
        # Eliminar objetos que llevan mucho tiempo sin verse
        tracks_to_remove = []
        for track_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj['frames_missing'] > self.max_frames_missing:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
    
    def get_counts(self):
        """
        Retorna el conteo de objetos únicos por clase.
        """
        return dict(self.object_counts)

def analyze_video(model_path, video_path, output_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Analiza un video usando un modelo PyTorch y genera un nuevo video con el análisis.
    
    Args:
        model_path (str): Ruta al archivo del modelo PyTorch (.pt)
        video_path (str): Ruta al archivo de video (.mp4)
        output_path (str): Ruta donde guardar el video de salida
        device (str): Dispositivo donde ejecutar el modelo ('cuda' o 'cpu')
    """
    print(f"Usando dispositivo: {device}")
    
    # Cargar el modelo - para modelos YOLO de Ultralytics
    try:
        # Intentar cargar como modelo de Ultralytics (YOLO)
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print(f"Modelo YOLO cargado desde {model_path}")
            model_type = "yolo"
        except ImportError:
            print("Ultralytics no está instalado. Intentando cargar como modelo PyTorch genérico...")
            # Intentar cargar como modelo PyTorch genérico
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            model.to(device)
            model_type = "pytorch"
            print(f"Modelo PyTorch genérico cargado desde {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Configurar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Inicializar el tracker
    tracker = ObjectTracker(iou_threshold=0.3, max_frames_missing=30)
    
    # Para modelos PyTorch genéricos
    if model_type == "pytorch":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            frames_per_second = frame_count / elapsed
            remaining_time = (total_frames - frame_count) / frames_per_second
            print(f"Procesando frame {frame_count}/{total_frames} - {frames_per_second:.2f} FPS - Tiempo restante: {remaining_time:.2f} segundos")
        
        # Procesar el frame según el tipo de modelo
        if model_type == "yolo":
            # Para modelos YOLO
            results = model(frame)
            frame_with_results = results[0].plot()  # YOLO ya incluye visualización
            
            # Extraer detecciones del frame actual
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Obtener coordenadas del bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'class': class_name,
                            'box': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': confidence
                        })
            
            # Actualizar el tracker con las detecciones
            tracker.update(detections)
            
            # Dibujar contador de objetos únicos en la esquina superior izquierda
            frame_with_results = draw_detection_counter(frame_with_results, tracker.get_counts())
        else:
            # Para modelos PyTorch genéricos
            with torch.no_grad():
                # Convertir frame a formato PIL
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Aplicar transformaciones y preparar para el modelo
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                # Ejecutar el modelo
                outputs = model(input_tensor)
                
                # Visualizar resultados
                frame_with_results = visualize_results(frame, outputs, tracker)
        
        # Escribir el frame resultante
        out.write(frame_with_results)
    
    # Liberar recursos
    cap.release()
    out.release()
    
    # Mostrar resumen final
    print(f"\nVideo analizado guardado en: {output_path}")
    print(f"\nResumen de objetos únicos detectados:")
    total_unique_objects = tracker.get_counts()
    for label, count in sorted(total_unique_objects.items()):
        print(f"  {label}: {count}")
    print(f"\nTotal de objetos únicos: {sum(total_unique_objects.values())}")

def draw_detection_counter(frame, detection_counts):
    """
    Dibuja un contador de detecciones en la esquina superior izquierda.
    
    Args:
        frame (numpy.ndarray): Frame con las detecciones
        detection_counts (dict): Diccionario con conteo de objetos únicos por clase
    
    Returns:
        numpy.ndarray: Frame con el contador dibujado
    """
    if not detection_counts:
        return frame
    
    # Configuración del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)  # Blanco
    bg_color = (0, 0, 0)  # Negro
    padding = 10
    line_height = 25
    
    # Calcular el tamaño del recuadro
    max_text_width = 0
    num_lines = len(detection_counts)
    
    for label, count in detection_counts.items():
        text = f"{label}: {count}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_width)
    
    # Dimensiones del recuadro
    box_width = max_text_width + 2 * padding
    box_height = num_lines * line_height + 2 * padding
    
    # Dibujar recuadro negro semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Dibujar el texto
    y_offset = padding + 20
    for label, count in sorted(detection_counts.items()):
        text = f"{label}: {count}"
        cv2.putText(frame, text, (padding, y_offset), font, font_scale, text_color, font_thickness)
        y_offset += line_height
    
    return frame

def visualize_results(frame, outputs, tracker):
    """
    Visualiza los resultados del modelo en el frame.
    Debes adaptar esta función según el tipo de modelo y su salida.
    
    Args:
        frame (numpy.ndarray): Frame original
        outputs: Salida del modelo
        tracker (ObjectTracker): Tracker de objetos
    
    Returns:
        numpy.ndarray: Frame con visualización de resultados
    """
    # IMPORTANTE: Esta función es un ejemplo y debe ser adaptada
    # según el tipo específico de modelo que estés usando y su salida
    
    # Ejemplo para un modelo de clasificación:
    if isinstance(outputs, torch.Tensor):
        # Convertir a numpy si es un tensor
        probs = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
        top_classes = np.argsort(probs)[-3:][::-1]  # Top 3 clases
        
        # Crear detecciones (para modelos de clasificación necesitarías adaptar esto)
        detections = []
        for i, class_idx in enumerate(top_classes):
            class_prob = probs[class_idx]
            if class_prob > 0.5:  # Solo contar si probabilidad > 50%
                detections.append({
                    'class': f"Clase {class_idx}",
                    'box': [0, 0, frame.shape[1], frame.shape[0]],  # Bounding box de todo el frame
                    'confidence': class_prob
                })
        
        # Actualizar tracker
        tracker.update(detections)
        
        # Dibujar contador
        frame = draw_detection_counter(frame, tracker.get_counts())
    
    return frame

if __name__ == "__main__":
    # Rutas de los archivos
    model_path = "bnsf_model.pt"  # Cambiar a la ruta correcta
    video_path = "bnsf_types_dataset.mp4"  # Cambiar a la ruta correcta
    output_path = "bnsf_types_tracker.mp4"  # Cambiar a la ruta deseada
    
    # Comprobar si existen los archivos
    if not os.path.exists(model_path):
        print(f"Error: El archivo del modelo '{model_path}' no existe.")
        sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"Error: El archivo de video '{video_path}' no existe.")
        sys.exit(1)
    
    # Ejecutar análisis
    analyze_video(model_path, video_path, output_path)


