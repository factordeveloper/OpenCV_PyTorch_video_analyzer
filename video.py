import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import os
import sys

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
            # Para modelos YOLO - modificado para eliminar porcentajes
            results = model(frame)
            # Obtener los resultados sin renderizar
            boxes = results[0].boxes
            
            # Crear una copia del frame para dibujar solo las cajas sin porcentajes
            frame_with_results = frame.copy()
            
            # Dibujar las cajas de detección sin porcentajes
            for box in boxes:
                # Obtener coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Obtener la clase
                cls = int(box.cls[0].item())
                
                # Obtener el nombre de la clase si está disponible
                label = results[0].names[cls] if hasattr(results[0], 'names') else f"Clase {cls}"
                
                # Dibujar el rectángulo
                cv2.rectangle(frame_with_results, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Dibujar etiqueta sin porcentaje
                cv2.putText(frame_with_results, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Para modelos PyTorch genéricos
            with torch.no_grad():
                # Convertir frame a formato PIL
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Aplicar transformaciones y preparar para el modelo
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                # Ejecutar el modelo
                outputs = model(input_tensor)
                
                # Visualizar resultados sin probabilidades
                frame_with_results = visualize_results_without_probs(frame, outputs)
        
        # Escribir el frame resultante
        out.write(frame_with_results)
    
    # Liberar recursos
    cap.release()
    out.release()
    print(f"Video analizado guardado en: {output_path}")

def visualize_results_without_probs(frame, outputs):
    """
    Visualiza los resultados del modelo en el frame sin mostrar probabilidades.
    
    Args:
        frame (numpy.ndarray): Frame original
        outputs: Salida del modelo
    
    Returns:
        numpy.ndarray: Frame con visualización de resultados
    """
    # Versión modificada para modelos de clasificación (sin probabilidades)
    if isinstance(outputs, torch.Tensor):
        probs = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
        top_class = np.argmax(probs)  # Solo la clase con mayor probabilidad
        
        # Dibujar solo la etiqueta de clase sin probabilidad
        text = f"Clase {top_class}"
        cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Para modelos de detección de objetos más complejos,
    # se necesitaría adaptar según la estructura de salida específica
    
    return frame

if __name__ == "__main__":
    # Rutas de los archivos
    model_path = "my_model.pt"  # Cambiar a la ruta correcta
    video_path = "video_final.mp4"    # Cambiar a la ruta correcta
    output_path = "video_zipaq.mp4"  # Cambiar a la ruta deseada
    
    # Comprobar si existen los archivos
    if not os.path.exists(model_path):
        print(f"Error: El archivo del modelo '{model_path}' no existe.")
        sys.exit(1)
    
    if not os.path.exists(video_path):
        print(f"Error: El archivo de video '{video_path}' no existe.")
        sys.exit(1)
    
    # Ejecutar análisis
    analyze_video(model_path, video_path, output_path)