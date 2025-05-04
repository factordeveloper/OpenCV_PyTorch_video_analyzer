import json
import cv2
import easyocr
import numpy as np
import os

# Crear directorio para guardar los resultados si no existe
output_dir = 'resultados_ocr'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar el archivo JSON exportado desde Label Studio
with open('modelo_ocr.json', 'r') as f:
    anotaciones = json.load(f)

# Inicializar EasyOCR
reader = easyocr.Reader(['es', 'en'])  # Idiomas que quieres detectar

# Procesar un video
video_path = 'uptrain.mp4'  # Reemplaza con la ruta de tu video
cap = cv2.VideoCapture(video_path)

# Configurar el escritor de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = None

frame_count = 0
detected_texts = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inicializar el escritor de video en el primer frame para obtener dimensiones correctas
        if output_video is None:
            height, width = frame.shape[:2]
            output_video = cv2.VideoWriter(
                os.path.join(output_dir, 'output_video.avi'),
                fourcc, 20.0, (width, height)
            )
        
        # Crear una copia del frame para dibujar los resultados
        result_frame = frame.copy()
        
        # Texto detectado en este frame
        frame_texts = []
        
        # Usar las regiones definidas en el JSON para delimitar áreas de OCR
        for anotacion in anotaciones:
            if 'annotations' in anotacion and len(anotacion['annotations']) > 0:
                for result in anotacion['annotations'][0].get('result', []):
                    if result.get('type') == 'rectangle':
                        # Extraer las coordenadas del rectángulo
                        value = result.get('value', {})
                        if 'x' in value and 'y' in value:
                            # Obtener coordenadas en píxeles
                            x = int(value['x'] * frame.shape[1] / 100)
                            y = int(value['y'] * frame.shape[0] / 100)
                            width = int(value['width'] * frame.shape[1] / 100)
                            height = int(value['height'] * frame.shape[0] / 100)
                            
                            # Verificar que las coordenadas estén dentro de los límites del frame
                            if x >= 0 and y >= 0 and x + width <= frame.shape[1] and y + height <= frame.shape[0]:
                                # Recortar la región de interés
                                roi = frame[y:y+height, x:x+width]
                                
                                # Verificar que el ROI no esté vacío
                                if roi.size > 0:
                                    # Aplicar OCR a esta región
                                    ocr_results = reader.readtext(roi)
                                    
                                    # Registrar y dibujar resultados
                                    for (bbox, text, prob) in ocr_results:
                                        if prob > 0.5:  # Solo considerar resultados con probabilidad > 0.5
                                            # Coordenadas relativas a la imagen completa
                                            tl = (x + int(bbox[0][0]), y + int(bbox[0][1]))
                                            br = (x + int(bbox[2][0]), y + int(bbox[2][1]))
                                            
                                            # Dibujar rectángulo y texto
                                            cv2.rectangle(result_frame, tl, br, (0, 255, 0), 2)
                                            cv2.putText(result_frame, text, (tl[0], tl[1] - 10), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                            
                                            # Guardar texto detectado
                                            frame_texts.append({
                                                'text': text,
                                                'confidence': prob,
                                                'bbox': [tl, br]
                                            })
        
        # Guardar imagen del frame con anotaciones
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.jpg'), result_frame)
        
        # Escribir en video de salida
        output_video.write(result_frame)
        
        # Guardar textos detectados para este frame
        detected_texts.append({
            'frame': frame_count,
            'texts': frame_texts
        })
        
        frame_count += 1
        
        # Opcional: imprimir progreso cada 10 frames
        if frame_count % 10 == 0:
            print(f"Procesados {frame_count} frames")

except Exception as e:
    print(f"Error al procesar el video: {e}")

finally:
    # Liberar recursos
    cap.release()
    if output_video is not None:
        output_video.release()
    
    # Guardar resultados de OCR en un archivo JSON
    with open(os.path.join(output_dir, 'detected_texts.json'), 'w', encoding='utf-8') as f:
        json.dump(detected_texts, f, ensure_ascii=False, indent=2)
    
    print(f"Procesamiento completado. Se analizaron {frame_count} frames.")
    print(f"Resultados guardados en la carpeta '{output_dir}'")