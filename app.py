from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
from ultralytics import YOLO
from deepface import DeepFace
from collections import Counter
import numpy as np
from PIL import Image
import io
import logging

app = FastAPI(title="Detección de Aglomeraciones API")

# Configuración de logging para producción
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carga el modelo PyTorch original
model = YOLO('models/best.pt')  # Ruta a tu best.pt (ponlo en carpeta models/)

# Parámetros óptimos
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
AGGLOMERACION_THRESHOLD = 50
MAX_EMOTION_ANALYSES = 10  # Limita DeepFace para optimización

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lee la imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Redimensiona para optimización (opcional, acelera YOLO)
        height, width = image_cv.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_cv = cv2.resize(image_cv, (new_width, new_height))

        # Inferencia con YOLO (usando best.pt)
        results = model.predict(image_cv, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

        # Procesa detecciones
        detections = results[0].boxes
        boxes = detections.xyxy.cpu().numpy()
        confidences = detections.conf.cpu().numpy()
        classes = detections.cls.cpu().numpy()

        valid_detections = []
        for i in range(len(boxes)):
            if confidences[i] > CONFIDENCE_THRESHOLD and classes[i] == 0:  # Clase 'person'
                x1, y1, x2, y2 = boxes[i]
                valid_detections.append((x1, y1, x2, y2, confidences[i]))

        conteo = len(valid_detections)
        es_aglomeracion = conteo > AGGLOMERACION_THRESHOLD

        # Análisis de emociones (limitado para optimización)
        emociones = []
        for idx, (x1, y1, x2, y2, conf) in enumerate(valid_detections[:MAX_EMOTION_ANALYSES]):
            face_roi = image_cv[int(y1):int(y2), int(x1):int(x2)]
            try:
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']
                emociones.append(dominant_emotion)
            except:
                emociones.append('unknown')

        # Estadísticas de emociones
        if emociones:
            emotion_counts = Counter(emociones)
            most_common_emotion = emotion_counts.most_common(1)[0][0]
        else:
            most_common_emotion = "N/A"

        # Respuesta JSON
        response = {
            "conteo": conteo,
            "aglomeracion": es_aglomeracion,
            "emociones": emociones,
            "emocion_dominante": most_common_emotion,
            "confidence_promedio": np.mean([conf for _, _, _, _, conf in valid_detections]) if valid_detections else 0
        }

        logger.info(f"Predicción completada: {conteo} personas, aglomeración: {es_aglomeracion}")

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)