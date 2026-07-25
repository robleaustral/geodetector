import argparse
import os

import cv2
import imageio
import numpy as np
import pandas as pd
from ultralytics import YOLO


# Analiza argumentos de línea de comandos para la predicción en video.
def parse_args():
    parser = argparse.ArgumentParser(description="Predicción de video y extracción de frames")
    parser.add_argument("--video-path", required=True, help="Ruta del video de entrada")
    parser.add_argument("--model", default="../MODELO/best.pt", help="Ruta al modelo YOLO")
    parser.add_argument("--output-dir", default="../detecciones/frames", help="Directorio base de salida")
    parser.add_argument("--csv-output", default="../detecciones/csv", help="Directorio para CSV de detecciones")
    parser.add_argument("--frame-step-sec", type=float, default=1.0, help="Intervalo en segundos entre frames guardados")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral mínimo de confianza")
    return parser.parse_args()


# Procesa el video, extrae frames de detección y guarda resultados en CSV.
def main():
    args = parse_args()
    model = YOLO(args.model)
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    reader = imageio.get_reader(args.video_path)
    fps = reader.get_meta_data()["fps"]
    frame_step = max(1, int(fps * args.frame_step_sec))
    output_dir = os.path.join(args.output_dir, f"frames_{video_name}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.csv_output, exist_ok=True)
    records = []
    for frame_index, frame in enumerate(reader):
        if frame_index % frame_step != 0:
            continue
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model(frame_bgr, conf=args.conf)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        image_records = []
        for box, score, cls in zip(boxes, scores, classes):
            if score < args.conf:
                continue
            x1, y1, x2, y2 = map(int, box)
            second = int(frame_index / fps)
            class_name = model.names.get(cls, str(int(cls)))
            image_records.append({
                "frame": second,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": class_name,
                "score": float(score)
            })
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - text_h - baseline), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(frame_bgr, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        if image_records:
            second = int(frame_index / fps)
            output_path = os.path.join(output_dir, f"frame_{second}.jpg")
            cv2.imwrite(output_path, frame_bgr)
            records.extend(image_records)
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.csv_output, f"detecciones_{video_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Procesamiento terminado. Frames guardados en {output_dir}")
    print(f"CSV generado en {csv_path}")


if __name__ == "__main__":
    main()
