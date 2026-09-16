import argparse
import os

import cv2
import imageio
import pandas as pd
from ultralytics import YOLO


# Calcula un histograma HSV normalizado para comparar recortes de detecciones.
def compute_histogram(crop_bgr):
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def histogram_similarity(histogram_a, histogram_b):
    """Calcula la similitud entre dos histogramas mediante correlación."""
    if histogram_a is None or histogram_b is None:
        return 0.0
    return cv2.compareHist(histogram_a, histogram_b, cv2.HISTCMP_CORREL)


class Track:
    """Representa un letrero seguido a través de frames muestreados."""
    _next_id = 0

    def __init__(self, box, score, cls, class_name, second, histogram):
        self.id = Track._next_id
        Track._next_id += 1

        self.cls = cls
        self.histogram = histogram
        self.best_box = box
        self.best_score = score
        self.best_second = second
        self.class_name = class_name

        self.missed = 0

    def update(self, box, score, second, histogram):
        self.histogram = histogram
        self.missed = 0
        if score > self.best_score:
            self.best_box = box
            self.best_score = score
            self.best_second = second


class AppearanceTracker:
    """Asocia detecciones entre frames muestreados usando su apariencia."""

    def __init__(self, similarity_threshold=0.6, max_missed=1):
        self.similarity_threshold = similarity_threshold
        self.max_missed = max_missed
        self.active_tracks = []
        self.finished_tracks = []

    def update(self, detections, second):
        """
        detections: lista de tuplas (box, score, cls, class_name, histogram)
        second: segundo del video al que corresponde este frame muestreado
        """
        matched_track_ids = set()
        matched_det_idx = set()

        # Empareja cada track con la detección más similar de su misma clase.
        for t_idx, track in enumerate(self.active_tracks):
            best_sim = 0.0
            best_d_idx = -1
            for d_idx, (box, score, cls, class_name, hist) in enumerate(detections):
                if d_idx in matched_det_idx or cls != track.cls:
                    continue
                sim = histogram_similarity(track.histogram, hist)
                if sim > best_sim:
                    best_sim = sim
                    best_d_idx = d_idx

            if best_sim >= self.similarity_threshold and best_d_idx != -1:
                box, score, cls, class_name, hist = detections[best_d_idx]
                track.update(box, score, second, hist)
                matched_track_ids.add(t_idx)
                matched_det_idx.add(best_d_idx)

        still_active = []
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_track_ids:
                track.missed += 1
                if track.missed > self.max_missed:
                    self.finished_tracks.append(track)
                    continue
            still_active.append(track)
        self.active_tracks = still_active

        for d_idx, (box, score, cls, class_name, hist) in enumerate(detections):
            if d_idx not in matched_det_idx:
                self.active_tracks.append(Track(box, score, cls, class_name, second, hist))

    def finalize(self):
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks = []
        return self.finished_tracks


# Extrae las detecciones válidas y sus histogramas desde un resultado de YOLO.
def extract_detections(result, frame_bgr, class_names, detection_threshold):
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        if score < detection_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        histogram = compute_histogram(crop)
        detections.append(((x1, y1, x2, y2), float(score), int(cls), class_names[int(cls)], histogram))
    return detections


# Dibuja las detecciones sobre el frame y devuelve el resultado anotado.
def draw_detections(frame_bgr, detections):
    for box, score, cls, class_name, histogram in detections:
        x1, y1, x2, y2 = box
        label = f"{class_name} {score:.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            frame_bgr,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame_bgr,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
    return frame_bgr


# Convierte los tracks finalizados en un DataFrame con su mejor detección.
def tracks_to_dataframe(tracks):
    rows = []
    for track in tracks:
        x1, y1, x2, y2 = track.best_box
        rows.append({
            "track_id": track.id,
            "second": track.best_second,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "clase": track.class_name,
            "score": track.best_score,
        })
    columns = ["track_id", "second", "x1", "y1", "x2", "y2", "clase", "score"]
    return pd.DataFrame(rows, columns=columns)


# Procesa un video, guarda frames anotados y exporta las detecciones a CSV.
def process_video(
    model_path,
    video_path,
    output_dir,
    csv_path,
    model_confidence=0.1,
    detection_threshold=0.5,
    seconds_per_frame=1,
    similarity_threshold=0.6,
    max_missed=1,
):
    model = YOLO(model_path)
    print("Modelo cargado")

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    frame_step = max(1, int(fps * seconds_per_frame))
    print(f"Procesando cada {frame_step} frames (~{seconds_per_frame} segundo(s))")

    os.makedirs(output_dir, exist_ok=True)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    tracker = AppearanceTracker(similarity_threshold, max_missed)
    saved_count = 0

    for frame_index, frame in enumerate(reader):
        if frame_index % frame_step != 0:
            continue

        second = int(frame_index / fps)
        print(f"Procesando frame {frame_index} (segundo {second})")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = model(frame_bgr, conf=model_confidence)[0]
        detections = extract_detections(result, frame_bgr, model.names, detection_threshold)
        tracker.update(detections, second)

        if detections:
            frame_bgr = draw_detections(frame_bgr, detections)
            output_path = os.path.join(output_dir, f"frame_{second}.jpg")
            cv2.imwrite(output_path, frame_bgr)
            saved_count += 1
            print(f"Guardado frame segundo {second}: {output_path}")

    final_tracks = tracker.finalize()
    tracks_to_dataframe(final_tracks).to_csv(csv_path, index=False)
    print(f"CSV de detecciones guardado en: {csv_path}")
    print(f"Procesamiento terminado. Se guardaron {saved_count} frames con predicciones.")
    print(f"Letreros únicos (tracks) detectados: {len(final_tracks)}")


# Analiza los argumentos de línea de comandos para la predicción de video.
def parse_args():
    parser = argparse.ArgumentParser(description="Predicción de letreros en videos con YOLO")
    parser.add_argument("--model", default="../MODELO/best.pt", help="Ruta al modelo YOLO")
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta del video de entrada",
    )
    parser.add_argument(
        "--output-dir",
        default="../detecciones/frames",
        help="Directorio para frames anotados",
    )
    parser.add_argument(
        "--csv",
        default="../detecciones/csv/detecciones_video.csv",
        help="Archivo CSV de detecciones",
    )
    parser.add_argument("--model-conf", type=float, default=0.1, help="Confianza mínima para YOLO")
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="Confianza mínima guardada")
    parser.add_argument("--seconds-per-frame", type=float, default=1, help="Intervalo de muestreo en segundos")
    parser.add_argument("--similarity-threshold", type=float, default=0.6, help="Similitud mínima entre tracks")
    parser.add_argument("--max-missed", type=int, default=1, help="Frames muestreados sin match permitidos")
    return parser.parse_args()


# Ejecuta la predicción de video usando la configuración indicada por el usuario.
def main():
    args = parse_args()
    process_video(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output_dir,
        csv_path=args.csv,
        model_confidence=args.model_conf,
        detection_threshold=args.detection_threshold,
        seconds_per_frame=args.seconds_per_frame,
        similarity_threshold=args.similarity_threshold,
        max_missed=args.max_missed,
    )


if __name__ == "__main__":
    main()