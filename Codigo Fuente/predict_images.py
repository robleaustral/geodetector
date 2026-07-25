import argparse
from ultralytics import YOLO


# Analiza argumentos de línea de comandos para la predicción de imágenes.
def parse_args():
    parser = argparse.ArgumentParser(description="Predicción de imágenes con YOLO")
    parser.add_argument("--model", default="../entrenamiento_modelo_m/weights/best.pt", help="Ruta al modelo YOLO")
    parser.add_argument("--input-dir", default="../dataset/images/test", help="Directorio con imágenes de prueba")
    parser.add_argument("--project", default="../runs/predict", help="Directorio de salida")
    parser.add_argument("--name", default="predict_test", help="Nombre del experimento de predicción")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    return parser.parse_args()


# Carga el modelo YOLO y ejecuta la predicción sobre un directorio de imágenes.
def main():
    args = parse_args()
    model = YOLO(args.model)
    model(args.input_dir, save=True, project=args.project, name=args.name, conf=args.conf)
    print(f"Predicción completada. Resultados en: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
