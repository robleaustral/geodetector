import argparse
from ultralytics import YOLO


# Analiza argumentos de línea de comandos para la configuración de entrenamiento.
def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo YOLO")
    parser.add_argument("--data", default="../dataset/data.yaml", help="Ruta al archivo YAML de datos")
    parser.add_argument("--weights", default="yolo11m.pt", help="Pesos iniciales de YOLO")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--project", default="../", help="Directorio de salida del proyecto")
    parser.add_argument("--name", default="entrenamiento_modelo_m", help="Nombre de la carpeta del experimento")
    parser.add_argument("--patience", type=int, default=5, help="Paciencia de early stopping")
    parser.add_argument("--device", default="0", help="Dispositivo de entrenamiento")
    return parser.parse_args()


# Ejecuta el entrenamiento del modelo YOLO usando los argumentos proporcionados.
def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        plots=True,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
