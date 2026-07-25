import os
import yaml


# Genera un YAML de configuración de dataset compatible con YOLO
# y lo guarda en la ruta especificada.
def create_dataset_yaml(output_path="dataset/data.yaml"):
    data = {
        "path": "../dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "signboard"
        }
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    print(f"Archivo YAML creado en {output_path}")


if __name__ == "__main__":
    create_dataset_yaml()
