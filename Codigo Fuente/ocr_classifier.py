import argparse
import glob
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import unidecode
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR


RAW_TAXONOMY = {
    "Alojamiento": [
        "cabañas", "apart hotel", "hostal", "hotel", "casa", "suítes", "hostel", "hostería", "motel",
        "posada", "hospedaje", "departamento", "deptos", "condominio", "apartamentos", "residencial",
        "quincho"
    ],
    "Alimentación": [
        "restaurante", "menú", "waffle", "waffles", "restobar", "parrilladas", "cookies", "heladería",
        "cocina chilena", "cocina peruana", "cocina argentina", "gastronomía", "pastas", "licorería",
        "chochoca", "once", "onces", "minimarket", "market", "almacén", "cafetería", "café", "coffee",
        "cocina", "kiosko", "chocolatería", "fonda", "miel", "kuchen", "rotisería", "sopaipillas",
        "cócteles", "picada", "panadería", "comida rápida", "comida casera", "sandwichería", "pizzería",
        "platos caseros", "pub", "almuerzos", "cena", "productos orgánicos", "empanadas", "tortillas",
        "churros", "jugos naturales", "jugos tropicales", "food truck", "hand rolls", "huevos",
        "lomo a lo pobre", "asados", "pastel de choclo", "mote con huesillo", "cordero al palo",
        "humitas", "sushi", "cazuela", "desayunos", "pescado frito", "hamburguesería", "quesos",
        "pan amasado", "pan", "pollo a las brasas", "pollo", "colaciones", "mercado", "supermercado",
        "minimercado", "paila marina"
    ],
    "Recreativas": [
        "rafting", "cabalgatas", "paseos en caballo", "mountain bike", "canopy", "gimnasio",
        "centro de eventos", "kayak", "golf", "yoga", "sendero", "galería", "paseos", "paseos en bote",
        "pesca deportiva", "festival", "laguna", "surf", "trekking", "escalada", "ski", "safari", "playa",
        "playa grande", "tour", "vivero", "rio", "snowboard", "canchas de tenis", "canchas de futbol",
        "canchas de padel", "museo", "centro artesanal", "fiestas costumbristas", "termas", "parques",
        "artesanías", "feria", "museo", "plaza", "cascadas", "espectáculo", "club", "expediciones",
        "casino", "enjoy", "camping", "hidrospeed", "agencia de turismo", "juegos de mesa", "reserva natural",
        "miradores", "lounge", "jardin botanico", "flores de madera", "biblioteca"
    ]
}

NORMALIZED_TAXONOMY = {
    category: [unidecode.unidecode(term).lower().replace(" ", "").strip() for term in terms]
    for category, terms in RAW_TAXONOMY.items()
}

ocr_engine = PaddleOCR(use_angle_cls=True, lang="es")


# Normaliza el texto removiendo acentos, espacios y pasando a minúsculas.
def normalize_text(text):
    return unidecode.unidecode(text).lower().replace(" ", "").strip()


# Preprocesa la imagen para OCR, convirtiéndola a escala de grises y aplicando CLAHE.
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# Ejecuta PaddleOCR sobre una imagen preprocesada y devuelve texto y confianza.
def run_ocr(image_path):
    preprocessed = preprocess_image(image_path)
    if preprocessed is None:
        return "", 0.0, []
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, preprocessed)
    try:
        results = ocr_engine.ocr(temp_path, det=False, rec=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    if not results:
        return "", 0.0, []
    detected_words = [line[1][0] for line in results]
    confidences = [float(line[1][1]) for line in results]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return " ".join(detected_words), average_confidence, detected_words


# Clasifica el texto detectado en una categoría usando búsqueda difusa y umbrales.
def classify_text(detected_words, confidence, confidence_threshold=0.6, fuzzy_score_threshold=80):
    if confidence < confidence_threshold:
        return "Otro"
    text = normalize_text("".join(detected_words))
    for category, normalized_terms in NORMALIZED_TAXONOMY.items():
        for term in normalized_terms:
            if term in text:
                return category
            if fuzz.partial_ratio(term, text) >= fuzzy_score_threshold:
                return category
    return "Otro"


# Procesa todas las imágenes de un directorio, ejecuta OCR y clasifica resultados.
def process_images(input_dir, output_csv=None):
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {input_dir}")
    records = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        text, confidence, words = run_ocr(image_path)
        category = classify_text(words, confidence)
        records.append({
            "image": filename,
            "text": text,
            "confidence": confidence,
            "category": category
        })
    df = pd.DataFrame(records)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Archivo CSV generado: {output_csv}")
    return df


# Analiza argumentos de línea de comandos para el módulo OCR y clasificación.
def parse_args():
    parser = argparse.ArgumentParser(description="Clasificador OCR y taxonomía")
    parser.add_argument("--input-dir", required=True, help="Directorio de imágenes")
    parser.add_argument("--output-csv", help="Archivo CSV de salida opcional")
    return parser.parse_args()


# Función principal que ejecuta el flujo de OCR y guarda resultados opcionales.
def main():
    args = parse_args()
    process_images(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()
