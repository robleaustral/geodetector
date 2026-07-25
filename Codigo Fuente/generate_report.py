import argparse
import os

import pandas as pd


# Analiza argumentos de línea de comandos para generar el reporte final.
def parse_args():
    parser = argparse.ArgumentParser(description="Generar reporte final de detecciones y OCR")
    parser.add_argument("--detections-csv", required=True)
    parser.add_argument("--sync-csv", required=True)
    parser.add_argument("--ocr-csv", required=True)
    parser.add_argument("--output-xlsx", default="reporte_final.xlsx")
    return parser.parse_args()


# Exporta el DataFrame final a un archivo Excel con formato de latitud/longitud.
# Exporta el DataFrame final a un archivo Excel y ajusta el formato numérico de coordenadas.
def export_to_excel(df, output_path):
    writer = pd.ExcelWriter(output_path, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="Reporte")
    ws = writer.sheets["Reporte"]
    number_format = "0.0000000000"
    columns = ["latitude", "longitude"]
    header = [cell.value for cell in ws[1]]
    for col_name in columns:
        if col_name in header:
            col_index = header.index(col_name) + 1
            for row in ws.iter_rows(min_row=2, min_col=col_index, max_col=col_index):
                for cell in row:
                    cell.number_format = number_format
    writer.close()


# Construye el reporte final uniendo detecciones, sincronización GPS y resultados OCR.
# Une los datos de detección, sincronización y OCR en una tabla final ordenada.
def build_report(detections_csv, sync_csv, ocr_csv):
    df_det = pd.read_csv(detections_csv)
    df_sync = pd.read_csv(sync_csv)
    df_ocr = pd.read_csv(ocr_csv)
    df_sync = df_sync[df_sync["match_letrero"] == True]
    df_merge = pd.merge(df_sync, df_det[["frame", "score"]], left_on="n_frame", right_on="frame", how="inner")
    df_ocr_clean = df_ocr.dropna(subset=["ocr_text"]).copy()
    if "frame_name" in df_ocr_clean.columns:
        df_ocr_clean["frame_number"] = df_ocr_clean["frame_name"].astype(str).str.extract(r"(\d+)").astype(float)
    else:
        df_ocr_clean["frame_number"] = pd.NA
    df_final = pd.merge(
        df_merge,
        df_ocr_clean[["frame_number", "frame_name", "video_id", "ocr_text", "confidence", "category", "semaforo"]],
        left_on="n_frame",
        right_on="frame_number",
        how="inner",
    )
    columns = [
        "video_id", "frame_name", "time", "latitude", "longitude",
        "speed_kmh", "score", "ocr_text", "confidence", "category", "semaforo"
    ]
    df_final = df_final.loc[:, [c for c in columns if c in df_final.columns]]
    return df_final.sort_values(by="time")


# Función principal que genera el reporte final y lo guarda en Excel.
def main():
    args = parse_args()
    df_final = build_report(args.detections_csv, args.sync_csv, args.ocr_csv)
    export_to_excel(df_final, args.output_xlsx)
    print(f"Reporte final guardado en: {args.output_xlsx}")


if __name__ == "__main__":
    main()
