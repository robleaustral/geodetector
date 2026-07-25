import argparse
import os
from datetime import datetime

import gpxpy
import pandas as pd
from geopy.distance import geodesic
from pymediainfo import MediaInfo
import pytz


# Convierte un archivo GPX en un CSV de posicionamiento con velocidad y remuestreo.
def gpx_to_csv(gpx_path, output_csv, timezone="America/Santiago"):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    records = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                records.append({
                    "time": point.time,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No se encontraron puntos en el archivo GPX")
    df = df.sort_values(by="time").set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    speeds = [None]
    for i in range(1, len(df)):
        coord_prev = (df.iloc[i - 1]["latitude"], df.iloc[i - 1]["longitude"])
        coord_curr = (df.iloc[i]["latitude"], df.iloc[i]["longitude"])
        distance_m = geodesic(coord_prev, coord_curr).meters
        delta_seconds = (df.index[i] - df.index[i - 1]).total_seconds()
        speed_kmh = (distance_m / delta_seconds) * 3.6 if delta_seconds > 0 else None
        speeds.append(speed_kmh)
    df["speed_kmh"] = speeds
    df_resampled = df.resample("1S").interpolate()
    if df_resampled.index.tz is not None:
        tz = pytz.timezone(timezone)
        df_resampled.index = df_resampled.index.tz_convert(tz)
    df_resampled.index = df_resampled.index.tz_localize(None)
    df_resampled["speed_kmh"] = df_resampled["speed_kmh"].round(1)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_resampled.to_csv(output_csv)
    print(f"CSV de GPS generado en: {output_csv}")


# Alinea el CSV de GPS con la duración del video y calcula n_frame por segundo.
def align_video(video_path, input_csv, output_csv, timezone="America/Santiago"):
    media_info = MediaInfo.parse(video_path)
    creation_time = None
    duration_s = None
    for track in media_info.tracks:
        if track.track_type == "General":
            duration_s = float(track.duration) / 1000 if track.duration else None
            data = track.to_data()
            for key, value in data.items():
                if value and "creation" in key.lower():
                    creation_time = value
                    break
            break
    if creation_time is None or duration_s is None:
        raise ValueError("No se pudo obtener metadata completa del video")
    video_end = pd.to_datetime(creation_time)
    if video_end.tzinfo is None:
        video_end = video_end.tz_localize("UTC")
    tz = pytz.timezone(timezone)
    video_end = video_end.tz_convert(tz)
    video_start = video_end - pd.Timedelta(seconds=int(duration_s))
    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    df["match_video"] = df.index.to_series().between(video_start, video_end)
    df.loc[df["match_video"], "n_frame"] = ((df.loc[df["match_video"]].index - video_start).total_seconds().round().astype(int))
    df["n_frame"].fillna(-1, inplace=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv)
    print(f"CSV de video alineado guardado en: {output_csv}")


# Marca las filas sincronizadas que tienen detecciones de letreros en frames extraídos.
def match_detections(detections_csv, sync_csv, frames_dir, output_csv):
    df_sync = pd.read_csv(sync_csv, index_col=0, parse_dates=True)
    if "n_frame" not in df_sync.columns:
        raise KeyError("El CSV sincronizado no contiene la columna 'n_frame'")
    frame_files = os.listdir(frames_dir)
    matched_frames = {int(f.split("frame_")[1].split(".")[0]) for f in frame_files if f.startswith("frame_")}
    df_sync["match_sign"] = df_sync["n_frame"].isin(matched_frames)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_sync.to_csv(output_csv)
    print(f"CSV actualizado con coincidencias en: {output_csv}")


# Analiza los subcomandos disponibles para la sincronización GPS/video.
def parse_args():
    parser = argparse.ArgumentParser(description="Sincronización GPS y video")
    subparsers = parser.add_subparsers(dest="command", required=True)
    gpx_parser = subparsers.add_parser("gpx_to_csv")
    gpx_parser.add_argument("--gpx", required=True)
    gpx_parser.add_argument("--output", default="../sync/output/sync_gps.csv")
    gpx_parser.add_argument("--timezone", default="America/Santiago")
    align_parser = subparsers.add_parser("video_align")
    align_parser.add_argument("--video", required=True)
    align_parser.add_argument("--input-csv", required=True)
    align_parser.add_argument("--output", default="../sync/output/sync_video.csv")
    align_parser.add_argument("--timezone", default="America/Santiago")
    match_parser = subparsers.add_parser("match_detections")
    match_parser.add_argument("--detections-csv", required=True)
    match_parser.add_argument("--sync-csv", required=True)
    match_parser.add_argument("--frames-dir", required=True)
    match_parser.add_argument("--output", default="../sync/output/sync_with_detections.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "gpx_to_csv":
        gpx_to_csv(args.gpx, args.output, args.timezone)
    elif args.command == "video_align":
        align_video(args.video, args.input_csv, args.output, args.timezone)
    elif args.command == "match_detections":
        match_detections(args.detections_csv, args.sync_csv, args.frames_dir, args.output)


if __name__ == "__main__":
    main()
