import torch
import pandas as pd
from pathlib import Path
import os
import uuid
from PIL import Image
import cv2

def detect_and_save(source, model_path='best.pt', conf_threshold=0.25):
    # Direktori output
    result_dir = Path("runs/detect") / f"exp_{uuid.uuid4().hex[:6]}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load model dari torch.hub (tanpa folder lokal yolov5)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.conf = conf_threshold

    # Deteksi
    results = model(source)
    results.save(save_dir=result_dir)  # Simpan hasil ke direktori

    # Buat data CSV
    data_list = []
    for i, pred in enumerate(results.pred):
        filename = os.path.basename(results.files[i])
        if pred is not None and len(pred):
            for *box, conf, cls in pred.tolist():
                xmin, ymin, xmax, ymax = map(int, box)
                data_list.append({
                    "filename": filename,
                    "label": model.names[int(cls)],
                    "confidence": round(conf, 3),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })
        else:
            data_list.append({
                "filename": filename,
                "label": "No detection",
                "confidence": None,
                "xmin": None,
                "ymin": None,
                "xmax": None,
                "ymax": None
            })

    # Simpan ke CSV
    df = pd.DataFrame(data_list)
    csv_path = result_dir / "hasil_deteksi.csv"
    df.to_csv(csv_path, index=False)

    return result_dir, csv_path
