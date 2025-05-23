import torch
import pandas as pd
from pathlib import Path
import os
import sys

sys.path.append('yolov5')  # tambahkan path YOLOv5
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

def get_latest_run_folder(base="runs/detect"):
    folders = list(Path(base).glob("exp*"))
    if not folders:
        return None
    latest = max(folders, key=os.path.getctime)
    return latest

def detect_and_save(source, model_path='best.pt', conf_threshold=0.25):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.conf = conf_threshold

    results = model(source)
    results.save()

    data_list = []
    for i, pred in enumerate(results.pred):
        filename = results.files[i]
        if pred is not None and len(pred):
            for *box, conf, cls in pred.tolist():
                xmin, ymin, xmax, ymax = box
                data_list.append({
                    "filename": os.path.basename(filename),
                    "label": model.names[int(cls)],
                    "confidence": round(conf, 3),
                    "xmin": round(xmin),
                    "ymin": round(ymin),
                    "xmax": round(xmax),
                    "ymax": round(ymax)
                })
        else:
            data_list.append({
                "filename": os.path.basename(filename),
                "label": "No detection",
                "confidence": None,
                "xmin": None,
                "ymin": None,
                "xmax": None,
                "ymax": None
            })

    df = pd.DataFrame(data_list)
    outdir = get_latest_run_folder()
    csv_path = outdir / "hasil_deteksi.csv"
    df.to_csv(csv_path, index=False)
    return outdir, csv_path
