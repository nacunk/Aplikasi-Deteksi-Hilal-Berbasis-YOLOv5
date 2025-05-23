import torch
import pandas as pd
from pathlib import Path
import os

def detect_and_save(source, model_path='best.pt', conf_threshold=0.25):
    # Load model dari torch hub (bukan dari models.common)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.conf = conf_threshold

    results = model(source)
    results.save()  # simpan hasil deteksi (gambar)

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
    outdir = Path("runs/detect/exp")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "hasil_deteksi.csv"
    df.to_csv(csv_path, index=False)
    return outdir, csv_path
