import torch
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import uuid
import shutil
import sys

# Tambahkan path ke folder YOLOv5 lokal
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device

def detect_and_save(source, model_path='best.pt', conf_threshold=0.25):
    # Siapkan direktori output
    result_dir = Path("runs/detect") / f"exp_{uuid.uuid4().hex[:6]}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device('')
    model = DetectMultiBackend(model_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(640, s=stride)

    # Load gambar atau video
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    data_list = []

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=conf_threshold)

        for i, det in enumerate(pred):
            p = Path(path)
            save_path = result_dir / p.name
            im0 = im0s.copy()

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    data_list.append({
                        "filename": p.name,
                        "label": names[int(cls)],
                        "confidence": float(conf),
                        "xmin": int(xyxy[0]),
                        "ymin": int(xyxy[1]),
                        "xmax": int(xyxy[2]),
                        "ymax": int(xyxy[3]),
                    })
                    # Gambar kotak deteksi pada gambar
                    from utils.plots import Annotator, colors
                    annotator = Annotator(im0, line_width=2)
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    im0 = annotator.result()
            else:
                data_list.append({
                    "filename": p.name,
                    "label": "No detection",
                    "confidence": None,
                    "xmin": None,
                    "ymin": None,
                    "xmax": None,
                    "ymax": None,
                })

            # Simpan gambar hasil deteksi
            Image.fromarray(im0).save(save_path)

    # Simpan ke CSV
    df = pd.DataFrame(data_list)
    csv_path = result_dir / "hasil_deteksi.csv"
    df.to_csv(csv_path, index=False)

    return result_dir, csv_path
