import streamlit as st
from detect_utils import detect_and_save
from PIL import Image
import os
import uuid
import pandas as pd

st.set_page_config(page_title="Deteksi Hilal", page_icon="🌙")
st.title("🌙 Aplikasi Deteksi Hilal Berbasis YOLOv5")

uploaded_file = st.file_uploader("Unggah Gambar atau Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    # Simpan dengan nama unik
    unique_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    with open(unique_filename, 'wb') as f:
        f.write(uploaded_file.read())

    st.success("📤 File berhasil diunggah. Memulai deteksi...")

    # Jalankan deteksi
    result_dir, csv_path = detect_and_save(unique_filename)

    # Cari file gambar hasil deteksi (jpg)
    detected_images = list(result_dir.glob("*.jpg"))
    if detected_images:
        for img_path in detected_images:
            st.image(Image.open(img_path), caption=f"Hasil Deteksi: {img_path.name}")
            with open(img_path, "rb") as f:
                st.download_button("📸 Unduh Gambar Deteksi", data=f, file_name=img_path.name)
    else:
        st.warning("⚠️ Gambar hasil deteksi tidak ditemukan.")

    # Tampilkan dan unduh hasil CSV
    if csv_path.exists():
        st.dataframe(pd.read_csv(csv_path))
        with open(csv_path, "rb") as f:
            st.download_button("📥 Unduh Hasil (CSV)", data=f, file_name="hasil_deteksi.csv")
    else:
        st.warning("⚠️ File CSV tidak ditemukan.")