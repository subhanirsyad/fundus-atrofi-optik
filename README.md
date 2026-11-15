# Klasifikasi atrofi optik pada citra fundus

Web skripsi: Swin-UNet (segmentasi optic disc) → ROI → ResNet-18 (Normal / Atrofi Parsial / Atrofi Total) → Grad-CAM.
Bobot diunduh otomatis dari repo model Hugging Face [`Subhaaannn/SwinUnet`](https://huggingface.co/Subhaaannn/SwinUnet)
saat aplikasi pertama kali dibuka, jadi tidak disimpan di repo ini.

## Jalankan lokal

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Online (Streamlit Community Cloud)

1. Login di https://share.streamlit.io pakai akun GitHub.
2. Create app → Deploy a public app from GitHub → repo `fundus-atrofi-optik`, branch `main`, file `streamlit_app.py`.
3. Advanced settings → Python 3.11 → Deploy.

## Isi

| File | Isi |
|---|---|
| `streamlit_app.py` | Halaman Beranda dan Klasifikasi |
| `pipeline.py` | Arsitektur model, pra-proses, ROI, Grad-CAM (disalin dari notebook skripsi) |
| `gaya.css`, `.streamlit/config.toml` | Tampilan |
| `app.py`, `index.html` | Versi lama HTML + FastAPI (`uvicorn app:app`) |
| `cek_pipeline.py` | Uji cepat pipeline dengan bobot acak |

Bukan alat diagnosis; dibuat untuk penelitian dan sidang skripsi.
