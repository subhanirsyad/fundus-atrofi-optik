"""Cek cepat alur inferensi dengan bobot ACAK dan citra fundus sintetis (bukan uji akurasi).
Jalankan: python cek_pipeline.py  -> harus berakhir dengan "OK".
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

folder = tempfile.mkdtemp()
os.environ["MODEL_DIR"] = folder
import pipeline as P  # noqa: E402  (MODEL_DIR harus di-set sebelum import)

P.muat_model()
assert P.GALAT_MUAT and len(P.GALAT_MUAT["hilang"]) == 2, "bobot kosong harus terdeteksi"

torch.save(P.SwinUnet(depths=P.DEPTHS, depths_decoder=P.DEPTHS).state_dict(), os.path.join(folder, P.SWIN_FILE))
torch.save(P.Klasifier(3).state_dict(), os.path.join(folder, P.RESNET_FILE))
P.muat_model()
assert P.GALAT_MUAT is None

img = np.zeros((900, 1100, 3), np.uint8)
cv2.circle(img, (550, 450), 420, (40, 70, 170), -1)       # retina (BGR)
cv2.circle(img, (720, 420), 70, (150, 210, 240), -1)      # disc terang
ok, buf = cv2.imencode(".png", img)
h = P.analisis(buf.tobytes())
assert h["kelas"] in P.NAMA_KELAS
assert abs(sum(h["probabilitas"].values()) - 1) < 1e-4
assert h["sumber_roi"] in ("swin", "cadangan")
assert all(v.startswith("data:image/jpeg;base64,") for v in h["gambar"].values())
try:
    P.analisis(b"bukan gambar")
    sys.exit("file rusak harus ditolak")
except ValueError:
    pass
print("OK", h["kelas"], h["sumber_roi"], h["detik"], "detik")
