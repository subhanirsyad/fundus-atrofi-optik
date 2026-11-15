"""Web skripsi versi Streamlit: Beranda -> Klasifikasi. Jalankan: streamlit run streamlit_app.py"""
import base64
import math
import os
import random

import cv2
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download

import pipeline

REPO_MODEL = os.environ.get("REPO_MODEL", "Subhaaannn/SwinUnet")
IDENTITAS = "Skripsi · Subhan Irsyaduddien Alhaq · Matematika, Universitas Indonesia"
DIR = os.path.dirname(os.path.abspath(__file__))
NAMA = {"Normal": "Normal", "Parsial": "Atrofi Parsial", "Total": "Atrofi Total"}
WARNA = {"Normal": "var(--k-normal)", "Parsial": "var(--k-parsial)", "Total": "var(--k-total)"}
URUT = ["Normal", "Parsial", "Total"]

st.set_page_config(page_title="Klasifikasi Atrofi Optik", page_icon=":material/visibility:", layout="wide")


def latar_svg(seed=11):
    """Gambar garis fundus (lingkaran FOV, optic disc, pembuluh bercabang) untuk latar halaman. Dibuat sendiri, bebas lisensi."""
    rng = random.Random(seed)
    c, R, dx, dy = 450, 420, 610, 430
    garis = []

    def cabang(x, y, sudut, panjang, tebal, belok, sisa):
        pts = [(x, y)]
        for _ in range(int(panjang / 12)):
            sudut += belok + rng.uniform(-0.18, 0.18)
            x, y = x + 12 * math.cos(sudut), y + 12 * math.sin(sudut)
            if math.hypot(x - c, y - c) > R - 8:
                break
            pts.append((x, y))
        if len(pts) < 2:
            return
        garis.append('<polyline points="%s" stroke-width="%.1f"/>' % (" ".join("%.0f,%.0f" % p for p in pts), tebal))
        if sisa:
            for arah in (-1, 1):
                cabang(x, y, sudut + arah * rng.uniform(0.35, 0.7), panjang * rng.uniform(0.6, 0.8), tebal * 0.72,
                       belok * 0.5, sisa - 1)

    # arkade temporal melengkung mengitari makula (kiri disc), cabang nasal ke kanan
    for sudut, belok in ((-2.3, -0.035), (2.3, 0.035), (-2.9, -0.01), (2.9, 0.01), (-0.8, 0.0), (0.8, 0.0)):
        cabang(dx, dy, sudut, rng.uniform(170, 230), 5.0, belok, 4)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 900" fill="none" stroke="#f0cbb6" '
           'stroke-linecap="round" stroke-linejoin="round">'
           f'<circle cx="{c}" cy="{c}" r="{R}" stroke-width="1.5"/>'
           f'<circle cx="{c}" cy="{c}" r="{R + 22}" stroke-width="1" stroke-dasharray="2 8"/>'
           + "".join(garis) +
           f'<circle cx="{dx}" cy="{dy}" r="36" stroke-width="2.5" fill="#ffffff"/>'
           f'<circle cx="{dx}" cy="{dy}" r="16" stroke-width="1.2"/>'
           '<circle cx="300" cy="455" r="24" stroke-width="1" stroke-dasharray="3 5"/></svg>')
    return 'url("data:image/svg+xml;base64,%s")' % base64.b64encode(svg.encode()).decode()


@st.cache_resource(show_spinner="Menyiapkan model (sekali saja, unduh dari Hugging Face)…")
def siapkan_model():
    os.makedirs(pipeline.MODEL_DIR, exist_ok=True)
    for f in (pipeline.SWIN_FILE, pipeline.RESNET_FILE):
        if not os.path.exists(os.path.join(pipeline.MODEL_DIR, f)):
            hf_hub_download(REPO_MODEL, f, local_dir=pipeline.MODEL_DIR)   # HF_TOKEN dari env kalau repo privat
    pipeline.muat_model()
    return pipeline.GALAT_MUAT


@st.cache_data(max_entries=16, show_spinner=False)
def analisis(data):
    return pipeline.analisis(data)


def persen(v):
    return f"{v * 100:.1f}".replace(".", ",") + "%"


def pratinjau(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    s = 640 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return pipeline._jpeg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 85)


# ======================================================================
# Halaman 1: Beranda
# ======================================================================
def beranda():
    st.html("<style>.stApp { background-image: var(--ilustrasi); }</style>")   # <style> harus sendirian di st.html
    st.html(f"""
<p class="meta">{IDENTITAS}</p>
<h1 class="judul">Klasifikasi atrofi optik pada citra fundus</h1>
<p class="lead">Swin-UNet mencari optic disc, lalu ResNet-18 membaca daerah di sekitarnya dan memilih satu dari tiga kelas:
Normal, Atrofi Parsial, atau Atrofi Total. Grad-CAM menunjukkan bagian citra yang paling memengaruhi keputusan itu.</p>

<section class="bagian">
  <h2>Tiga kelas yang dibedakan</h2>
  <p>Ketiganya dibedakan dari warna dan luas pucat pada optic disc, bagian kecil yang hanya menempati sekitar 1–2% luas citra.</p>
  <div class="kelas3">
    <div class="kartu"><h3 style="color:var(--k-normal)"><i class="cincin"></i>Normal</h3>
      <p>Optic disc berwarna jingga kemerahan dengan batas yang jelas. Tidak ada daerah pucat.</p></div>
    <div class="kartu"><h3 style="color:var(--k-parsial)"><i class="cincin"></i>Atrofi Parsial</h3>
      <p>Pucat hanya di sebagian optic disc, paling sering di sisi temporal.</p></div>
    <div class="kartu"><h3 style="color:var(--k-total)"><i class="cincin"></i>Atrofi Total</h3>
      <p>Seluruh permukaan optic disc pucat sampai keputihan.</p></div>
  </div>
</section>

<section class="bagian">
  <h2>Dari citra ke prediksi</h2>
  <p>Setiap citra yang diunggah melewati enam tahap yang sama dengan eksperimen di skripsi.</p>
  <ol class="alur">
    <li><b>Potong FOV</b><span>Latar hitam di luar lingkaran fundus dibuang (ambang 10, margin 2%).</span></li>
    <li><b>CLAHE</b><span>Kontras lokal dinaikkan. Hanya dipakai sebagai masukan Swin-UNet.</span></li>
    <li><b>Segmentasi Swin-UNet</b><span>Memetakan piksel optic disc. Dilatih pada 62 citra berlabel dokter, Dice uji 0,921.</span></li>
    <li><b>Potong ROI</b><span>Kotak disc + margin 10% tiap sisi, dibuat persegi 224 px. Kalau disc tidak ketemu, dipakai daerah paling terang.</span></li>
    <li><b>Klasifikasi ResNet-18</b><span>Pretrained ImageNet, fine-tune layer3–layer4, keluaran tiga kelas.</span></li>
    <li><b>Grad-CAM</b><span>Peta panas dari layer4 di atas ROI.</span></li>
  </ol>
</section>

<section class="bagian">
  <h2>Hasil evaluasi</h2>
  <p>Validasi silang 5 fold yang dibagi per pasien, sehingga mata kiri dan kanan pasien yang sama tidak pernah terpisah ke latih dan validasi.</p>
  <div class="tabel-gulir"><table>
    <caption>3 kelas, rata-rata 5 fold ± simpangan baku. 350 citra dari 203 pasien (Normal 118, Parsial 114, Total 118), seed 42.</caption>
    <thead><tr><th>Model</th><th class="n">Akurasi</th><th class="n">F1 makro</th><th class="n">AUC</th></tr></thead>
    <tbody>
      <tr class="dipakai"><td>Swin-UNet ROI + ResNet-18<span>dipakai di web ini</span></td><td class="n">78,9% <small>± 3,3</small></td><td class="n">77,5% <small>± 4,0</small></td><td class="n">0,904 <small>± 0,011</small></td></tr>
      <tr><td>ResNet-18, fundus utuh</td><td class="n">74,0% <small>± 4,8</small></td><td class="n">73,4% <small>± 4,0</small></td><td class="n">0,884 <small>± 0,035</small></td></tr>
    </tbody>
  </table></div>
  <p class="catatan">Web ini memakai bobot dari fold 2, fold dengan akurasi validasi tertinggi (82,6%). Angka itu dipilih
  dari skornya sendiri sehingga cenderung optimis; angka yang dilaporkan tetap rata-rata 5 fold di tabel.</p>
</section>

<div class="ajakan">
  <h2>Coba dengan citra fundus</h2>
  <p>Unggah satu citra, lalu lihat kelas yang dipilih model, probabilitas tiap kelas, dan jejak pemrosesannya.</p>
</div>
""")
    if st.button("Lanjut ke klasifikasi  →", type="primary"):
        st.switch_page(HAL_KLASIFIKASI)
    st.html('<p class="kaki">Dibuat untuk penelitian dan sidang skripsi. Ini bukan alat diagnosis; keputusan klinis tetap di tangan dokter.</p>')


# ======================================================================
# Halaman 2: Klasifikasi
# ======================================================================
def klasifikasi():
    if st.button("← Beranda", type="tertiary"):
        st.switch_page(HAL_BERANDA)
    st.html(f'<p class="meta">{IDENTITAS}</p><h1 class="judul" style="font-size:clamp(1.9rem,1.3rem + 2vw,2.6rem);max-width:none">Klasifikasi citra fundus</h1>')

    galat = siapkan_model()
    if galat:
        st.error(f"Bobot model tidak ditemukan di {galat['folder']}: {', '.join(galat['hilang'])}")
        st.stop()

    kiri, kanan = st.columns([5, 6], gap="large")
    with kiri:
        berkas = st.file_uploader("Citra fundus (JPG, PNG, BMP, TIFF; maks. 25 MB)",
                                  type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        data = berkas.getvalue() if berkas else None
        src = pratinjau(data) if data else None
        isi = (f'<img src="{src}" alt="Citra yang diunggah">' if src else
               '<div><strong>Belum ada citra</strong>Pilih file di atas. Citra hanya diproses di server ini dan tidak disimpan.</div>')
        st.html(f'<div class="lensa"><div class="bulat{" terisi" if src else ""}">{isi}</div>'
                f'<p class="lensa-ket">{berkas.name if berkas else "Belum ada citra."}</p></div>')

    hasil = None
    with kanan:
        if not data:
            st.html('<p class="label-kecil">Hasil</p><h2 style="margin:0 0 10px">Belum ada citra yang dianalisis</h2>'
                    '<p style="color:var(--teks-2);max-width:56ch">Setelah citra diunggah, di sini muncul kelas yang dipilih model '
                    'dan probabilitas tiap kelas. Di bawahnya ada kontur optic disc, ROI, dan Grad-CAM.</p>')
        else:
            try:
                with st.spinner("Menganalisis: potong FOV, Swin-UNet, ROI, ResNet-18, Grad-CAM…"):
                    hasil = analisis(data)
            except ValueError as e:
                st.error(f"Citra ini tidak bisa dianalisis. {e}")
        if hasil:
            p = hasil["probabilitas"]
            k = hasil["kelas"]
            baris = "".join(
                f'<li class="{"puncak" if n == k else ""}" style="color:{WARNA[n] if n == k else "var(--teks-2)"}">'
                f'<span class="nama-k" style="color:var(--teks)"><i class="cincin" style="color:{WARNA[n]}"></i>{NAMA[n]}</span>'
                f'<span class="persen" style="color:var(--teks)">{persen(p[n])}</span>'
                f'<span class="batang"><i style="width:{p[n] * 100:.1f}%;color:{WARNA[n]}"></i></span></li>' for n in URUT)
            dua = sorted(p.values(), reverse=True)
            catatan = ""
            if hasil["sumber_roi"] != "swin":
                catatan += ('<p class="catatan">Swin-UNet tidak menemukan optic disc pada citra ini. ROI diambil dari daerah '
                            'paling terang di dalam FOV (metode cadangan), jadi prediksinya lebih lemah dari biasanya.</p>')
            if dua[0] - dua[1] < 0.15:
                catatan += (f'<p class="catatan">Selisih dua kelas teratas hanya {persen(dua[0] - dua[1])}. '
                            'Anggap hasil ini ragu.</p>')
            sumber = "ROI dari Swin-UNet" if hasil["sumber_roi"] == "swin" else "ROI cadangan"
            st.html(f'<p class="label-kecil">Prediksi model</p>'
                    f'<p class="hasil-kelas" style="color:{WARNA[k]}"><i class="cincin"></i><span style="color:var(--teks)">{NAMA[k]}</span></p>'
                    f'<p class="hasil-sub">Probabilitas <b>{persen(p[k])}</b> · {sumber} · '
                    f'{str(hasil["detik"]).replace(".", ",")} detik di CPU</p>'
                    f'<ul class="prob">{baris}</ul>{catatan}')

    if hasil:
        g = hasil["gambar"]
        st.html('<section class="bagian"><h2>Jejak pemrosesan</h2>'
                '<p>Tiga tahap antara citra asli dan prediksi. Gambar ini keluaran model untuk citra yang baru diunggah, bukan contoh.</p></section>')
        geser, _ = st.columns([1, 2])
        opasitas = geser.slider("Opasitas Grad-CAM", 0, 100, 45, format="%d%%")
        st.html(f"""
<div class="panel">
  <figure class="utama"><div class="bingkai"><img src="{g['segmentasi']}" alt="FOV + CLAHE dengan kontur optic disc dan kotak ROI"></div>
    <figcaption><span class="no">01</span><b>Segmentasi optic disc.</b> Masukan Swin-UNet (FOV + CLAHE, 224 px).&nbsp;
    <span class="legenda"><i></i>kontur disc</span> <span class="legenda"><i class="putih"></i>kotak ROI</span></figcaption></figure>
  <figure><div class="bingkai"><img src="{g['roi']}" alt="ROI optic disc 224 piksel"></div>
    <figcaption><span class="no">02</span><b>ROI.</b> Kotak disc + margin 10% tiap sisi, dari FOV tanpa CLAHE. Ini masukan ResNet-18.</figcaption></figure>
  <figure><div class="bingkai"><img src="{g['roi']}" alt=""><img class="tumpang" style="opacity:{opasitas / 100}" src="{g['gradcam']}" alt="Grad-CAM di atas ROI"></div>
    <figcaption><span class="no">03</span><b>Grad-CAM layer4.</b> Merah = daerah yang paling mendorong prediksi.</figcaption></figure>
</div>""")
    st.html('<p class="kaki">Model hanya pernah melihat citra fundus. Citra lain tetap akan diberi kelas, dan kelas itu tidak berarti apa-apa. '
            'Ini bukan alat diagnosis.</p>')


HAL_BERANDA = st.Page(beranda, title="Beranda", url_path="beranda", default=True)
HAL_KLASIFIKASI = st.Page(klasifikasi, title="Klasifikasi", url_path="klasifikasi")
with open(os.path.join(DIR, "gaya.css"), encoding="utf-8") as f:
    st.html(f"<style>:root{{--ilustrasi:{latar_svg()}}}\n{f.read()}</style>")
st.navigation([HAL_BERANDA, HAL_KLASIFIKASI], position="hidden").run()
