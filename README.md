# Retrolens - Hand Tracking Filter & Portal

Aplikasi filter kamera interaktif menggunakan gesture tangan (hand tracking) dengan MediaPipe dan OpenCV. Anda dapat membuat "portal" dengan jari Anda yang akan menerapkan berbagai filter menarik, ditambah trail cahaya neon, kerangka tangan bercahaya, mode layar penuh, screenshot, dan perekaman video.

## Fitur

- **18 filter** di dalam portal: Mono, Dual-Tone, Pixelate, Invert, Sepia, Blur, Thermal, Sketch, Glitch, Neon, Galaxy, Rainbow, Kaleidoscope, Swirl, Bulge, VHS, Watercolor, Emboss
- **Portal animasi** dengan border glow berputar dan smoothing untuk gerakan yang lebih halus
- **Trail cahaya neon** yang mengikuti ujung jari + kerangka tangan bercahaya
- **Mode layar penuh** (filter diterapkan ke seluruh frame)
- **Screenshot** dan **rekam video** (hasil disimpan di folder `captures/` dan `recordings/`)
- **FPS meter** dan carousel filter di bagian bawah layar

## Persyaratan Sistem
- Python 3.7 atau lebih baru
- Webcam

## Cara Install dan Menjalankan

### 1. Clone Repository
Pertama, clone repository ini ke komputer Anda dan masuk ke foldernya (ganti URL dengan link repository GitHub Anda):
```bash
git clone <URL_GITHUB_ANDA>
cd <NAMA_FOLDER_REPO>
```

### 2. Buat Virtual Environment (Opsional tapi Sangat Disarankan)
Gunakan virtual environment agar dependencies (library) tidak bentrok dengan project Python lainnya di komputer Anda.
```bash
# Untuk Windows
python -m venv venv
venv\Scripts\activate

# Untuk macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install semua library Python yang dibutuhkan dengan menjalankan perintah berikut:
```bash
pip install -r requirements.txt
```

Atau jika Anda ingin menginstallnya secara manual satu per satu:
```bash
pip install opencv-python mediapipe numpy
```

### 4. Pastikan Model MediaPipe Tersedia
Aplikasi ini membutuhkan dua file model dari MediaPipe yang seharusnya sudah ada di dalam repository ini:
1. `hand_landmarker.task` (Untuk mendeteksi titik pada tangan)
2. `selfie_segmenter.tflite` (Untuk filter Galaxy / memisahkan background)

Jika file tersebut belum ada, pastikan untuk meletakkannya di dalam folder yang sama dengan file `main.py`.

### 5. Jalankan Aplikasi
Ada dua cara:

**Cara cepat (disarankan):**
```bash
# Windows: double-click run.bat, atau dari terminal:
run.bat

# macOS / Linux / Git Bash:
./run.sh
```

**Cara manual (dengan virtual environment aktif):**
```bash
# Windows
venv\Scripts\activate
python main.py

# macOS / Linux
source venv/bin/activate
python main.py
```

> **Catatan:** Jika `python` tidak dikenali di terminal Anda (mis. hanya muncul Microsoft Store di Windows), pastikan Anda sudah mengaktifkan virtual environment (`venv\Scripts\activate`), atau gunakan `run.bat` / `run.sh` yang otomatis memakai Python dari dalam `venv/`.

## Cara Penggunaan Fitur

### Gesture Tangan

- **Membuka Portal:** Pinch jempol + telunjuk dari **kedua** tangan di depan kamera (total 4 jari). Portal persegi empat terbentuk di antara keempat jari, dan efek filter muncul di dalamnya.
- **Mengganti Filter (gesture):**
  - Sentuhkan jempol + kelingking, ATAU
  - Dekatkan ujung telunjuk kedua tangan.

### Tombol Keyboard

| Tombol | Fungsi |
|---|---|
| `q` | Keluar dari aplikasi |
| `s` | Simpan screenshot (ke folder `captures/`) |
| `r` | Mulai / berhenti merekam video (ke folder `recordings/`) |
| `f` | Beralih mode portal <-> layar penuh |
| `t` | Aktifkan / nonaktifkan trail & kerangka tangan |
| `h` | Tampilkan / sembunyikan panel bantuan |
| `n` / `→` | Filter berikutnya |
| `p` / `←` | Filter sebelumnya |

**Menutup Aplikasi:** Pastikan jendela Retrolens sedang aktif (diklik), lalu tekan **`q`**.

### Hasil Screenshot & Video
- Screenshot: `captures/retrolens_<tanggal_waktu>.png`
- Video: `recordings/retrolens_<tanggal_waktu>.avi` (codec MJPG)

---
**Catatan:** Pastikan ruangan memiliki pencahayaan yang cukup agar deteksi tangan dari kamera dapat bekerja dengan optimal.
  