"""
run.py — Air Canvas (Tracking & UI)
====================================
Modul Air Canvas untuk proyek "Hangul Character Recognition from
Air-Writing Gestures Using Traditional Computer Vision" — Group 10.

Dikembangkan oleh: David Christian Golden Mahaviro (2802501306)

CHANGELOG v2:
  - FIX: PEACE butuh minimum stroke sebelum bisa trigger (anti-prematur)
  - FIX: PEACE cooldown 45 frame agar tidak multi-trigger saat ditahan
  - FIX: save_canvas_snapshot menyimpan snapshot SEBELUM clear, bukan sesudah
  - NEW: Fitur multi-kata — tulis kata → PEACE → tulis kata → PEACE → FIST gabung
  - NEW: Word buffer ditampilkan di toolbar bawah
  - NEW: FIST = gabungkan semua kata yang sudah dikumpulkan → tampilkan hasil akhir
"""

import os
import sys
import site
site.addsitedir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env_baru', 'Lib', 'site-packages'))

import time
import math
import threading
from datetime import datetime
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
import mediapipe as mp
import joblib
from skimage.feature import hog


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 0 — KONFIGURASI GLOBAL                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CAMERA_INDEX = 0
FRAME_W      = 1280
FRAME_H      = 720

_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(_BASE_DIR, "model_saved", "svm_hangul_model.pkl")
CLASS_NAMES_PATH = os.path.join(_BASE_DIR, "data", "processed", "class_names.npy")
OUTPUT_DIR       = os.path.join(_BASE_DIR, "captures")
WINDOW_NAME      = "Hangul Air Canvas  |  Q=Quit  Z=Undo  C=Clear  S=Save"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tuning gestur ──────────────────────────────────────────────────────────
# Jumlah frame PEACE harus konsisten sebelum trigger (anti-jitter gestur)
PEACE_CONFIRM_FRAMES = 12   # ~0.4 detik @30fps — naikkan jika masih prematur
# Cooldown setelah satu trigger PEACE (frame) — cegah double-trigger
PEACE_COOLDOWN_FRAMES = 45  # ~1.5 detik @30fps
# Jumlah frame FIST harus konsisten sebelum trigger gabung kata
FIST_CONFIRM_FRAMES  = 20   # ~0.67 detik @30fps
# Minimum jumlah stroke di kanvas agar PEACE boleh trigger klasifikasi
MIN_STROKES_TO_CLASSIFY = 2


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — PREPROCESSING + HOG (Wesley + Hasan pipeline)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

IMG_SIZE            = 64
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM      = 'L2-Hys'


def preprocess_canvas_for_classification(canvas_bgr: np.ndarray):
    """
    Snapshot kanvas → vektor fitur HOG (pipeline Wesley + Hasan).
    Returns None jika kanvas kosong.
    """
    gray   = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return None

    # Auto-crop bounding box + padding
    pad = 6
    x, y, w, h = cv2.boundingRect(coords)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(canvas_bgr.shape[1], x + w + pad), min(canvas_bgr.shape[0], y + h + pad)
    cropped = gray[y1:y2, x1:x2]

    # Otsu binarization
    _, binarized = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize + normalize
    resized    = cv2.resize(binarized, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    normalized = resized.astype(np.float32) / 255.0

    # HOG features
    features = hog(
        normalized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False
    )
    return features


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — SVM CLASSIFIER                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SVMClassifier:
    def __init__(self, model_path: str, class_names_path: str):
        self._model       = None
        self._class_names = []
        self._load_ok     = False
        self._load(model_path, class_names_path)
        self.state      = "IDLE"   # IDLE | RUNNING | DONE | ERROR
        self.label      = ""
        self.confidence = ""
        self.error      = ""
        self._thread    = None

    def _load(self, model_path, class_names_path):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
            self._model = joblib.load(model_path)
            print(f"[SVM] ✅ Model dimuat: {model_path}")
        except Exception as e:
            print(f"[SVM] ❌ Gagal memuat model: {e}"); return
        try:
            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f"class_names.npy tidak ditemukan: {class_names_path}")
            self._class_names = np.load(class_names_path, allow_pickle=True).tolist()
            print(f"[SVM] ✅ {len(self._class_names)} kelas dimuat.")
        except Exception as e:
            print(f"[SVM] ⚠️  {e}")
        self._load_ok = True

    def is_ready(self):
        return self._load_ok and self._model is not None

    def submit(self, canvas_bgr: np.ndarray):
        """Kirim kanvas untuk klasifikasi asinkron."""
        if self.state == "RUNNING":
            return
        self.state = "RUNNING"; self.label = ""; self.confidence = ""; self.error = ""
        self._thread = threading.Thread(target=self._run, args=(canvas_bgr.copy(),), daemon=True)
        self._thread.start()

    def reset(self):
        self.state = "IDLE"; self.label = ""; self.confidence = ""; self.error = ""

    def _run(self, canvas_bgr):
        try:
            if not self.is_ready():
                self.error = "Model belum dimuat."; self.state = "ERROR"; return
            features = preprocess_canvas_for_classification(canvas_bgr)
            if features is None:
                self.error = "Kanvas kosong."; self.state = "ERROR"; return
            features_2d = features.reshape(1, -1)
            pred_idx    = self._model.predict(features_2d)[0]
            try:
                decision    = self._model.decision_function(features_2d)
                conf_raw    = np.max(decision)
                conf_pct    = 1.0 / (1.0 + np.exp(-conf_raw * 0.5))
                self.confidence = f"{conf_pct * 100:.1f}%"
            except Exception:
                self.confidence = "N/A"
            if self._class_names and int(pred_idx) < len(self._class_names):
                self.label = str(self._class_names[int(pred_idx)])
            else:
                self.label = str(pred_idx)
            self.state = "DONE"
        except Exception as e:
            self.error = f"Error: {e}"; self.state = "ERROR"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — GESTURE ENGINE                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Gesture(Enum):
    INDEX_ONLY = auto()
    PEACE      = auto()
    OPEN_HAND  = auto()
    FIST       = auto()
    IDLE       = auto()


@dataclass
class HandData:
    gesture:       Gesture
    index_tip:     tuple
    raw_landmarks: object


class GestureEngine:
    FINGERTIP_IDS  = [4, 8, 12, 16, 20]
    FINGER_PIP_IDS = [3, 6, 10, 14, 18]

    def __init__(self, max_hands=1):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self.hands     = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.80,
            min_tracking_confidence=0.75,
        )

    def process_frame(self, rgb_frame):
        results   = self.hands.process(rgb_frame)
        hand_list = []
        if not results.multi_hand_landmarks:
            return hand_list
        h, w = rgb_frame.shape[:2]
        for lm, hi in zip(results.multi_hand_landmarks, results.multi_handedness):
            label   = hi.classification[0].label
            fingers = self._fingers_up(lm, label)
            gesture = self._classify(fingers)
            tip     = lm.landmark[8]
            hand_list.append(HandData(
                gesture=gesture,
                index_tip=(int(tip.x * w), int(tip.y * h)),
                raw_landmarks=lm
            ))
        return hand_list

    def draw_landmarks(self, frame, hand_data):
        self._mp_draw.draw_landmarks(
            frame, hand_data.raw_landmarks, self._mp_hands.HAND_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 220, 120), thickness=2, circle_radius=3),
            self._mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2),
        )

    def _fingers_up(self, landmarks, hand_label):
        lm = landmarks.landmark
        fingers = []
        if hand_label == 'Right':
            fingers.append(1 if lm[4].x < lm[3].x else 0)
        else:
            fingers.append(1 if lm[4].x > lm[3].x else 0)
        for tip_id, pip_id in zip(self.FINGERTIP_IDS[1:], self.FINGER_PIP_IDS[1:]):
            fingers.append(1 if lm[tip_id].y < lm[pip_id].y else 0)
        return fingers

    def _classify(self, fingers):
        thumb, index, middle, ring, pinky = fingers
        extended = sum(fingers[1:])
        if extended >= 4:
            return Gesture.OPEN_HAND
        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            return Gesture.PEACE
        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return Gesture.INDEX_ONLY
        if extended == 0 and thumb == 0:
            return Gesture.FIST
        return Gesture.IDLE


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — CANVAS ENGINE                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Stroke:
    def __init__(self, points, color, thickness):
        self.points = points; self.color = color; self.thickness = thickness


class CanvasEngine:
    EMA_ALPHA      = 0.55
    MAX_JUMP_PX    = 120
    BASE_THICKNESS = 8
    SPEED_SCALE    = 0.45
    SPEED_NORM     = 60.0

    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self._strokes        = []
        self._current_pts    = []
        self._current_color  = (255, 255, 255)
        self._base_thickness = self.BASE_THICKNESS
        self._ema_x = None
        self._ema_y = None
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def stroke_count(self):
        """Jumlah stroke yang sudah selesai (bukan yang sedang digambar)."""
        return len(self._strokes)

    def smooth(self, x, y):
        if self._ema_x is None:
            self._ema_x, self._ema_y = float(x), float(y)
        else:
            self._ema_x = self.EMA_ALPHA * x + (1 - self.EMA_ALPHA) * self._ema_x
            self._ema_y = self.EMA_ALPHA * y + (1 - self.EMA_ALPHA) * self._ema_y
        return int(self._ema_x), int(self._ema_y)

    def begin_stroke(self, x, y):
        self._current_pts = [(x, y)]

    def continue_stroke(self, x, y):
        if not self._current_pts:
            self.begin_stroke(x, y); return x, y
        px, py = self._current_pts[-1]
        if math.hypot(x - px, y - py) > self.MAX_JUMP_PX:
            return x, y
        speed = math.hypot(x - px, y - py)
        thick = max(2, int(self._base_thickness * (1.0 - self.SPEED_SCALE * min(speed / self.SPEED_NORM, 1.0))))
        cv2.line(self._canvas, (px, py), (x, y), self._current_color, thick)
        self._current_pts.append((x, y))
        return x, y

    def end_stroke(self):
        if len(self._current_pts) > 1:
            self._strokes.append(Stroke(list(self._current_pts), self._current_color, self._base_thickness))
        self._current_pts = []
        self._ema_x = self._ema_y = None

    def erase(self, x, y, radius=30):
        cv2.circle(self._canvas, (x, y), radius, (0, 0, 0), -1)
        self._strokes = [s for s in self._strokes
                         if not all(math.hypot(px - x, py - y) <= radius for px, py in s.points)]

    def undo(self):
        if self._strokes:
            self._strokes.pop(); self._rebuild()

    def clear(self):
        self._strokes.clear(); self._current_pts = []
        self._canvas[:] = 0; self._ema_x = self._ema_y = None

    def set_color(self, bgr):   self._current_color  = bgr
    def set_thickness(self, t): self._base_thickness = max(2, min(40, t))

    @property
    def thickness(self): return self._base_thickness

    def composite(self, bgr_frame):
        gray     = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        return cv2.add(
            cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask_inv),
            cv2.bitwise_and(self._canvas, self._canvas, mask=mask)
        )

    def get_canvas_snapshot(self): return self._canvas.copy()

    def has_content(self): return bool(self._strokes) or bool(self._current_pts)

    def _rebuild(self):
        self._canvas[:] = 0
        for s in self._strokes:
            for i in range(1, len(s.points)):
                px, py = s.points[i-1]; cx, cy = s.points[i]
                speed = math.hypot(cx - px, cy - py)
                thick = max(2, int(s.thickness * (1.0 - self.SPEED_SCALE * min(speed / self.SPEED_NORM, 1.0))))
                cv2.line(self._canvas, (px, py), (cx, cy), s.color, thick)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — UI RENDERER                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PALETTE = [
    ("White",  (255, 255, 255)),
    ("Red",    (  0,   0, 220)),
    ("Orange", (  0, 140, 255)),
    ("Yellow", (  0, 220, 220)),
    ("Green",  ( 30, 200,  60)),
    ("Cyan",   (220, 200,   0)),
    ("Blue",   (230,  80,   0)),
    ("Purple", (200,   0, 180)),
    ("Eraser", (  0,   0,   0)),
]
ERASER_IDX   = len(PALETTE) - 1
TOOLBAR_H    = 90
SWATCH_SZ    = 52
SWATCH_PAD   = 10
DWELL_FRAMES = 18


class UIRenderer:
    def __init__(self, fw, fh):
        self.fw = fw; self.fh = fh
        self.selected_palette_idx = 0
        self._dwell_target = None
        self._dwell_count  = 0
        self._frame_count  = 0
        self._swatch_zones = self._compute_swatch_zones()
        self._undo_zone    = self._btn(0)
        self._clear_zone   = self._btn(1)
        self._save_zone    = self._btn(2)

    @property
    def current_color(self): return PALETTE[self.selected_palette_idx][1]
    @property
    def toolbar_height(self): return TOOLBAR_H

    def draw_toolbar(self, frame):
        self._frame_count += 1
        cv2.rectangle(frame, (0, 0), (self.fw, TOOLBAR_H), (20, 20, 28), -1)
        cv2.line(frame, (0, TOOLBAR_H), (self.fw, TOOLBAR_H), (60, 60, 80), 1)
        for i, (x1, y1, x2, y2) in enumerate(self._swatch_zones):
            name, bgr = PALETTE[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
            if name == "Eraser":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(frame, "X", (x1+17, y2-13), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,150,150), 2)
            if i == self.selected_palette_idx:
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 230, 120), 3)
        for (x1,y1,x2,y2), lbl, dk, br in [
            (self._undo_zone,  "UNDO",  (120,80,0),    (180,120,0)),
            (self._clear_zone, "CLEAR", (150,0,0),     (220,40,40)),
            (self._save_zone,  "SAVE",  (0,80,160),    (0,140,220)),
        ]:
            cv2.rectangle(frame, (x1,y1), (x2,y2), dk, -1)
            tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(frame, lbl, (x1+((x2-x1)-tw)//2, y1+((y2-y1)+th)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, br, 1, cv2.LINE_AA)

    def draw_cursor(self, frame, cx, cy, gesture_name, color, thickness):
        if cy < TOOLBAR_H: return
        if gesture_name == "INDEX_ONLY":
            cv2.circle(frame, (cx,cy), thickness+2, color, -1)
            cv2.circle(frame, (cx,cy), thickness+4, (255,255,255), 1)
        elif gesture_name == "PEACE":
            cv2.circle(frame, (cx,cy), 12, (0,230,255), 2)
            cv2.putText(frame, "KLASIFIKASI", (cx-40,cy+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,230,255), 1)
        elif gesture_name == "OPEN_HAND":
            cv2.circle(frame, (cx,cy), 32, (100,100,255), 2)
            cv2.putText(frame, "HAPUS", (cx-22,cy+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)
        elif gesture_name == "FIST":
            cv2.circle(frame, (cx,cy), 18, (0,180,255), 2)
            cv2.putText(frame, "GABUNG KATA", (cx-42,cy+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1)

    def draw_status(self, frame, mode, thickness, word_buffer):
        """Status bar bawah — tampilkan mode + kata-kata yang sudah terkumpul."""
        words_str = "  |  Kata terkumpul: [" + "  ".join(word_buffer) + "]" if word_buffer else ""
        text = (f" Mode: {mode}  |  Brush: {thickness}px"
                f"  |  Telunjuk=Tulis  DuaJari=Klasifikasi  TanganBuka=Hapus  Kepalan=GabungKata"
                f"{words_str}")
        cv2.putText(frame, text, (8, self.fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 160), 1)

    def draw_peace_progress(self, frame, fx, fy, count, max_count):
        """Arc melingkar di ujung jari saat PEACE sedang diconfirm."""
        if count <= 0 or fy < TOOLBAR_H: return
        end_angle = int(360 * count / max_count)
        cv2.ellipse(frame, (fx, fy), (22, 22), -90, 0, end_angle, (0, 230, 255), 3)

    def draw_fist_progress(self, frame, fx, fy, count, max_count):
        """Arc oranye saat FIST sedang diconfirm."""
        if count <= 0 or fy < TOOLBAR_H: return
        end_angle = int(360 * count / max_count)
        cv2.ellipse(frame, (fx, fy), (28, 28), -90, 0, end_angle, (0, 180, 255), 3)

    def update_dwell(self, fx, fy):
        zone = self._hit_test(fx, fy)
        if zone is None: self._dwell_target = None; self._dwell_count = 0; return None
        if zone != self._dwell_target: self._dwell_target = zone; self._dwell_count = 0; return None
        self._dwell_count += 1
        if self._dwell_count >= DWELL_FRAMES:
            self._dwell_count = 0; self._dwell_target = None; return zone
        return None

    def draw_dwell_progress(self, frame, fx, fy):
        if not self._dwell_target or not self._dwell_count: return
        c = self._zone_center(self._dwell_target)
        if c:
            cv2.ellipse(frame, c, (28,28), -90, 0, int(360*self._dwell_count/DWELL_FRAMES), (0,230,120), 3)

    def draw_word_buffer_panel(self, frame, word_buffer):
        """Panel kecil di bawah toolbar yang menampilkan kata-kata terkumpul."""
        if not word_buffer: return
        panel_y1 = TOOLBAR_H + 2
        panel_y2 = TOOLBAR_H + 36
        cv2.rectangle(frame, (0, panel_y1), (self.fw, panel_y2), (15, 15, 30), -1)
        words_display = "  +  ".join(word_buffer)
        cv2.putText(frame, f"Buffer kata:  {words_display}  → Kepal tangan untuk gabung",
                    (12, panel_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 120), 1, cv2.LINE_AA)

    def draw_result_overlay(self, frame, classifier, word_buffer, final_sentence=""):
        """
        Panel overlay hasil klasifikasi.
        Jika final_sentence terisi → tampilkan kalimat gabungan dari FIST.
        Jika kosong → tampilkan satu kata terakhir dari PEACE.
        """
        import math as _math
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = 40, h // 5, w - 40, h - 60
        ov = frame.copy()
        cv2.rectangle(ov, (px1, py1), (px2, py2), (10, 10, 20), -1)
        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (60, 200, 120), 2)

        if classifier.state == "RUNNING":
            dots = "." * (1 + (self._frame_count // 10) % 3)
            g = int(100 + 120 * abs(_math.sin(self._frame_count * 0.1)))
            cv2.putText(frame, f"Menganalisis{dots}", (px1+40, py1+80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, g, 80), 2)
            return

        if classifier.state in ("DONE", "ERROR"):
            title = "Hasil Gabungan Kalimat" if final_sentence else "Hasil Klasifikasi Karakter"
            cv2.putText(frame, title, (px1+30, py1+50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 230, 120), 2)
            cy = py1 + 110

            if final_sentence:
                # Tampilkan kalimat gabungan
                cv2.putText(frame, "Kalimat gabungan:", (px1+30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120,120,140), 1)
                cy += 44
                for line in self._wrap(final_sentence, 50):
                    cv2.putText(frame, line, (px1+50, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 180), 2, cv2.LINE_AA)
                    cy += 46
                # Juga tampilkan kata-kata individual di bawahnya
                cy += 10
                cv2.putText(frame, f"Kata-kata: {' + '.join(word_buffer)}", (px1+30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,200), 1)
            else:
                if classifier.state == "DONE":
                    cv2.putText(frame, "Karakter terdeteksi:", (px1+30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120,120,140), 1)
                    cy += 44
                    cv2.putText(frame, classifier.label, (px1+50, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 1.5, (200, 255, 200), 2, cv2.LINE_AA)
                    cy += 50
                    cv2.putText(frame, f"Confidence: {classifier.confidence}", (px1+30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,200,255), 1)
                    cy += 30
                    # Tampilkan juga kata buffer saat ini
                    if word_buffer:
                        cv2.putText(frame, f"Buffer: {' + '.join(word_buffer)}  (Kepal=gabung)",
                                    (px1+30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,200,120), 1)
                else:
                    cv2.putText(frame, f"Error: {classifier.error}", (px1+30, cy+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,220), 1)

            cv2.putText(frame, "R = tutup  |  S = simpan  |  C = reset semua",
                        (px1+30, py2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100,100,130), 1)

    def _compute_swatch_zones(self):
        zones, m = [], 12
        y1 = (TOOLBAR_H - SWATCH_SZ) // 2; y2 = y1 + SWATCH_SZ
        for i in range(len(PALETTE)):
            x1 = m + i * (SWATCH_SZ + SWATCH_PAD)
            zones.append((x1, y1, x1 + SWATCH_SZ, y2))
        return zones

    def _btn(self, idx):
        bw, bh, m = 90, 54, 12
        y1 = (TOOLBAR_H - bh) // 2; x2 = self.fw - m - idx * (bw + 10)
        return (x2 - bw, y1, x2, y1 + bh)

    def _hit_test(self, fx, fy):
        if fy > TOOLBAR_H: return None
        for i, (x1,y1,x2,y2) in enumerate(self._swatch_zones):
            if x1 <= fx <= x2 and y1 <= fy <= y2: return f"color_{i}"
        for zone, name in [(self._undo_zone,"undo"),(self._clear_zone,"clear"),(self._save_zone,"save")]:
            x1,y1,x2,y2 = zone
            if x1 <= fx <= x2 and y1 <= fy <= y2: return name
        return None

    def _zone_center(self, zone_id):
        if zone_id.startswith("color_"):
            x1,y1,x2,y2 = self._swatch_zones[int(zone_id.split("_")[1])]
            return ((x1+x2)//2, (y1+y2)//2)
        m = {"undo":self._undo_zone,"clear":self._clear_zone,"save":self._save_zone}
        if zone_id in m:
            x1,y1,x2,y2 = m[zone_id]; return ((x1+x2)//2, (y1+y2)//2)
        return None

    @staticmethod
    def _wrap(text, n):
        words = text.split(); lines, line = [], ""
        for w in words:
            if len(line)+len(w)+1 <= n: line += ("" if not line else " ")+w
            else:
                if line: lines.append(line)
                line = w
        if line: lines.append(line)
        return lines


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — HELPER SAVE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_snapshot(image: np.ndarray, prefix="canvas"):
    """
    Simpan image (BGR np.ndarray) ke folder captures/.
    Menerima image yang sudah di-snapshot SEBELUM clear — tidak lagi hitam.
    """
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{ts}.png")
    cv2.imwrite(path, image)
    print(f"[SAVE] ✅ Tersimpan: {path}")
    return path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — MAIN LOOP                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 62)
    print("  Hangul Air Canvas v2 — Group 10 Computer Vision Project")
    print("=" * 62)

    gesture_engine = GestureEngine(max_hands=1)
    canvas_engine  = CanvasEngine(FRAME_W, FRAME_H)
    ui             = UIRenderer(FRAME_W, FRAME_H)
    classifier     = SVMClassifier(MODEL_PATH, CLASS_NAMES_PATH)

    if not classifier.is_ready():
        print(f"\n⚠️  Model SVM belum siap. Path: {MODEL_PATH}\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"❌ Kamera index {CAMERA_INDEX} tidak bisa dibuka.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_W, FRAME_H)

    print("\n  Gestur:")
    print("    ☝️  Satu jari           → TULIS")
    print("    ✌️  Dua jari (tahan)    → KLASIFIKASI karakter, simpan ke buffer kata")
    print("    ✊  Kepalan (tahan)      → GABUNGKAN semua kata di buffer")
    print("    🖐  Tangan buka         → HAPUS")
    print("  Keyboard: Q=Keluar  R=Tutup result  Z=Undo  C=Clear  S=Save\n")

    # ── State variabel ──────────────────────────────────────────────────────
    show_result     = False
    was_drawing     = False
    mode_label      = "HOVER"

    # Gesture confirmation counters (cegah trigger dari 1 frame saja)
    peace_counter   = 0   # naik tiap frame PEACE terdeteksi, reset jika bukan PEACE
    peace_cooldown  = 0   # countdown setelah trigger, selama ini PEACE diabaikan
    fist_counter    = 0   # naik tiap frame FIST terdeteksi

    # Multi-kata buffer: list of (label_string, snapshot_bgr)
    word_buffer        = []   # label kata-kata yang sudah dikumpulkan
    word_snapshots     = []   # snapshot kanvas tiap kata (untuk save)
    final_sentence     = ""   # hasil gabungan setelah FIST
    last_snapshot      = None # snapshot terakhir yang di-save (fix foto hitam)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05); continue

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_list = gesture_engine.process_frame(rgb)

            current_gesture = Gesture.IDLE
            fx, fy = FRAME_W // 2, FRAME_H // 2  # default jika tidak ada tangan

            if hand_list:
                hand            = hand_list[0]
                current_gesture = hand.gesture
                fx, fy          = hand.index_tip

                gesture_engine.draw_landmarks(frame, hand)

                # Dwell toolbar (hanya saat tidak ada overlay)
                if not show_result:
                    action = ui.update_dwell(fx, fy)
                    if action:
                        if action.startswith("color_"):
                            idx = int(action.split("_")[1])
                            ui.selected_palette_idx = idx
                            if PALETTE[idx][0] != "Eraser":
                                canvas_engine.set_color(PALETTE[idx][1])
                        elif action == "undo":  canvas_engine.undo()
                        elif action == "clear":
                            canvas_engine.clear(); classifier.reset()
                            word_buffer.clear(); word_snapshots.clear()
                            final_sentence = ""; show_result = False
                        elif action == "save" and last_snapshot is not None:
                            save_snapshot(last_snapshot)
                    ui.draw_dwell_progress(frame, fx, fy)

                is_eraser = (ui.selected_palette_idx == ERASER_IDX)

                # ── Kurangi cooldown setiap frame ────────────────────────────
                if peace_cooldown > 0:
                    peace_cooldown -= 1

                # ── INDEX_ONLY: MENULIS ──────────────────────────────────────
                if current_gesture == Gesture.INDEX_ONLY and not show_result:
                    peace_counter = 0; fist_counter = 0
                    sx, sy = canvas_engine.smooth(fx, fy)
                    if fy > ui.toolbar_height:
                        if not was_drawing:
                            canvas_engine.begin_stroke(sx, sy)
                        if is_eraser: canvas_engine.erase(sx, sy, radius=28)
                        else:         canvas_engine.continue_stroke(sx, sy)
                    was_drawing = True
                    mode_label  = "HAPUS" if is_eraser else "TULIS"

                # ── OPEN_HAND: ERASER ────────────────────────────────────────
                elif current_gesture == Gesture.OPEN_HAND and not show_result:
                    peace_counter = 0; fist_counter = 0
                    sx, sy = canvas_engine.smooth(fx, fy)
                    if fy > ui.toolbar_height:
                        canvas_engine.erase(sx, sy, radius=40)
                    if was_drawing: canvas_engine.end_stroke(); was_drawing = False
                    mode_label = "HAPUS (Open Hand)"

                # ── PEACE: KONFIRMASI → KLASIFIKASI → SIMPAN KE BUFFER ───────
                elif current_gesture == Gesture.PEACE and not show_result:
                    if was_drawing: canvas_engine.end_stroke(); was_drawing = False
                    fist_counter = 0

                    # Naikkan counter hanya jika cooldown sudah habis DAN
                    # kanvas punya cukup stroke
                    enough_strokes = canvas_engine.stroke_count() >= MIN_STROKES_TO_CLASSIFY
                    if peace_cooldown == 0 and enough_strokes:
                        peace_counter += 1
                    else:
                        peace_counter = 0

                    # Tampilkan progress arc di ujung jari
                    ui.draw_peace_progress(frame, fx, fy, peace_counter, PEACE_CONFIRM_FRAMES)

                    if peace_counter >= PEACE_CONFIRM_FRAMES:
                        # ── TRIGGER KLASIFIKASI ──────────────────────────────
                        # Ambil snapshot SEBELUM clear (fix foto hitam)
                        snapshot = canvas_engine.get_canvas_snapshot()
                        last_snapshot = snapshot.copy()

                        classifier.reset()
                        classifier.submit(snapshot)

                        # Simpan snapshot per-karakter
                        save_snapshot(snapshot, prefix=f"kata{len(word_buffer)+1}")

                        # Kanvas di-clear untuk karakter berikutnya
                        canvas_engine.clear()

                        show_result    = True
                        peace_counter  = 0
                        peace_cooldown = PEACE_COOLDOWN_FRAMES
                        final_sentence = ""   # reset kalimat final, ini baru satu kata

                        # Tunggu classifier selesai (blocking max 2 detik) lalu simpan ke buffer
                        # Ini dilakukan di loop render (lihat bawah)

                    mode_label = f"KLASIFIKASI ({peace_counter}/{PEACE_CONFIRM_FRAMES})"

                # ── FIST: KONFIRMASI → GABUNGKAN SEMUA KATA ──────────────────
                elif current_gesture == Gesture.FIST:
                    peace_counter = 0
                    if was_drawing: canvas_engine.end_stroke(); was_drawing = False

                    if len(word_buffer) > 0:
                        fist_counter += 1
                    else:
                        fist_counter = 0

                    ui.draw_fist_progress(frame, fx, fy, fist_counter, FIST_CONFIRM_FRAMES)

                    if fist_counter >= FIST_CONFIRM_FRAMES:
                        # ── GABUNGKAN KATA-KATA ──────────────────────────────
                        final_sentence = " ".join(word_buffer)
                        # Tampilkan overlay dengan kalimat gabungan
                        # (gunakan state classifier.state = "DONE" yang sudah ada)
                        show_result  = True
                        fist_counter = 0
                        print(f"[FIST] Kalimat gabungan: {final_sentence}")

                    mode_label = f"GABUNG KATA ({fist_counter}/{FIST_CONFIRM_FRAMES})" if fist_counter > 0 else "HOVER"

                else:
                    if was_drawing: canvas_engine.end_stroke(); was_drawing = False
                    peace_counter = 0; fist_counter = 0
                    mode_label = "HOVER"

                ui.draw_cursor(frame, fx, fy, current_gesture.name,
                               ui.current_color, canvas_engine.thickness)
            else:
                if was_drawing: canvas_engine.end_stroke(); was_drawing = False
                peace_counter = 0; fist_counter = 0
                mode_label = "TIDAK ADA TANGAN"

            # ── Setelah classifier selesai, simpan label ke word_buffer ─────
            # (dilakukan di sini agar tidak blocking loop utama)
            if show_result and classifier.state == "DONE" and classifier.label:
                # Simpan ke buffer hanya jika belum disimpan (cek via final_sentence kosong
                # dan buffer belum mengandung label ini sebagai elemen terakhir)
                last_in_buffer = word_buffer[-1] if word_buffer else None
                if last_in_buffer != classifier.label or final_sentence:
                    # Hanya push ke buffer saat baru saja dari PEACE (final_sentence kosong)
                    if not final_sentence and (not word_buffer or word_buffer[-1] != classifier.label):
                        word_buffer.append(classifier.label)
                        print(f"[BUFFER] Kata #{len(word_buffer)}: {classifier.label} | Buffer: {word_buffer}")

            # ── Render ───────────────────────────────────────────────────────
            frame = canvas_engine.composite(frame)
            ui.draw_toolbar(frame)
            ui.draw_word_buffer_panel(frame, word_buffer)
            ui.draw_status(frame, mode_label, canvas_engine.thickness, word_buffer)

            if show_result:
                ui.draw_result_overlay(frame, classifier, word_buffer, final_sentence)

            cv2.imshow(WINDOW_NAME, frame)

            # ── Keyboard ─────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[QUIT] Menutup..."); break
            elif key == ord('r'):
                show_result = False
                if not final_sentence:
                    classifier.reset()
                # Jangan reset word_buffer saat R — user masih mau lanjut nulis kata berikutnya
            elif key == ord('z'):
                canvas_engine.undo()
            elif key == ord('c'):
                # Reset TOTAL: kanvas + buffer + hasil
                canvas_engine.clear(); classifier.reset()
                word_buffer.clear(); word_snapshots.clear()
                final_sentence = ""; show_result = False
                print("[CLEAR] Semua di-reset.")
            elif key == ord('s'):
                if last_snapshot is not None:
                    save_snapshot(last_snapshot)
                else:
                    save_snapshot(canvas_engine.get_canvas_snapshot())

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Selesai.")


if __name__ == "__main__":
    main()