"""
app.py — Flask Web Backend for Hangul Air Writing Recognition
=============================================================
Group 10 - Computer Vision Project
Connects canvas_web.html frontend → SVM model → JSON response

Usage:
    pip install flask opencv-python scikit-image scikit-learn joblib numpy
    python app.py
    Then open http://localhost:5000
"""

import os
import io
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, send_from_directory
from skimage.feature import hog

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "model_saved", "svm_hangul_model.pkl")
CLASS_NAMES_PATH= os.path.join(BASE_DIR, "data", "processed", "class_names.npy")
STATIC_DIR      = BASE_DIR   # canvas_web.html lives in the same folder

# HOG params — harus sama persis dengan training pipeline
IMG_SIZE             = 64
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)
HOG_BLOCK_NORM       = 'L2-Hys'

app = Flask(__name__, static_folder=STATIC_DIR)

# ── Load model once at startup ────────────────────────────────────────────────
print("[INIT] Loading SVM model...")
try:
    model       = joblib.load(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
    print(f"[INIT] ✅ Model loaded — {len(class_names)} classes")
    MODEL_READY = True
except Exception as e:
    print(f"[INIT] ❌ Model load failed: {e}")
    model       = None
    class_names = []
    MODEL_READY = False

# ── Romanization map (64 Hangul syllables from the dataset) ──────────────────
# Maps romanization label → Hangul character
ROMAN_TO_HANGUL = {
    "ga": "가", "na": "나", "da": "다", "ra": "라", "ma": "마",
    "ba": "바", "sa": "사", "a":  "아", "ja": "자", "cha": "차",
    "ka": "카", "ta": "타", "pa": "파", "ha": "하",
    "gya": "갸", "nya": "냐", "dya": "댜", "rya": "랴", "mya": "먀",
    "bya": "뱌", "sya": "샤", "ya": "야", "jya": "쟈", "chya": "챠",
    "kya": "캬", "tya": "탸", "pya": "퍄", "hya": "햐",
    "geo": "거", "neo": "너", "deo": "더", "reo": "러", "meo": "머",
    "beo": "버", "seo": "서", "eo": "어", "jeo": "저", "cheo": "처",
    "keo": "커", "teo": "터", "peo": "퍼", "heo": "허",
    "go": "고", "no": "노", "do": "도", "ro": "로", "mo": "모",
    "bo": "보", "so": "소", "o":  "오", "jo": "조", "cho": "초",
    "ko": "코", "to": "토", "po": "포", "ho": "호",
    "gu": "구", "nu": "누", "du": "두", "ru": "루", "mu": "무",
    "bu": "부", "su": "수", "u":  "우", "ju": "주", "chu": "추",
    "ku": "쿠", "tu": "투", "pu": "푸", "hu": "후",
}

# Simple translation hints for demo purposes
TRANSLATION_HINTS = {
    "ga": "go (verb stem)", "na": "I / me", "da": "all / everything",
    "ra": "come! (imperative)", "ma": "mom / horse", "ba": "sea / bar",
    "sa": "four / person", "a": "ah / child (suffix)", "ja": "자 (now / let's)",
    "cha": "차 (car / tea)", "ka": "ka (sound)", "ta": "ta (sound)",
    "pa": "pa / green onion", "ha": "하 (do / one)",
    "geo": "것 (thing)", "neo": "you", "deo": "더 (more)",
    "seo": "서 (at / stand)", "eo": "어 (interjection)",
    "go": "고 (and / high)", "no": "노 (no / row)",
    "do": "도 (also / degree)", "ro": "로 (by / road)",
    "mo": "모 (shape / all)", "bo": "보 (see / step)",
    "so": "소 (cow / small)", "o": "오 (come / five)",
    "jo": "조 (help / dynasty)", "cho": "초 (second / candle)",
    "ko": "코 (nose)", "ho": "호 (lake / call)",
    "gu": "구 (nine / old)", "du": "두 (two / head)",
    "mu": "무 (nothing / radish)", "bu": "부 (part / rich)",
    "su": "수 (water / number)", "u": "우 (right / cow)",
    "ju": "주 (wine / week)", "chu": "추 (autumn /추)",
    "ku": "크 (big)", "hu": "후 (after / breath)",
}


# ── Preprocessing pipeline (matches training) ─────────────────────────────────
def preprocess_image(img_bytes: bytes):
    """
    Bytes dari canvas PNG → vektor HOG features.
    Returns (features_array, preview_base64) or raises ValueError.
    """
    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Gagal decode gambar.")

    # Handle RGBA (canvas PNG bisa transparan)
    if img.shape[2] == 4:
        # Blend ke background hitam
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb   = img[:, :, :3].astype(np.float32)
        img   = (rgb * alpha).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cek apakah ada konten
    coords = cv2.findNonZero(gray)
    if coords is None:
        raise ValueError("Canvas kosong — gambar karakter dulu!")

    # Auto-crop + padding
    pad = 6
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - pad);   y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    cropped = gray[y1:y2, x1:x2]

    # Binarisasi Otsu
    _, binarized = cv2.threshold(cropped, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize & normalize
    resized    = cv2.resize(binarized, (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_LANCZOS4)
    normalized = resized.astype(np.float32) / 255.0

    # HOG
    features = hog(
        normalized,
        orientations    = HOG_ORIENTATIONS,
        pixels_per_cell = HOG_PIXELS_PER_CELL,
        cells_per_block = HOG_CELLS_PER_BLOCK,
        block_norm      = HOG_BLOCK_NORM,
        visualize       = False,
    )
    return features


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve canvas_web.html"""
    return send_from_directory(STATIC_DIR, "canvas_web.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Body: multipart form-data dengan field 'file' (PNG gambar dari canvas)
    Returns: JSON { success, result: { korean, romanization, translation, confidence } }
              atau { success: false, error: "..." }
    """
    if not MODEL_READY:
        return jsonify({"success": False,
                        "error": "Model SVM belum dimuat. Cek path model_saved/svm_hangul_model.pkl"}), 503

    if "file" not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file yang dikirim."}), 400

    file_bytes = request.files["file"].read()
    if not file_bytes:
        return jsonify({"success": False, "error": "File kosong."}), 400

    try:
        features = preprocess_image(file_bytes)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        return jsonify({"success": False, "error": f"Preprocessing error: {e}"}), 500

    # Klasifikasi
    try:
        features_2d = features.reshape(1, -1)
        pred_idx    = int(model.predict(features_2d)[0])

        # Confidence via decision function
        try:
            decision = model.decision_function(features_2d)
            conf_raw = float(np.max(decision))
            confidence = float(1.0 / (1.0 + np.exp(-conf_raw * 0.5)))
        except Exception:
            confidence = None

        # Label
        if class_names and pred_idx < len(class_names):
            romanization = str(class_names[pred_idx])
        else:
            romanization = str(pred_idx)

        # Hangul karakter
        roman_lower = romanization.lower().strip()
        korean_char = ROMAN_TO_HANGUL.get(roman_lower, romanization)

        # Terjemahan hint
        translation = TRANSLATION_HINTS.get(roman_lower, "—")

        return jsonify({
            "success": True,
            "result": {
                "korean":        korean_char,
                "romanization":  romanization,
                "translation":   translation,
                "confidence":    round(confidence, 4) if confidence is not None else None,
                "pred_index":    pred_idx,
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Klasifikasi gagal: {e}"}), 500


@app.route("/status")
def status():
    """Health check"""
    return jsonify({
        "model_ready": MODEL_READY,
        "num_classes": len(class_names),
        "classes_sample": class_names[:10] if class_names else [],
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Hangul Air Writing — Flask Web Server")
    print("="*55)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Classes: {CLASS_NAMES_PATH}")
    print(f"  Status : {'✅ Ready' if MODEL_READY else '❌ Model not found'}")
    print(f"\n  Open   : http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
