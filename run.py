# ── Paste your FREE Gemini key here ──────────────────────────────────────── #
GEMINI_API_KEY = "AIzaSyCWl4V8wm2KwYISJOZ5EYSv_YGp85ryWLM"

# Leave blank if using Tesseract
# GEMINI_API_KEY = ""

import os
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

if GEMINI_API_KEY and not GEMINI_API_KEY.endswith("HERE"):
    print("✅ Gemini key set — using Google Gemini analyzer.")
else:
    print("⚠️  No Gemini key — make sure to use Tesseract in Step 5.")

import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


class Gesture(Enum):
    INDEX_ONLY = auto()
    PEACE      = auto()
    OPEN_HAND  = auto()
    FIST       = auto()
    PINCH      = auto()
    IDLE       = auto()


@dataclass
class HandData:
    gesture:       Gesture
    index_tip:     tuple
    raw_landmarks: object


class GestureEngine:
    def __init__(self, max_hands=1):
        self._mp_hands     = mp.solutions.hands
        self._mp_draw      = mp.solutions.drawing_utils
        self.hands         = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.80,
            min_tracking_confidence=0.75,
        )
        self.FINGERTIP_IDS  = [4, 8, 12, 16, 20]
        self.FINGER_PIP_IDS = [3, 6, 10, 14, 18]

    def process_frame(self, rgb_frame):
        results = self.hands.process(rgb_frame)
        hand_data_list = []
        if not results.multi_hand_landmarks:
            return hand_data_list
        h, w = rgb_frame.shape[:2]
        for landmarks, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            hand_label = handedness_info.classification[0].label
            fingers    = self._fingers_up(landmarks, hand_label)
            gesture    = self._classify_gesture(fingers, landmarks, w, h)
            tip = landmarks.landmark[8]
            ix  = int(tip.x * w)
            iy  = int(tip.y * h)
            hand_data_list.append(HandData(
                gesture=gesture, index_tip=(ix, iy), raw_landmarks=landmarks
            ))
        return hand_data_list

    def draw_landmarks(self, bgr_frame, hand_data):
        self._mp_draw.draw_landmarks(
            bgr_frame, hand_data.raw_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
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

    def _classify_gesture(self, fingers, landmarks, w, h):
        thumb, index, middle, ring, pinky = fingers
        extended_count = sum(fingers[1:])
        if extended_count == 0 and thumb == 0:
            return Gesture.FIST
        if extended_count >= 4:
            return Gesture.OPEN_HAND
        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            return Gesture.PEACE
        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return Gesture.INDEX_ONLY
        if index == 1 and thumb == 1 and middle == 0:
            lm   = landmarks.landmark
            dx   = (lm[4].x - lm[8].x) * w
            dy   = (lm[4].y - lm[8].y) * h
            if np.hypot(dx, dy) < 40:
                return Gesture.PINCH
        return Gesture.IDLE

print("✅ GestureEngine ready.")

import cv2
from collections import deque


class Stroke:
    def __init__(self, points, color, thickness):
        self.points    = points
        self.color     = color
        self.thickness = thickness


class CanvasEngine:
    def __init__(self, width, height):
        self.width   = width
        self.height  = height
        self._strokes        = []
        self._current_pts    = []
        self._current_color  = (255, 255, 255)
        self._base_thickness = 8
        self._smooth_buf     = deque(maxlen=6)
        self._canvas         = np.zeros((height, width, 3), dtype=np.uint8)

    def smooth(self, x, y):
        self._smooth_buf.append((x, y))
        if len(self._smooth_buf) < 2: return x, y
        alpha = 0.55
        sx, sy = self._smooth_buf[-2]
        return int(alpha*x+(1-alpha)*sx), int(alpha*y+(1-alpha)*sy)

    def begin_stroke(self, x, y):
        self._current_pts = [(x, y)]

    def continue_stroke(self, x, y):
        if not self._current_pts:
            self.begin_stroke(x, y); return x, y
        px, py = self._current_pts[-1]
        speed  = np.hypot(x-px, y-py)
        thick  = max(2, int(self._base_thickness*(1.0-0.45*np.clip(speed/60,0,1))))
        cv2.line(self._canvas, (px,py), (x,y), self._current_color, thick)
        self._current_pts.append((x, y))
        return x, y

    def end_stroke(self):
        if len(self._current_pts) > 1:
            self._strokes.append(Stroke(
                list(self._current_pts), self._current_color, self._base_thickness))
        self._current_pts = []
        self._smooth_buf.clear()

    def erase(self, x, y, radius=30):
        cv2.circle(self._canvas, (x,y), radius, (0,0,0), -1)
        self._strokes = [s for s in self._strokes
                         if np.any(np.hypot(
                             np.array(s.points)[:,0]-x,
                             np.array(s.points)[:,1]-y) > radius)]

    def undo(self):
        if self._strokes:
            self._strokes.pop()
            self._rebuild_canvas()

    def clear(self):
        self._strokes.clear(); self._current_pts = []; self._canvas[:] = 0

    def set_color(self, bgr): self._current_color = bgr
    def set_thickness(self, t): self._base_thickness = max(2, min(40, t))

    @property
    def thickness(self): return self._base_thickness

    def composite(self, bgr_frame):
        gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        return cv2.add(
            cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask_inv),
            cv2.bitwise_and(self._canvas, self._canvas, mask=mask))

    def get_canvas_snapshot(self): return self._canvas.copy()
    def has_content(self): return bool(self._strokes) or bool(self._current_pts)

    def _rebuild_canvas(self):
        self._canvas[:] = 0
        for s in self._strokes:
            for i in range(1, len(s.points)):
                px,py = s.points[i-1]; cx,cy = s.points[i]
                speed = np.hypot(cx-px,cy-py)
                thick = max(2, int(s.thickness*(1.0-0.45*np.clip(speed/60,0,1))))
                cv2.line(self._canvas,(px,py),(cx,cy),s.color,thick)

print("✅ CanvasEngine ready.")

import base64, json, threading, io
from enum import Enum, auto
from PIL import Image


class AnalysisState(Enum):
    IDLE = auto(); RUNNING = auto(); DONE = auto(); ERROR = auto()


class AnalysisResult:
    def __init__(self):
        self.korean=self.romanization=self.translation=self.notes=self.raw=""


# ══════════════════════════════════════════════════════════════════════════════
# OPTION A — Google Gemini (free tier)
# Get key: https://aistudio.google.com/app/apikey  (no credit card)
# ══════════════════════════════════════════════════════════════════════════════
class GeminiKoreanAnalyzer:
    PROMPT = """You are a Korean OCR assistant. The image shows air-drawn Korean characters
(drawn with a finger in front of a webcam — strokes may be imperfect).
Identify every Korean character, provide romanization (Revised Romanization), and English translation.
Respond ONLY with a JSON object, no markdown:
{"korean":"","romanization":"","translation":"","confidence":"high|medium|low","notes":""}"""

    def __init__(self, api_key):
        from google import genai
        self._client = genai.Client(api_key=api_key)   # free tier model
        self.state  = AnalysisState.IDLE
        self.result = AnalysisResult()
        self._thread = None

    def submit(self, canvas):
        if self.state == AnalysisState.RUNNING: return
        self.state  = AnalysisState.RUNNING
        self.result = AnalysisResult()
        prepped = self._preprocess(canvas)
        if prepped is None:
            self.result.notes = "Canvas is empty."; self.state = AnalysisState.DONE; return
        self._thread = threading.Thread(target=self._run, args=(prepped,), daemon=True)
        self._thread.start()

    def reset(self): self.state = AnalysisState.IDLE; self.result = AnalysisResult()

    def _preprocess(self, canvas):
        gray   = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None: return None
        x,y,w,h = cv2.boundingRect(coords)
        pad=30
        x1,y1=max(0,x-pad),max(0,y-pad)
        x2,y2=min(canvas.shape[1],x+w+pad),min(canvas.shape[0],y+h+pad)
        crop = canvas[y1:y2,x1:x2]
        white_bg = np.full_like(crop,255)
        gray_c=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray_c,10,255,cv2.THRESH_BINARY)
        result_img=cv2.add(
            cv2.bitwise_and(white_bg,white_bg,mask=cv2.bitwise_not(mask)),
            cv2.bitwise_and(crop,crop,mask=mask))
        ch,cw=result_img.shape[:2]
        if max(cw,ch)<512:
            scale=512/max(cw,ch)
            result_img=cv2.resize(result_img,(int(cw*scale),int(ch*scale)),
                                  interpolation=cv2.INTER_LANCZOS4)
        rgb=cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)
        buf=io.BytesIO(); Image.fromarray(rgb).save(buf,format="PNG")
        return buf.getvalue()   # raw bytes for Gemini

    def _run(self, img_bytes):
        try:
            from google import genai
            img_part = {"mime_type":"image/png","data":img_bytes}
            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[self.PROMPT, img_part]
            )
            raw = response.text.strip()
            self.result.raw = raw
            if raw.startswith("```"):
                raw=raw.split("```")[1]
                if raw.startswith("json"): raw=raw[4:]
            data=json.loads(raw)
            self.result.korean       = data.get("korean","")
            self.result.romanization = data.get("romanization","")
            self.result.translation  = data.get("translation","")
            self.result.notes        = data.get("notes","")
            self.state = AnalysisState.DONE
        except json.JSONDecodeError:
            self.result.translation = self.result.raw; self.state = AnalysisState.DONE
        except Exception as e:
            self.result.notes = f"Gemini error: {e}"; self.state = AnalysisState.ERROR


# ══════════════════════════════════════════════════════════════════════════════
# OPTION B — Tesseract OCR (100% local, zero API calls)
# Install binary first (see Step 2 instructions above)
# ══════════════════════════════════════════════════════════════════════════════
class TesseractKoreanAnalyzer:
    # Windows default path — change if yours is different
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def __init__(self):
        import pytesseract
        import platform
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_PATH
        self._tess  = pytesseract
        self.state  = AnalysisState.IDLE
        self.result = AnalysisResult()
        self._thread = None

    def submit(self, canvas):
        if self.state == AnalysisState.RUNNING: return
        self.state  = AnalysisState.RUNNING
        self.result = AnalysisResult()
        prepped = self._preprocess(canvas)
        if prepped is None:
            self.result.notes = "Canvas is empty."; self.state = AnalysisState.DONE; return
        self._thread = threading.Thread(target=self._run, args=(prepped,), daemon=True)
        self._thread.start()

    def reset(self): self.state = AnalysisState.IDLE; self.result = AnalysisResult()

    def _preprocess(self, canvas):
        gray   = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None: return None
        x,y,w,h = cv2.boundingRect(coords)
        pad=40
        x1,y1=max(0,x-pad),max(0,y-pad)
        x2,y2=min(canvas.shape[1],x+w+pad),min(canvas.shape[0],y+h+pad)
        crop = canvas[y1:y2,x1:x2]
        white_bg=np.full_like(crop,255)
        gray_c=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray_c,10,255,cv2.THRESH_BINARY)
        result_img=cv2.add(
            cv2.bitwise_and(white_bg,white_bg,mask=cv2.bitwise_not(mask)),
            cv2.bitwise_and(crop,crop,mask=mask))
        ch,cw=result_img.shape[:2]
        if max(cw,ch)<512:
            scale=512/max(cw,ch)
            result_img=cv2.resize(result_img,(int(cw*scale),int(ch*scale)),
                                  interpolation=cv2.INTER_LANCZOS4)
        return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    def _run(self, img_array):
        try:
            pil_img = Image.fromarray(img_array)
            # kor = Korean | eng = fallback English
            text = self._tess.image_to_string(pil_img, lang="kor+eng").strip()
            if text:
                self.result.korean      = text
                self.result.translation = "(Tesseract: translation not available — copy Korean text into Google Translate)"
                self.result.notes       = "Tesseract OCR — accuracy depends on stroke clarity"
            else:
                self.result.notes = "No text detected. Try writing larger and slower."
            self.state = AnalysisState.DONE
        except Exception as e:
            self.result.notes = f"Tesseract error: {e}"; self.state = AnalysisState.ERROR


# ══════════════════════════════════════════════════════════════════════════════
# CHOOSE YOUR ANALYZER HERE
# ══════════════════════════════════════════════════════════════════════════════
import os

GEMINI_KEY = os.environ.get("GEMINI_API_KEY","")

if GEMINI_KEY and not GEMINI_KEY.endswith("HERE"):
    analyzer_backend = GeminiKoreanAnalyzer(api_key=GEMINI_KEY)
    print("✅ Using Google Gemini analyzer (free tier)")
else:
    analyzer_backend = TesseractKoreanAnalyzer()
    print("✅ Using Tesseract OCR analyzer (local)")

import math

PALETTE = [
    ("White",  (255,255,255)),
    ("Red",    (0,  0,  220)),
    ("Orange", (0,  130,255)),
    ("Yellow", (0,  220,220)),
    ("Green",  (30, 200, 60)),
    ("Cyan",   (220,200,  0)),
    ("Blue",   (230, 80,  0)),
    ("Purple", (200,  0,180)),
    ("Eraser", (0,   0,  0)),
]
TOOLBAR_H=90; SWATCH_SZ=52; SWATCH_PAD=10; DWELL_FRAMES=18


class UIRenderer:
    def __init__(self, fw, fh):
        self.fw=fw; self.fh=fh
        self.selected_palette_idx=0
        self._dwell_target=None; self._dwell_count=0; self._frame_count=0
        self._swatch_zones=self._compute_swatch_zones()
        self._analyze_zone=self._btn(0); self._undo_zone=self._btn(1)
        self._clear_zone=self._btn(2);  self._save_zone=self._btn(3)

    @property
    def current_color(self): return PALETTE[self.selected_palette_idx][1]
    @property
    def toolbar_height(self): return TOOLBAR_H

    def draw_toolbar(self, frame):
        self._frame_count+=1
        cv2.rectangle(frame,(0,0),(self.fw,TOOLBAR_H),(20,20,28),-1)
        cv2.line(frame,(0,TOOLBAR_H),(self.fw,TOOLBAR_H),(60,60,80),1)
        for i,(x1,y1,x2,y2) in enumerate(self._swatch_zones):
            name,bgr=PALETTE[i]
            cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,-1)
            if name=="Eraser":
                cv2.rectangle(frame,(x1,y1),(x2,y2),(80,80,80),1)
                cv2.putText(frame,"X",(x1+17,y2-13),cv2.FONT_HERSHEY_SIMPLEX,0.8,(150,150,150),2)
            if i==self.selected_palette_idx:
                cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(0,230,120),3)
        for (x1,y1,x2,y2),lbl,dk,br in [
            (self._analyze_zone,"ANALYZE",(0,160,60),(0,220,100)),
            (self._undo_zone,"UNDO",(120,80,0),(180,120,0)),
            (self._clear_zone,"CLEAR",(150,0,0),(220,40,40)),
            (self._save_zone,"SAVE",(0,80,160),(0,140,220))]:
            cv2.rectangle(frame,(x1,y1),(x2,y2),dk,-1)
            tw,th=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)[0]
            cv2.putText(frame,lbl,(x1+((x2-x1)-tw)//2,y1+((y2-y1)+th)//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,br,1,cv2.LINE_AA)

    def draw_cursor(self,frame,cx,cy,gname,color,thickness):
        if cy<TOOLBAR_H: return
        if gname=="INDEX_ONLY":
            cv2.circle(frame,(cx,cy),thickness+2,color,-1)
            cv2.circle(frame,(cx,cy),thickness+4,(255,255,255),1)
        elif gname=="PEACE": cv2.circle(frame,(cx,cy),10,(0,230,255),2)
        elif gname=="OPEN_HAND":
            cv2.circle(frame,(cx,cy),32,(100,100,255),2)
            cv2.putText(frame,"ERASE",(cx-22,cy+45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,255),1)
        elif gname=="FIST":
            cv2.circle(frame,(cx,cy),18,(0,200,100),2)
            cv2.putText(frame,"ANALYZE",(cx-35,cy+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,100),1)

    def draw_status(self,frame,mode,thickness):
        text=(f" {mode}  |  Color: {PALETTE[self.selected_palette_idx][0]}  |  Brush: {thickness}px  |  "
              f"Index:Draw  Peace:Hover  OpenHand:Erase  Fist:Analyze")
        cv2.putText(frame,text,(8,self.fh-12),cv2.FONT_HERSHEY_SIMPLEX,0.42,(140,140,160),1)

    def update_dwell(self,fx,fy):
        zone=self._hit(fx,fy)
        if zone is None: self._dwell_target=None; self._dwell_count=0; return None
        if zone!=self._dwell_target: self._dwell_target=zone; self._dwell_count=0; return None
        self._dwell_count+=1
        if self._dwell_count>=DWELL_FRAMES:
            self._dwell_count=0; self._dwell_target=None; return zone
        return None

    def draw_dwell_progress(self,frame,fx,fy):
        if not self._dwell_target or not self._dwell_count: return
        c=self._zone_center(self._dwell_target)
        if c: cv2.ellipse(frame,c,(28,28),-90,0,int(360*self._dwell_count/DWELL_FRAMES),(0,230,120),3)

    def draw_result_overlay(self,frame,result,state_name):
        h,w=frame.shape[:2]
        px1,py1,px2,py2=40,h//5,w-40,h-60
        ov=frame.copy(); cv2.rectangle(ov,(px1,py1),(px2,py2),(10,10,20),-1)
        cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
        cv2.rectangle(frame,(px1,py1),(px2,py2),(60,200,120),2)
        if state_name=="RUNNING":
            dots="."*(1+(self._frame_count//10)%3)
            g=int(100+120*abs(math.sin(self._frame_count*0.1)))
            cv2.putText(frame,f"Analyzing{dots}",(px1+40,py1+80),cv2.FONT_HERSHEY_DUPLEX,1.1,(0,g,80),2)
            cv2.putText(frame,"Reading your Korean strokes...",(px1+40,py1+130),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(130,130,160),1)
            return
        if state_name in("DONE","ERROR"):
            cv2.putText(frame,"Korean Analysis",(px1+30,py1+50),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,230,120),2)
            y=py1+110
            for lbl,val,col in [("Korean",result.korean,(200,200,255)),
                                 ("Romanization",result.romanization,(180,255,200)),
                                 ("Translation",result.translation,(255,255,180)),
                                 ("Notes",result.notes,(160,160,180))]:
                if not val: continue
                cv2.putText(frame,f"{lbl}:",(px1+30,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,120,140),1)
                for line in self._wrap(val,55):
                    y+=30
                    cv2.putText(frame,line,(px1+50,y),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,1,cv2.LINE_AA)
                y+=18
            cv2.putText(frame,"Press R to close  |  S to save",(px1+30,py2-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(100,100,130),1)

    def _compute_swatch_zones(self):
        z,m=[],12; y1=(TOOLBAR_H-SWATCH_SZ)//2; y2=y1+SWATCH_SZ
        for i in range(len(PALETTE)):
            x1=m+i*(SWATCH_SZ+SWATCH_PAD); z.append((x1,y1,x1+SWATCH_SZ,y2))
        return z

    def _btn(self,idx):
        bw,bh,m=90,54,12; y1=(TOOLBAR_H-bh)//2; x2=self.fw-m-idx*(bw+10); return(x2-bw,y1,x2,y1+bh)

    def _hit(self,fx,fy):
        if fy>TOOLBAR_H: return None
        for i,(x1,y1,x2,y2) in enumerate(self._swatch_zones):
            if x1<=fx<=x2 and y1<=fy<=y2: return f"color_{i}"
        for z,n in[(self._analyze_zone,"analyze"),(self._undo_zone,"undo"),
                   (self._clear_zone,"clear"),(self._save_zone,"save")]:
            x1,y1,x2,y2=z
            if x1<=fx<=x2 and y1<=fy<=y2: return n
        return None

    def _zone_center(self,zone_id):
        if zone_id.startswith("color_"):
            x1,y1,x2,y2=self._swatch_zones[int(zone_id.split("_")[1])]; return((x1+x2)//2,(y1+y2)//2)
        m={"analyze":self._analyze_zone,"undo":self._undo_zone,"clear":self._clear_zone,"save":self._save_zone}
        if zone_id in m: x1,y1,x2,y2=m[zone_id]; return((x1+x2)//2,(y1+y2)//2)

    @staticmethod
    def _wrap(text,n):
        words=text.split(); lines,line=[],""
        for w in words:
            if len(line)+len(w)+1<=n: line+=("" if not line else " ")+w
            else:
                if line: lines.append(line)
                line=w
        if line: lines.append(line)
        return lines

print("✅ UIRenderer ready.")

import os, time
from datetime import datetime

CAMERA_INDEX = 0   # change to 1 or 2 if camera doesn't open
FRAME_W      = 1280
FRAME_H      = 720
ERASER_IDX   = len(PALETTE) - 1
OUTPUT_DIR   = "captures"
WINDOW_NAME  = "Korean Air Canvas  |  Q to quit"

os.makedirs(OUTPUT_DIR, exist_ok=True)

gesture_engine = GestureEngine(max_hands=1)
canvas_engine  = CanvasEngine(FRAME_W, FRAME_H)
ui             = UIRenderer(FRAME_W, FRAME_H)
analyzer       = analyzer_backend   # set in Step 5

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}. Try 1 or 2.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, FRAME_W, FRAME_H)

show_result=False; was_drawing=False; fist_triggered=False; mode_label="HOVER"

def save_canvas():
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    path=os.path.join(OUTPUT_DIR,f"canvas_{ts}.png")
    cv2.imwrite(path,canvas_engine.get_canvas_snapshot())
    print(f"[SAVE] → {path}")

def handle_action(action):
    global show_result
    if action.startswith("color_"):
        idx=int(action.split("_")[1]); ui.selected_palette_idx=idx
        if PALETTE[idx][0]!="Eraser": canvas_engine.set_color(PALETTE[idx][1])
    elif action=="analyze" and canvas_engine.has_content():
        analyzer.submit(canvas_engine.get_canvas_snapshot()); show_result=True
    elif action=="undo": canvas_engine.undo()
    elif action=="clear": canvas_engine.clear(); analyzer.reset(); show_result=False
    elif action=="save": save_canvas()

print("✅ Launching... Look for the popup window.")

try:
    while True:
        ret,frame=cap.read()
        if not ret: time.sleep(0.05); continue
        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        hand_list=gesture_engine.process_frame(rgb)

        if hand_list:
            hand=hand_list[0]; gesture=hand.gesture; fx,fy=hand.index_tip
            gesture_engine.draw_landmarks(frame,hand)
            if not show_result:
                action=ui.update_dwell(fx,fy)
                if action: handle_action(action)
                ui.draw_dwell_progress(frame,fx,fy)
            is_eraser=(ui.selected_palette_idx==ERASER_IDX)

            if gesture==Gesture.INDEX_ONLY and not show_result:
                sx,sy=canvas_engine.smooth(fx,fy)
                if fy>ui.toolbar_height:
                    if not was_drawing: canvas_engine.begin_stroke(sx,sy)
                    if is_eraser: canvas_engine.erase(sx,sy,28)
                    else: canvas_engine.continue_stroke(sx,sy)
                was_drawing=True; mode_label="ERASE" if is_eraser else "DRAW"
            elif gesture==Gesture.OPEN_HAND and not show_result:
                sx,sy=canvas_engine.smooth(fx,fy)
                if fy>ui.toolbar_height: canvas_engine.erase(sx,sy,40)
                if was_drawing: canvas_engine.end_stroke(); was_drawing=False
                mode_label="ERASE"
            elif gesture==Gesture.FIST and not show_result:
                if was_drawing: canvas_engine.end_stroke(); was_drawing=False
                if not fist_triggered and canvas_engine.has_content():
                    analyzer.submit(canvas_engine.get_canvas_snapshot())
                    show_result=True; fist_triggered=True
                mode_label="ANALYZE"
            else:
                if was_drawing: canvas_engine.end_stroke(); was_drawing=False
                fist_triggered=False; mode_label="HOVER"
            ui.draw_cursor(frame,fx,fy,gesture.name,ui.current_color,canvas_engine.thickness)
        else:
            if was_drawing: canvas_engine.end_stroke(); was_drawing=False
            fist_triggered=False; mode_label="NO HAND"

        frame=canvas_engine.composite(frame)
        ui.draw_toolbar(frame)
        ui.draw_status(frame,mode_label,canvas_engine.thickness)
        if show_result:
            ui.draw_result_overlay(frame,analyzer.result,analyzer.state.name)
        cv2.imshow(WINDOW_NAME,frame)

        key=cv2.waitKey(1)&0xFF
        if   key==ord('q'): break
        elif key==ord('r'): show_result=False; analyzer.reset()
        elif key==ord('z'): canvas_engine.undo()
        elif key==ord('c'): canvas_engine.clear(); show_result=False; analyzer.reset()
        elif key==ord('s'): save_canvas()
finally:
    cap.release(); cv2.destroyAllWindows(); print("👋 Canvas closed.")