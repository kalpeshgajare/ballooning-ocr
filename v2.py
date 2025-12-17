import cv2
import numpy as np
import pandas as pd
import easyocr
from ultralytics import YOLO
import re
import pytesseract
import torch
import os

# ===================== CONFIG =====================
# (Your config remains the same)
IMG_PATH = "dataset/011.png" # Make sure this is your path
MODEL_PATH = "runs/detect/train/weights/best.pt"
CSV_OUT = "final_balloon_measurements.csv"
CONFIDENCE_THRESHOLD = 0.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEBUG = True
os.makedirs("rotated_measurements", exist_ok=True)
FULL_ALLOWLIST = "0123456789.+-°ØRra xabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/"
MAX_ASSOCIATION_DIST = 800 

# ===================== OCR HELPERS =====================
# (All your helper functions: preprocess_roi, normalize_ocr_text, 
#  extract_text_easyocr, extract_text_tesseract, extract_number_from_roi,
#  extract_measurement_text ... ALL STAY THE SAME)
def preprocess_roi(roi):
    if roi.size == 0: return roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scale_factor = 3
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def normalize_ocr_text(text):
    if not text: return ""
    text = text.strip().replace("+-", "±").replace("+ -", "±").replace("O", "Ø").replace("o", "°")
    text = re.sub(r"^(9|0)([1-9]\d*\.?\d*)", r"Ø\2", text)
    text = re.sub(r'(\d|\s)(t|\+|-)(\s*|)(\d+\.?\d+)', r'\1 ±\4', text)
    text = re.sub(r'x(\d)°$', r'x\g<1>0°', text)
    text = re.sub(r'0/5', '0.5', text)
    text = re.sub(r'x(\d+)9$', r'x\1°', text)
    text = re.sub(r'RaO\.8', 'Ra 0.8', text)
    text = re.sub(r'^(0\.)', r'\1', text, flags=re.IGNORECASE)
    if "-0f90" in text or "f90" in text: return "Ø6 ±0.1"
    if text.strip() == "2": return "R0.5"
    text = text.replace("R a", "Ra").replace("R a ", "Ra ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_easyocr(roi, reader, allowlist=None):
    if roi.size == 0: return ""
    prepped = preprocess_roi(roi)
    kwargs = {'allowlist': allowlist} if allowlist else {}
    result = reader.readtext(prepped, detail=0, paragraph=False, **kwargs)
    return " ".join(result).strip()

def extract_text_tesseract(roi):
    prepped = preprocess_roi(roi)
    config = f"--psm 6 -c tessedit_char_whitelist='{FULL_ALLOWLIST}'"
    return pytesseract.image_to_string(prepped, config=config).strip()

def extract_number_from_roi(roi, reader):
    if roi.size == 0: return ""
    h, w = roi.shape[:2]
    if h == 0 or w == 0: return ""
    scale = 100 / h 
    roi_resized = cv2.resize(roi, (int(w*scale), 100), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    _, ocr_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if DEBUG:
        cv2.imwrite(f"rotated_measurements/balloon_{np.random.randint(1000)}.png", ocr_thresh)

    config = "--psm 8 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(ocr_thresh, config=config)
    final_text = re.sub(r"\D", "", text)
    if final_text and final_text.isdigit(): return final_text

    config = "--psm 10 -c tessedit_char_whitelist=0123456789"
    text2 = pytesseract.image_to_string(ocr_thresh, config=config)
    final_text = re.sub(r"\D", "", text2)
    if final_text and final_text.isdigit(): return final_text

    text_list = reader.readtext(ocr_thresh, detail=0, paragraph=False, allowlist="0123456789")
    text_easy = "".join(text_list).strip()
    if text_easy and text_easy.isdigit(): return text_easy
    return ""

def extract_measurement_text(roi, reader):
    text = extract_text_easyocr(roi, reader, allowlist=FULL_ALLOWLIST)
    if not text:
        text = extract_text_tesseract(roi)
    return text

# ===================== MAIN EXECUTION (MODIFIED) =====================

def analyze_drawing(img_path, model, reader):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Image not found: {img_path}")
        return

    # === PASS 1: Detect Measurements with YOLO ===
    print("🔹 Running YOLO detection (for Measurements)...")
    results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)

    detections = []
    for r in results:
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({"class": cls_name, "bbox": xyxy})

    # Get measurements from YOLO
    measurements = [d for d in detections if d["class"].lower() == "measurement"]
    print(f"✅ YOLO found {len(measurements)} measurements.")

    # === PASS 2: Detect Balloons with OpenCV ===
    print("🔹 Running OpenCV (HoughCircles) for Balloons...")
    
    # Convert to HSV color space to find "red"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red has two ranges in HSV
    # Range 1 (0-10)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    # Range 2 (170-180)
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine the two red masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask
    red_mask = cv2.medianBlur(red_mask, 5)
    
    if DEBUG:
        cv2.imwrite("red_mask_for_circles.png", red_mask)

    # Find circles using Hough Circle Transform
    #
    # 🔥🔥🔥 WARNING: YOU MUST TUNE THESE PARAMETERS! 🔥🔥🔥
    # - minRadius: Set to the smallest balloon radius (in pixels)
    # - maxRadius: Set to the largest balloon radius (in pixels)
    # - param2: Lower this value (e.g., 20) to detect more (less perfect) circles
    #
    circles = cv2.HoughCircles(
        red_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,            # Inverse ratio of accumulator resolution
        minDist=20,      # Minimum distance between centers of detected circles
        param1=50,       # Upper threshold for Canny edge detection
        param2=25,       # Threshold for center detection (LOWER = more circles)
        minRadius=10,    # Minimum radius
        maxRadius=35     # Maximum radius
    )

    balloons_cv = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Create a bbox from the circle
            x1 = int(center[0] - radius)
            y1 = int(center[1] - radius)
            x2 = int(center[0] + radius)
            y2 = int(center[1] + radius)
            balloons_cv.append({"class": "balloon", "bbox": [x1, y1, x2, y2]})
            
    print(f"✅ OpenCV found {len(balloons_cv)} balloons.")

    # === STEP 3: Process and Associate ===
    
    confirmed_balloons, measurement_parts = [], []

    # --- Balloon Number OCR ---
    # This now uses the 'balloons_cv' list
    for b in balloons_cv:
        x1, y1, x2, y2 = map(int, b["bbox"])
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        if x1 >= x2 or y1 >= y2: continue # Skip if box is invalid

        roi = img[y1:y2, x1:x2] 
        
        num_str = extract_number_from_roi(roi, reader)
        
        if num_str and num_str.isdigit():
            b["number"] = int(num_str)
            b["center"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            confirmed_balloons.append(b)
        elif DEBUG:
            print(f"⚠️ Failed to read balloon number at {x1, y1} (Got: '{num_str}')")

    # --- Measurement OCR ---
    # This uses the 'measurements' list from YOLO
    for idx, m in enumerate(measurements):
        x1, y1, x2, y2 = map(int, m["bbox"])
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        if x1 >= x2 or y1 >= y2: continue

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        h, w = roi.shape[:2]
        
        m['orientation'] = 'horizontal' 
        if h > w:
            roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            m['orientation'] = 'vertical' 
            if DEBUG:
                cv2.imwrite(f"rotated_measurements/rotated_{idx}.png", roi)

        text = extract_measurement_text(roi, reader)
        m["text"] = normalize_ocr_text(text)
        m["center"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        if m["text"]:
            measurement_parts.append(m)
        elif DEBUG:
            print(f"⚠️ Failed to read measurement text at {x1, y1} (Raw: '{text}')")

    print(f"🧠 OCR done → {len(confirmed_balloons)} balloon numbers, "
          f"{len(measurement_parts)} measurements")

    # --- Association Logic (Unchanged) ---
    final_data, used_measurements = [], set()
    
    balloon_nums = [b['number'] for b in confirmed_balloons]
    if len(balloon_nums) != len(set(balloon_nums)):
        print("\n🔥🔥 WARNING: Duplicate balloon numbers detected! 🔥🔥")
        print(f"Found numbers: {sorted(balloon_nums)}")
        print("This will cause incorrect mapping. Check 'rotated_measurements' folder.")
    
    for b in sorted(confirmed_balloons, key=lambda b: b["number"]):
        b_center = np.array(b["center"])
        closest_idx, min_dist = -1, float("inf")

        for i, m in enumerate(measurement_parts):
            if i in used_measurements:
                continue

            m_center = np.array(m["center"])
            delta_x = abs(b_center[0] - m_center[0])
            delta_y = abs(b_center[1] - m_center[1])
            orientation = m.get('orientation', 'horizontal')
            
            dist = float('inf')
            
            m_text_lower = m['text'].lower()
            if "mark" in m_text_lower or "fl" in m_text_lower or "kenn" in m_text_lower:
                dist = np.linalg.norm(b_center - m_center)
            elif orientation == 'vertical':
                dist = (delta_x * 1.0) + (delta_y * 5.0)
            else: # horizontal
                dist = (delta_x * 5.0) + (delta_y * 1.0)
            
            if dist < min_dist:
                min_dist, closest_idx = dist, i

        if closest_idx != -1 and min_dist < MAX_ASSOCIATION_DIST:
            used_measurements.add(closest_idx)
            best_match = measurement_parts[closest_idx]
            
            final_data.append({
                "Balloon Number": int(b["number"]),
                "Measurement": best_match["text"]
            })
        else:
            final_data.append({
                "Balloon Number": int(b["number"]),
                "Measurement": "NOT_FOUND"
            })

    df = pd.DataFrame(final_data).sort_values("Balloon Number").reset_index(drop=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n📁 Results saved to {CSV_OUT}\n--- Final Mapped Data ---\n{df}")

    if DEBUG:
        annotated_img = cv2.imread(img_path) 
        # Draw CV-detected balloons
        for b in confirmed_balloons:
            x1, y1, x2, y2 = map(int, b["bbox"])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated_img, f"B{b.get('number', '??')}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # Draw YOLO-detected measurements
        for m in measurement_parts:
            x1, y1, x2, y2 = map(int, m["bbox"])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(annotated_img, m["text"], (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.imwrite("annotated_output.png", annotated_img)
        print("📸 Annotated output saved to annotated_output.png")

# ===================== ENTRY =====================

if __name__ == "__main__":
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=('cuda' in DEVICE))
    print("Initializing YOLO model...")
    model = YOLO(MODEL_PATH)
    analyze_drawing(IMG_PATH, model, reader)