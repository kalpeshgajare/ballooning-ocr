import cv2
import numpy as np
import pandas as pd
import easyocr
import re
from typing import List, Dict, Any
import pytesseract

# ===================== CONFIG =====================
IMG_PATH = "dataset/007.png" # Using your latest uploaded image
CSV_OUT = "final_balloon_measurements.csv"
DEVICE = 'cuda' if 'torch' in locals() and torch.cuda.is_available() else 'cpu'

# --- Parameters for RED COLOR Detection (in HSV) ---
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])

# --- Parameters for Filtering Red Shapes ---
MIN_CONTOUR_AREA = 200      # Filters out small red noise.
MIN_CIRCULARITY = 0.8       # A perfect circle has a circularity of 1.0.

DEBUG = True

# ===================== VISION & TEXT PROCESSING =====================

def find_red_balloons(img: np.ndarray) -> List[Dict[str, Any]]:
    """Finds balloons by specifically looking for RED circular objects."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_balloons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA: continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if circularity > MIN_CIRCULARITY:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            found_balloons.append({
                "center": (int(x), int(y)),
                "radius": int(radius),
                "number": None
            })
    return found_balloons

# <<< KEY IMPROVEMENT: This function is now much more robust for thin digits
def get_number_from_roi(roi: np.ndarray, reader: easyocr.Reader) -> str:
    """Enlarges, cleans, and then performs OCR on the balloon's region of interest."""
    if roi.size == 0:
        return None
    
    # 1. Scale up the image to give OCR more detail to work with.
    # This is the crucial step for thin digits like '1'.
    scale_factor = 4
    width = int(roi.shape[1] * scale_factor)
    height = int(roi.shape[0] * scale_factor)
    resized_roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC)

    # 2. Preprocess the enlarged image for OCR robustness
    gray = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding which is great for bimodal (black/white) images
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Try EasyOCR on the clean, enlarged image
    easy_result = reader.readtext(thresh, allowlist='0123456789', detail=0)

    if DEBUG:
        cv2.imshow("Scaled and Processed ROI for OCR", thresh)
        cv2.waitKey(1) # Use waitKey(1) to allow the window to refresh without pausing

    if easy_result:
        return easy_result[0]

    # 4. Fallback with Pytesseract if EasyOCR fails
    try:
        # PSM 10 tells Tesseract to treat the image as a single character
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
        tess_result = pytesseract.image_to_string(thresh, config=config)
        tess_result = re.sub(r'\D', '', tess_result) # Clean out any non-digit characters
        if tess_result:
            return tess_result
    except Exception:
        pass # Pytesseract might not be installed or might fail
    
    return None

def extract_measurement_text(reader: easyocr.Reader, img: np.ndarray) -> List[Dict[str, Any]]:
    """Extracts all text, intended for measurements."""
    results = reader.readtext(img, paragraph=False)
    return [{'text': text.strip(), 'center': (int((bbox[0][0] + bbox[2][0]) / 2), int((bbox[0][1] + bbox[2][1]) / 2))} for (bbox, text, conf) in results]

def group_measurement_fragments(measurement_parts: List[Dict], merge_threshold: int = 80) -> List[Dict]:
    grouped_measurements, visited = [], set()
    for i, part1 in enumerate(measurement_parts):
        if i in visited: continue
        current_group, queue = [part1], [part1]
        visited.add(i)
        while queue:
            base = queue.pop(0)
            for j, part2 in enumerate(measurement_parts):
                if j not in visited and np.hypot(base['center'][0] - part2['center'][0], base['center'][1] - part2['center'][1]) < merge_threshold:
                    current_group.append(part2); visited.add(j); queue.append(part2)
        current_group.sort(key=lambda r: r['center'][0])
        full_text = ' '.join([r['text'] for r in current_group])
        center = (int(np.mean([r['center'][0] for r in current_group])), int(np.mean([r['center'][1] for r in current_group])))
        grouped_measurements.append({'text': full_text, 'center': center})
    return grouped_measurements

def parse_and_format_measurement(text: str) -> Dict[str, Any]:
    """Parses text and formats it into the required Measurement and Tolerance columns."""
    clean_text = re.sub(r'[Ø°Op]', '', text).strip()
    numbers = [n.replace(" ", "") for n in re.findall(r'[+\-±]?\s*\d+\.?\d*', clean_text)]
    if not numbers: return {'Measurement': clean_text, 'Tolerance (+/-)': 'N/A'}
    try: nominal = str(float(numbers[0]))
    except ValueError: return {'Measurement': clean_text, 'Tolerance (+/-)': 'N/A'}

    upper, lower = 0.0, 0.0
    if len(numbers) > 1 and '±' in numbers[1]: upper, lower = float(numbers[1].replace('±', '')), -float(numbers[1].replace('±', ''))
    elif len(numbers) == 2: val = float(numbers[1]); upper, lower = (val, 0.0) if val >= 0 else (0.0, val)
    elif len(numbers) >= 3:
        try: val1, val2 = float(numbers[1]), float(numbers[2]); upper, lower = max(val1, val2), min(val1, val2)
        except (ValueError, IndexError): pass
    
    return {'Measurement': nominal, 'Tolerance (+/-)': f"+{upper} / {lower}"}

# ===================== MAIN EXECUTION =====================
def analyze_drawing(img_path: str, reader: easyocr.Reader):
    img = cv2.imread(img_path)
    if img is None: return print(f"❌ Error: Image not found at '{img_path}'.")

    balloons = find_red_balloons(img)
    print(f"🔍 Found {len(balloons)} red circular shapes.")
    if not balloons: return

    confirmed_balloons = []
    for b in balloons:
        x, y, r = b['center'][0], b['center'][1], b['radius']
        padding = 5
        roi = img[max(0, y-r-padding):y+r+padding, max(0, x-r-padding):x+r+padding]
        
        number = get_number_from_roi(roi, reader)
        if number and number.isdigit():
            b['number'] = number
            confirmed_balloons.append(b)

    print(f"✅ Confirmed {len(confirmed_balloons)} balloons using scaled OCR.")

    # Exclude confirmed balloon numbers from the measurement text list
    all_text = extract_measurement_text(reader, img)
    confirmed_numbers = {b['number'] for b in confirmed_balloons}
    measurement_parts = []
    for text_item in all_text:
        # A simple check to avoid adding the balloon numbers to the measurement parts
        is_a_balloon_num = False
        for b in confirmed_balloons:
            if np.hypot(text_item['center'][0] - b['center'][0], text_item['center'][1] - b['center'][1]) < b['radius']:
                is_a_balloon_num = True
                break
        if not is_a_balloon_num:
            measurement_parts.append(text_item)
            
    measurement_groups = group_measurement_fragments(measurement_parts)
    print(f"🧩 Merged text fragments into {len(measurement_groups)} measurement groups.")

    final_data = []
    for balloon in confirmed_balloons:
        best_match = min(measurement_groups, key=lambda g: np.hypot(balloon['center'][0] - g['center'][0], balloon['center'][1] - g['center'][1]))
        formatted = parse_and_format_measurement(best_match['text'])
        final_data.append({"Balloon Number": int(balloon['number']), **formatted})
    
    df = pd.DataFrame(final_data).sort_values(by="Balloon Number").reset_index(drop=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n📁 Results saved to {CSV_OUT}\n--- Final Mapped Data ---\n{df.to_string()}")

    if DEBUG:
        for balloon in confirmed_balloons:
            cv2.circle(img, balloon['center'], balloon['radius'], (0, 255, 0), 3)
            cv2.putText(img, balloon['number'], (balloon['center'][0] - 15, balloon['center'][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.imshow("Final Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Initializing EasyOCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=(DEVICE == 'cuda'))
    analyze_drawing(IMG_PATH, ocr_reader)