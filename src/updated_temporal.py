import cv2
import numpy as np
import pytesseract
import re
import csv
import os
import time
from collections import Counter

# --- 1. CONFIGURATION & PATHS ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Get the absolute path of the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the project root (assuming script is in 'src')
# This moves UP one level from 'src' then into 'data'
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
PLATES_DIR = os.path.join(DATA_DIR, "plates")

MIN_AREA = 600
AR_MIN, AR_MAX = 2.0, 8.0
W_OUT, H_OUT = 450, 140

# --- 2. DIRECTORY INITIALIZATION ---
for folder in [LOGS_DIR, PLATES_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")

# --- 3. FUNCTIONS (OCR, CLEAN, VALIDATE) ---
def extract_text(image):
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def is_valid_plate(text):
    # Regex: 1-3 Letters, 1-4 Numbers, 0-3 Letters
    pattern = r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$"
    return bool(re.match(pattern, text))

class TemporalFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []

    def update(self, text):
        if not text: return None
        self.history.append(text)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        counter = Counter(self.history)
        most_common, count = counter.most_common(1)[0]
        if count >= 4:
            return most_common
        return None

# --- 4. IMAGE PROCESSING ---
def find_plate_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA: continue
        rect = cv2.minAreaRect(cnt)
        ar = max(rect[1]) / max(1.0, min(rect[1]))
        if AR_MIN <= ar <= AR_MAX:
            candidates.append(rect)
    return candidates

def order_points(pts):
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    new_pts[0] = pts[np.argmin(s)]
    new_pts[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    new_pts[1] = pts[np.argmin(diff)]
    new_pts[3] = pts[np.argmax(diff)]
    return new_pts

def warp_plate(frame, rect):
    box = cv2.boxPoints(rect)
    src = order_points(box)
    dst = np.array([[0, 0], [W_OUT-1, 0], [W_OUT-1, H_OUT-1], [0, H_OUT-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (W_OUT, H_OUT))

# --- 5. MAIN LOOP ---
def main():
    cap = cv2.VideoCapture(0)
    t_filter = TemporalFilter(window_size=10)
    last_saved_plate = None 

    print(f"Logging to: {LOGS_DIR}")

    while True:
        ok, frame = cap.read()
        if not ok: break
        
        vis = frame.copy()
        candidates = find_plate_candidates(frame)
        msg = "Scanning..."
        color = (0, 165, 255)

        if candidates:
            rect = max(candidates, key=lambda r: r[1][0] * r[1][1])
            box = cv2.boxPoints(rect).astype(int)
            cv2.polylines(vis, [box], True, (255, 0, 0), 2)

            warped = warp_plate(frame, rect)
            text = clean_text(extract_text(warped))

            if is_valid_plate(text):
                confirmed_plate = t_filter.update(text)
                
                if confirmed_plate:
                    msg = f"CONFIRMED: {confirmed_plate}"
                    color = (0, 255, 0)
                    
                    if confirmed_plate != last_saved_plate:
                        # SAVING LOGIC
                        ts_file = time.strftime("%Y%m%d-%H%M%S")
                        
                        # Save Image
                        img_path = os.path.join(PLATES_DIR, f"{confirmed_plate}_{ts_file}.jpg")
                        cv2.imwrite(img_path, warped)
                        
                        # Save CSV
                        csv_path = os.path.join(LOGS_DIR, "plates_log.csv")
                        file_exists = os.path.isfile(csv_path)
                        with open(csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["Plate Number", "Timestamp"])
                            writer.writerow([confirmed_plate, time.strftime("%Y-%m-%d %H:%M:%S")])
                        
                        print(f"[SAVED] {confirmed_plate}")
                        last_saved_plate = confirmed_plate

            cv2.imshow("Warped Plate", warped)

        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Result", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()