import cv2
import numpy as np
import pytesseract
import re
from collections import Counter

# --- Constants ---
MIN_AREA = 600
AR_MIN, AR_MAX = 2.0, 8.0
W_OUT, H_OUT = 450, 140

# --- OCR Functions ---
def extract_text(image):
    
    # Tesseract configuration:
    # --psm 7: Treat the image as a single text line.
    # -c tessedit_char_whitelist: Limit characters to alphanumeric.
    
    
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

# --- Validation Functions ---
def is_valid_plate(text):
    # Example for a specific plate format (e.g., ABC-1234 or ABC1234)
    # Adjust the regex based on your local plate format
    pattern = r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$"
    if re.match(pattern, text):
        return True
    return False

def clean_text(text):
    # Remove any non-alphanumeric characters
    return re.sub(r"[^A-Z0-9]", "", text.upper())

# --- Temporal Filtering ---
class TemporalFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []

    def update(self, text):
        if not text:
            return None
        self.history.append(text)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Return the most frequent string in the window
        counter = Counter(self.history)
        most_common, count = counter.most_common(1)[0]
        
        # Only return if we have some confidence (e.g., at least 30% of window)
        if count >= (self.window_size // 3):
            return most_common
        return None

# --- Image Processing Functions ---
def find_plate_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
            
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue
            
        ar = max(w, h) / max(1.0, min(w, h))
        if AR_MIN <= ar <= AR_MAX:
            candidates.append(rect)
    return candidates

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    
    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )

def warp_plate(frame, rect):
    box = cv2.boxPoints(rect)
    src = order_points(box)
    
    dst = np.array([
        [0, 0],
        [W_OUT - 1, 0],
        [W_OUT - 1, H_OUT - 1],
        [0, H_OUT - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (W_OUT, H_OUT))
    return warped

# --- Main Loop ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")
        
    t_filter = TemporalFilter(window_size=10)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        vis = frame.copy()
        candidates = find_plate_candidates(frame)
        
        msg = "Detecting plate..."
        color = (0, 200, 255)

        if candidates:
            # choose largest candidate
            rect = max(candidates, key=lambda r: r[1][0] * r[1][1])
            
            box = cv2.boxPoints(rect).astype(int)
            cv2.polylines(vis, [box], True, (255, 0, 0), 2)

            warped = warp_plate(frame, rect)
            raw_text = extract_text(warped)
            text = clean_text(raw_text)

            if is_valid_plate(text):
                filtered_text = t_filter.update(text)
                if filtered_text:
                    msg = f"Plate: {filtered_text}"
                    color = (0, 255, 0)
            
            cv2.imshow("Warped Plate", warped)

        cv2.putText(
            vis,
            msg,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
        
        cv2.imshow("Result", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()