# Car Number Plate Extraction

A computer vision project for detecting and extracting text from vehicle number plates using OpenCV and Tesseract OCR.

## Overview

This project implements a real-time number plate detection and recognition system that:
- Detects license plates in video streams from camera
- Extracts and aligns detected plates for better OCR accuracy
- Performs optical character recognition (OCR) to read plate numbers
- Includes temporal filtering for improved recognition consistency
- Validates plate formats and filters results

## Features

- **Real-time Detection**: Live camera feed processing for immediate plate detection
- **Plate Alignment**: Perspective transformation to straighten detected plates
- **OCR Integration**: Tesseract OCR engine for text extraction
- **Temporal Filtering**: Maintains history of detections to improve accuracy
- **Format Validation**: Validates extracted text against license plate patterns
- **Multiple Processing Modes**: Different modules for various use cases

## Project Structure

```
car-number-plate-extraction/
├── src/
│   ├── detect.py          # Basic plate detection visualization
│   ├── ocr.py            # Plate detection with OCR text extraction
│   ├── align.py          # Plate alignment and perspective correction
│   ├── temporal.py       # Temporal filtering for stable recognition
│   ├── updated_temporal.py # Enhanced temporal filtering
│   ├── validate.py       # Plate format validation
│   └── camera.py         # Camera utilities
├── data/
│   ├── logs/             # Application logs
│   └── plates/           # Extracted plate images
├── book/                 # Documentation or notebook files
└── .venv/               # Virtual environment
```

## Dependencies

- **OpenCV** (`cv2`) - Computer vision and image processing
- **NumPy** - Numerical operations and array handling
- **Pytesseract** - Python wrapper for Tesseract OCR engine
- **Tesseract-OCR** - OCR engine (must be installed separately)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd car-number-plate-extraction
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install Python dependencies:
```bash
pip install opencv-python numpy pytesseract
```

4. Install Tesseract OCR:
- Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Install and note the installation path
- Update the Tesseract path in the Python files if different from default

## Usage

### Basic Detection
Run the basic plate detection module:
```bash
python src/detect.py
```

### OCR with Text Extraction
Run the full detection and OCR pipeline:
```bash
python src/ocr.py
```

### Temporal Filtering
Run with temporal filtering for more stable results:
```bash
python src/temporal.py
```

### Enhanced Temporal Filtering
Run with improved temporal filtering algorithm:
```bash
python src/updated_temporal.py
```

## Controls

- **Press 'q'** to quit any running module
- Camera feed will display detected plates with bounding boxes
- OCR modules show extracted text, aligned plates, and thresholded images

## Configuration

### Tesseract Path
Update the Tesseract path in relevant Python files if installed elsewhere:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Detection Parameters
Adjust these constants in the source files:
- `MIN_AREA`: Minimum contour area for plate detection (default: 600)
- `AR_MIN`, `AR_MAX`: Aspect ratio range (default: 2.0, 8.0)
- `W_OUT`, H_OUT`: Output dimensions for aligned plates (default: 450x140)

### Plate Validation
Modify the regex pattern in `validate.py` and `temporal.py` to match your local license plate format:
```python
pattern = r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$"
```

## How It Works

1. **Detection**: Uses Canny edge detection and contour analysis to find rectangular regions matching license plate proportions
2. **Alignment**: Applies perspective transformation to straighten detected plates
3. **Preprocessing**: Converts to grayscale, applies blur and thresholding for OCR optimization
4. **OCR**: Uses Tesseract with custom configuration for alphanumeric character recognition
5. **Validation**: Filters results using regex patterns and temporal consistency
6. **Display**: Shows real-time results with bounding boxes and extracted text

## Modules Description

- **detect.py**: Simple plate detection with visual feedback
- **ocr.py**: Complete pipeline with OCR text extraction
- **align.py**: Focuses on plate alignment and perspective correction
- **temporal.py**: Adds temporal filtering for stable recognition over multiple frames
- **updated_temporal.py**: Enhanced version with logging and saving results to CSV
- **validate.py**: Plate format validation and text cleaning utilities
- **camera.py**: Camera interface utilities

## Requirements

- Python 3.7+
- Webcam or camera device
- Tesseract OCR engine installed
- Sufficient lighting for clear plate detection

## Limitations

- Performance depends on lighting conditions and camera quality
- OCR accuracy varies with plate condition and angle
- Currently configured for standard license plate formats
- Requires manual adjustment for different regional plate formats

