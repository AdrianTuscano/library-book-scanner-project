## Project Goal
Create a battery-powered device using a camera and display to:
1. Capture live video feed of book spines on shelves
2. Detect book titles, guess, and return metadata
3. Overlay the title with author information
4. Allow user to shelve books efficiently (fiction by author, non-fiction by Dewey Decimal)

---
## Hardware:
- Raspberry pi 4
- USB webcam

---
## The on Edge OCR Investigation

### Attempt 1: PaddleOCR 
**Result**: `Segmentation fault` and `Illegal instruction` errors

**Root cause**: PaddlePaddle pip packages compiled for x86/x64 CPUs with instruction sets that Raspberry Pi 4's ARM CPU doesn't support. Projects showing it working used custom hardware with AI accelerators, not  a standard Pi 4.

---

### Attempt 2: Tesseract OCR 

**Initial Results**: 
- Worked with blocky text and perfect lighting
- Good for black-on-white or white-on-black text
- Failed with colored backgrounds, varied fonts,  which are important with book spines
- Any preprocessing took a 100's of milleseconds before the image was even analyzed

**Preprocessing attempts**:
1. **Aggressive** (CLAHE + sharpening + denoising + thresholding) → Grainy, broken characters, hallucinations
2. **Balanced** (grayscale + 2x resize + CLAHE) → Better, still struggled with colors
3. **Top-hat transform** (morphological operations) → White blocky artifacts
4. **CLAHE + Binarization** → Clean black/white, but still limited

**Core limitation**: Tesseract designed for scanned documents, not real-world scenes with multiple fonts, angles, colors, and lighting variations.

---

### Attempt 3: EasyOCR 
**Result**: `Illegal instruction` error

**Root cause**: PyTorch dependency doesn't support Raspberry Pi's ARM CPU architecture.

---

### Attempt 4: TensorFlow Lite 
**Issue**: Would require to collect/label large dataset, train custom detection and recognition models, convert to TFLite, and optimize.

---

## The Cloud vs. Edge Pivot

### Why the pivot to cloud:
1. **Edge OCR fundamental limitations**: All local solutions hit accuracy walls with book spines
2. **Hardware incompatibility**: Raspberry Pi 4's ARM CPU can't run modern OCR frameworks
3. **Newfound available resources**: $300 Google Cloud credits

### Cloud Solution: Google Cloud Vision API

**Why this works**:
- More accuracy on varied fonts, colors, angles, light, etc
- Low hardware requirements beyond internet connection
- $300 credits ~  tens of thousands of OCR requests


**Architecture**:
1. Capture frame from webcam
2. Send to Google Vision API
3. Receive detected text (title, call number, author hint, dewey decimal)
4. Query Open Library API for full book metadata
5. Display overlay with complete information

**Current status**: Working prototype with single book detection based only on title. 

---

## Other Technical Challenges Encountered

### Challenge: Camera Device Detection
**Problem**: Multiple video devices (`/dev/video0` through `/dev/video23`)

**Investigation**:
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

**Result**: USB camera on `/dev/video0` and `/dev/video1`

**Solution**: Try `cv2.VideoCapture(0)` first, fallback to `1` if needed

---

## Key Learnings & Best Practices

---

### 1. Preprocessing Principles for OCR

**What helps**:
- Resize to larger resolution (2x)
- CLAHE for local contrast
- Clean binarization (pure black/white)
- Good lighting (would need a hardware ring light)

**What hurts**:
- Over-sharpening (creates artifacts)
- Aggressive denoising (blurs text)
- Adaptive thresholding on varied backgrounds (creates noise)
- Top-hat transform (creates artifacts)

---

### 2. The Cloud vs. Edge Decision Matrix

| Factor | Edge  | Cloud API |
|--------|------------------|-----------|
| **Accuracy** | Limited by hardware | State-of-the-art |
| **Latency** | Potentially Very fast (<100ms) | Consistently good (1> sec) |
| **Cost** | Hardware upfront | Per-request |
| **Internet** | Not required | Required |



---

### 3. The Two-Stage Lookup System

**Problem**: Book spines often show partial information
- Junior fiction: "ROWLING" but not "J.K."
- Only call number visible: "FIC ROW"
- Title but no author

**Solution**: Combine OCR + Database API
1. **OCR detects** what's visible: "Harry Potter", "ROW"
2. **Database enriches** with full metadata: "J.K. Rowling"

**APIs used**:
- Open Library API (free)
- Google Books API (also free but with credits )

**Current limitation**: Overall query logic needs to be improved to use all of the possible context. 

---

## Project Evolution Timeline

```
Research
    ↓
Research PaddleOCR (used in other examples)
    ↓
Attempt PaddleOCR → Segmentation Fault 
    ↓
Try Tesseract → Works but limited ✓
    ↓
Improve Tesseract with preprocessing → Better but not enough 
    ↓
Try EasyOCR → Illegal Instruction 
    ↓
Consider TensorFlow Lite → 
    ↓
Pivot to Google Cloud Vision → 
    ↓
Add Open Library lookup 
    ↓
[CURRENT STATUS] → Needs better query logic
```

---



## Resources & References

### AI Assistants
- **Claude Sonnet 4.5**
- **Gemini 3 Flash** 

### Successful Examples Analyzed
1. **Hackster.io Raspberry Pi OCR Edge AI Camera**
   - URL: https://www.hackster.io/johannasss/raspberry-pi-ocr-edge-ai-camera-2fdaa5
   - Hardware: Custom Raspberry Pi CM4 with AI accelerator
   - Not replicable on standard Pi 4

2. **YouTube: Raspberry Pi CM5 AI OCR Demo**
   - URL: https://www.youtube.com/watch?v=vgS-Vsnub0g
   - Hardware: Custom Raspberry Pi CM5 AI board
   - Uses PGNet OCR with custom hardware integration
   - Achieves 20 FPS on specialized hardware

3. **TensorFlow Blog: Optical Character Recognition**
   - URL: https://blog.tensorflow.org/2021/09/blog.tensorflow.org202109optical-character-recognition.html
   - Proof-of-concept, no ready-to-use model
   - Would require significant training effort

### APIs Used
- **Google Cloud Vision API**: https://cloud.google.com/vision/docs
- **Open Library API**: https://openlibrary.org/developers/api
- **Google Books API**: https://developers.google.com/books

### Key Documentation
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **OpenCV Python**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
