## Project Goal
The goal of this project is to shelve books more efficently with the use of OCR to help libraries when there aren't shelvers available 


1. Capture live video feed of book spines on shelves
2. Detect book titles, guess, and return metadata
3. Overlay the title with author information
4. Allow user to shelve books efficiently (fiction by author, non-fiction by Dewey Decimal)


---
## Trying local OCR

###  PaddleOCR and EasyOCR
**Result**: `Segmentation fault` and `Illegal instruction` errors

### Tesseract OCR 

**Results**:<br> 
Worked with blocky text and perfect lighting,
good for black on white or white on black text
but were bad with colored backgrounds, varied fonts,  which are important with book spines.
Any preprocessing took a 100's of milleseconds before the image was even analyzed and was not very helpful. 

Preprocessing attempted included CLAHE, sharpening, denoising, top-hat transform, all of which resulted in hallucinations. 

Local OCR may still be possible with better hardware and/or tensorflow lite with some training. 

## Google Cloud API
I decided that it would be beneficial for me to pivot to a cloud solution since any preprocessing would add more delay than a call to an API, and something like google's vision API would be perfect. I also have had some expeirence in AWS so navigating google cloud was easy! Additionally it has 300$ of free credit when you first sign up!


## Resources & References

### AI Assistants
- **Claude Sonnet 4.5**
- **Gemini 3 Flash** 

### Inspiration / Guidance 
1. **Hackster.io Raspberry Pi OCR Edge AI Camera**
   https://www.hackster.io/johannasss/raspberry-pi-ocr-edge-ai-camera-2fdaa5

2. **YouTube: Raspberry Pi CM5 AI OCR Demo**
   https://www.youtube.com/watch?v=vgS-Vsnub0g

3. **TensorFlow Blog: Optical Character Recognition**
   https://blog.tensorflow.org/2021/09/blog.tensorflow.org202109optical-character-recognition.html

### APIs Used
- **Google Cloud Vision API**: https://cloud.google.com/vision/docs
- **Open Library API**: https://openlibrary.org/developers/api
- **Google Books API**: https://developers.google.com/books

### Key Documentation
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **OpenCV Python**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
