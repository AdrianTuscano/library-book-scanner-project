import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from google.cloud import vision
import requests
from urllib.parse import quote_plus
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
if not CREDENTIALS_PATH:
    CREDENTIALS_PATH = os.environ.get('GOOGLE_VISION_CREDENTIALS', '')

SERVO_PIN = 14
SERVO_FREQ = 50
SERVO_FWD = 8.5
SERVO_REV = 6.5

class ServoController:

    def __init__(self, pin=SERVO_PIN):
        self.pin = pin
        self.pwm = None
        self.running = False
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.pin, SERVO_FREQ)
            self.pwm.start(0)

    def forward(self):
        if self.pwm:
            self.pwm.ChangeDutyCycle(SERVO_FWD)
            self.running = True

    def reverse(self):
        if self.pwm:
            self.pwm.ChangeDutyCycle(SERVO_REV)
            self.running = True

    def stop(self):
        if self.pwm:
            self.pwm.ChangeDutyCycle(0)
            self.running = False

    def cleanup(self):
        if self.pwm:
            self.pwm.ChangeDutyCycle(0)
            self.pwm.stop()
        if GPIO_AVAILABLE:
            GPIO.cleanup(self.pin)


class BookScanner:

    NOISE_WORDS = frozenset({'press', 'books', 'publishing', 'inc', 'ltd', 'co', 'e'})

    def __init__(self):
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set!")
        self.vision_client = vision.ImageAnnotatorClient()
        self._session = requests.Session()

    def _extract_book_info(self, doc, call_number, confidence):
        author_names = doc.get('author_name')
        return {
            'title': doc.get('title', 'Unknown'),
            'author_full': author_names[0] if author_names else 'Unknown',
            'first_publish_year': doc.get('first_publish_year', 'Unknown'),
            'call_number': call_number,
            'confidence': confidence
        }

    def _fetch_search(self, url):
        response = self._session.get(url, timeout=5)
        response.raise_for_status()
        return response.json()

    def smart_book_search(self, texts, call_number=None):
        filtered = [t for t in texts if t.lower() not in self.NOISE_WORDS]

        if not filtered:
            return None

        combined_text = quote_plus(" ".join(filtered))
        url = f"https://openlibrary.org/search.json?q={combined_text}"

        try:
            data = self._fetch_search(url)
            if data.get('docs'):
                confidence = 'high' if data['numFound'] < 10 else 'medium'
                return self._extract_book_info(data['docs'][0], call_number, confidence)
        except requests.RequestException:
            pass

        by_length = sorted(filtered, key=len, reverse=True)
        if by_length:
            title_guess = quote_plus(by_length[0])
            params = f"title={title_guess}"
            if len(by_length) > 1:
                params += f"&author={quote_plus(by_length[1])}"

            url = f"https://openlibrary.org/search.json?{params}"
            try:
                data = self._fetch_search(url)
                if data.get('docs'):
                    return self._extract_book_info(data['docs'][0], call_number, 'medium')
            except requests.RequestException:
                pass

        return None
    
    def detect_text_vision(self, image_frame):
        success, encoded_image = cv2.imencode('.jpg', image_frame)
        if not success:
            return None

        image = vision.Image(content=encoded_image.tobytes())

        try:
            response = self.vision_client.text_detection(image=image)
            if response.error.message or not response.text_annotations:
                return None
            return {
                'full_text': response.text_annotations[0].description,
                'text_annotations': response.text_annotations[1:]
            }
        except Exception:
            return None

    def get_text_center_x(self, annotation):
        vertices = annotation.bounding_poly.vertices
        return sum(v.x for v in vertices) / len(vertices)

    def cluster_books_by_gap(self, text_annotations, gap_threshold=20):
        if not text_annotations:
            return []

        words = sorted(
            [{'text': ann.description, 'x': self.get_text_center_x(ann), 'annotation': ann}
             for ann in text_annotations],
            key=lambda w: w['x']
        )

        clusters = [[words[0]]]
        for i in range(1, len(words)):
            if words[i]['x'] - words[i-1]['x'] > gap_threshold:
                clusters.append([words[i]])
            else:
                clusters[-1].append(words[i])

        return clusters
    
    def extract_call_number(self, texts):
        for text in texts:
            if re.match(r'^FIC\s+[A-Z]{3}', text) or re.match(r'^\d{3}\.?\d*', text):
                return text
        return None

    def scan_books(self, frame):
        detected = self.detect_text_vision(frame)
        if not detected or not detected['text_annotations']:
            return []

        clusters = self.cluster_books_by_gap(detected['text_annotations'], gap_threshold=20)
        if not clusters:
            return []

        search_tasks = [
            ([w['text'] for w in cluster], self.extract_call_number([w['text'] for w in cluster]))
            for cluster in clusters
        ]

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(self.smart_book_search, texts, call_num): idx
                for idx, (texts, call_num) in enumerate(search_tasks)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        result['position'] = idx + 1
                        results.append(result)
                except Exception:
                    pass

        results.sort(key=lambda x: x['position'])
        return results


class BookScannerGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Book Scanner")
        self.root.geometry("640x480")
        self.root.configure(bg='#2b2b2b')

        try:
            self.scanner = BookScanner()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.root.quit()
            return

        self.servo = ServoController()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            self.root.quit()
            return

        self.scanning = False
        self.current_frame = None

        self.setup_gui()
        self.update_video()

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def setup_gui(self):

        title_label = tk.Label(self.root, text="Book Scanner",
                              font=('Arial', 14, 'bold'),
                              bg='#2b2b2b', fg='white')
        title_label.pack(pady=2)

        main_container = tk.Frame(self.root, bg='#2b2b2b')
        main_container.pack(fill=tk.BOTH, expand=True, padx=5)

        left_frame = tk.Frame(main_container, bg='#2b2b2b')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        video_frame = tk.Frame(left_frame, bg='black', relief=tk.SUNKEN, bd=1)
        video_frame.pack(pady=2)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack()

        right_frame = tk.Frame(main_container, bg='#1e1e1e', relief=tk.SUNKEN, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.config(width=200)

        results_title = tk.Label(right_frame, text="Detected Books",
                                font=('Arial', 10, 'bold'),
                                bg='#1e1e1e', fg='white')
        results_title.pack(pady=5)

        self.results_frame = tk.Frame(right_frame, bg='#1e1e1e')
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.no_results_label = tk.Label(self.results_frame,
                                         text="Press SCAN\nto start",
                                         font=('Arial', 9),
                                         bg='#1e1e1e', fg='#888888')
        self.no_results_label.pack(expand=True)

        bottom_frame = tk.Frame(self.root, bg='#2b2b2b')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        button_container = tk.Frame(bottom_frame, bg='#2b2b2b')
        button_container.pack()

        self.scan_button = tk.Button(button_container, text="SCAN",
                                     command=self.scan_books,
                                     font=('Arial', 10, 'bold'),
                                     bg='#4CAF50', fg='white',
                                     padx=15, pady=5,
                                     relief=tk.RAISED, bd=2)
        self.scan_button.pack(side=tk.LEFT, padx=5)

        servo_frame = tk.Frame(button_container, bg='#2b2b2b')
        servo_frame.pack(side=tk.LEFT, padx=10)

        servo_buttons = tk.Frame(servo_frame, bg='#2b2b2b')
        servo_buttons.pack()

        self.reverse_button = tk.Button(servo_buttons, text="<",
                                        command=self.servo_reverse,
                                        font=('Arial', 10, 'bold'),
                                        bg='#2196F3', fg='white',
                                        padx=8, pady=3,
                                        relief=tk.RAISED, bd=2)
        self.reverse_button.pack(side=tk.LEFT, padx=1)

        self.stop_button = tk.Button(servo_buttons, text="||",
                                     command=self.servo_stop,
                                     font=('Arial', 10, 'bold'),
                                     bg='#FF9800', fg='white',
                                     padx=8, pady=3,
                                     relief=tk.RAISED, bd=2)
        self.stop_button.pack(side=tk.LEFT, padx=1)

        self.forward_button = tk.Button(servo_buttons, text=">",
                                        command=self.servo_forward,
                                        font=('Arial', 10, 'bold'),
                                        bg='#2196F3', fg='white',
                                        padx=8, pady=3,
                                        relief=tk.RAISED, bd=2)
        self.forward_button.pack(side=tk.LEFT, padx=1)

        quit_button = tk.Button(button_container, text="QUIT",
                               command=self.quit_app,
                               font=('Arial', 10),
                               bg='#f44336', fg='white',
                               padx=15, pady=5,
                               relief=tk.RAISED, bd=2)
        quit_button.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(bottom_frame, text="Ready",
                                     font=('Arial', 9),
                                     bg='#2b2b2b', fg='#4CAF50')
        self.status_label.pack(pady=2)

    def servo_forward(self):
        self.servo.forward()
        self.status_label.config(text="Servo: FWD", fg='#2196F3')

    def servo_reverse(self):
        self.servo.reverse()
        self.status_label.config(text="Servo: REV", fg='#2196F3')

    def servo_stop(self):
        self.servo.stop()
        self.status_label.config(text="Servo: STOP", fg='#FF9800')
    
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            if not self.scanning:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame_rgb.shape[:2]
                max_width = 380
                scale = max_width / width
                frame_rgb = cv2.resize(frame_rgb,
                                      (int(width * scale), int(height * scale)))

                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video)
    
    def scan_books(self):
        if self.scanning or self.current_frame is None:
            return

        self.scanning = True
        self.scan_button.config(state=tk.DISABLED, bg='#cccccc')
        self.status_label.config(text="Scanning...", fg='orange')

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        scanning_label = tk.Label(self.results_frame,
                                 text="Scanning...",
                                 font=('Arial', 9),
                                 bg='#1e1e1e', fg='orange')
        scanning_label.pack(expand=True)

        thread = threading.Thread(target=self._scan_worker)
        thread.daemon = True
        thread.start()
    
    def _scan_worker(self):
        try:
            results = self.scanner.scan_books(self.current_frame)
            self.root.after(0, self._display_results, results)
        except Exception as e:
            self.root.after(0, self._display_error, str(e))
    
    def _display_results(self, results):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not results:
            no_books = tk.Label(self.results_frame,
                               text="No books found\n\nTips:\n- Good lighting\n- Steady camera",
                               font=('Arial', 8),
                               bg='#1e1e1e', fg='#888888',
                               justify=tk.LEFT)
            no_books.pack(pady=10)
            self.status_label.config(text="No books", fg='orange')
        else:
            canvas = tk.Canvas(self.results_frame, bg='#1e1e1e', highlightthickness=0)
            scrollbar = tk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg='#1e1e1e')

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            for book in results:
                book_frame = tk.Frame(scrollable_frame, bg='#2d2d2d', relief=tk.RAISED, bd=1)
                book_frame.pack(fill=tk.X, padx=2, pady=2)

                position_text = f"#{book['position']}"

                pos_label = tk.Label(book_frame, text=position_text,
                                    font=('Arial', 8, 'bold'),
                                    bg='#4CAF50', fg='white',
                                    padx=3, pady=1)
                pos_label.pack(anchor=tk.W, padx=3, pady=(2, 0))

                title_label = tk.Label(book_frame, text=book.get('title', 'Unknown'),
                                      font=('Arial', 8, 'bold'),
                                      bg='#2d2d2d', fg='white',
                                      wraplength=170, justify=tk.LEFT)
                title_label.pack(anchor=tk.W, padx=5, pady=(2, 0))

                author_label = tk.Label(book_frame, text=f"by {book.get('author_full', 'Unknown')}",
                                       font=('Arial', 7),
                                       bg='#2d2d2d', fg='#cccccc')
                author_label.pack(anchor=tk.W, padx=5, pady=(0, 2))

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.status_label.config(text=f"Found {len(results)}", fg='#4CAF50')

        self.scanning = False
        self.scan_button.config(state=tk.NORMAL, bg='#4CAF50')
    
    def _display_error(self, error_msg):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        error_label = tk.Label(self.results_frame,
                              text=f"Error:\n{error_msg}",
                              font=('Arial', 8),
                              bg='#1e1e1e', fg='#ff5555')
        error_label.pack(pady=10)

        self.status_label.config(text="Failed", fg='red')
        self.scanning = False
        self.scan_button.config(state=tk.NORMAL, bg='#4CAF50')

    def quit_app(self):
        self.servo.cleanup()
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = BookScannerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()