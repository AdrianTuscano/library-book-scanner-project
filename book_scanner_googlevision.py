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
    import pigpio
    PIGPIO_AVAILABLE = True
except ImportError:
    PIGPIO_AVAILABLE = False

CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
if not CREDENTIALS_PATH:
    CREDENTIALS_PATH = os.environ.get('GOOGLE_VISION_CREDENTIALS', '')

SERVO_PIN = 14
SERVO_FWD = 1700
SERVO_REV = 1300
SERVO_STOP = 0

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360


class ServoController:

    def __init__(self, pin=SERVO_PIN):
        self.pin = pin
        self.pi = None
        self.running = False
        if PIGPIO_AVAILABLE:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                self.pi = None

    def forward(self):
        if self.pi:
            self.pi.set_servo_pulsewidth(self.pin, SERVO_FWD)
            self.running = True

    def reverse(self):
        if self.pi:
            self.pi.set_servo_pulsewidth(self.pin, SERVO_REV)
            self.running = True

    def stop(self):
        if self.pi:
            self.pi.set_servo_pulsewidth(self.pin, SERVO_STOP)
            self.running = False

    def cleanup(self):
        if self.pi:
            self.pi.set_servo_pulsewidth(self.pin, 0)
            self.pi.stop()


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
        self.root.geometry("800x480")
        self.root.configure(bg='#2b2b2b')
        self.root.resizable(False, False)

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
        top_frame = tk.Frame(self.root, bg='#2b2b2b')
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        video_frame = tk.Frame(top_frame, bg='black', relief=tk.SUNKEN, bd=2)
        video_frame.pack(side=tk.LEFT)

        self.video_label = tk.Label(video_frame, bg='black', width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.video_label.pack()

        controls_frame = tk.Frame(top_frame, bg='#1e1e1e', relief=tk.SUNKEN, bd=2)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self.scan_button = tk.Button(controls_frame, text="SCAN",
                                     command=self.scan_books,
                                     font=('Arial', 14, 'bold'),
                                     bg='#4CAF50', fg='white',
                                     width=10, height=2,
                                     relief=tk.RAISED, bd=2)
        self.scan_button.pack(pady=10, padx=10)

        servo_label = tk.Label(controls_frame, text="Servo",
                              font=('Arial', 10),
                              bg='#1e1e1e', fg='white')
        servo_label.pack(pady=(10, 5))

        servo_buttons = tk.Frame(controls_frame, bg='#1e1e1e')
        servo_buttons.pack()

        self.reverse_button = tk.Button(servo_buttons, text="<",
                                        command=self.servo_reverse,
                                        font=('Arial', 12, 'bold'),
                                        bg='#2196F3', fg='white',
                                        width=3, height=1,
                                        relief=tk.RAISED, bd=2)
        self.reverse_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = tk.Button(servo_buttons, text="||",
                                     command=self.servo_stop,
                                     font=('Arial', 12, 'bold'),
                                     bg='#FF9800', fg='white',
                                     width=3, height=1,
                                     relief=tk.RAISED, bd=2)
        self.stop_button.pack(side=tk.LEFT, padx=2)

        self.forward_button = tk.Button(servo_buttons, text=">",
                                        command=self.servo_forward,
                                        font=('Arial', 12, 'bold'),
                                        bg='#2196F3', fg='white',
                                        width=3, height=1,
                                        relief=tk.RAISED, bd=2)
        self.forward_button.pack(side=tk.LEFT, padx=2)

        self.status_label = tk.Label(controls_frame, text="Ready",
                                     font=('Arial', 10),
                                     bg='#1e1e1e', fg='#4CAF50')
        self.status_label.pack(pady=10)

        quit_button = tk.Button(controls_frame, text="QUIT",
                               command=self.quit_app,
                               font=('Arial', 12),
                               bg='#f44336', fg='white',
                               width=10, height=1,
                               relief=tk.RAISED, bd=2)
        quit_button.pack(pady=10, padx=10)

        bottom_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.SUNKEN, bd=2)
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        results_header = tk.Frame(bottom_frame, bg='#1e1e1e')
        results_header.pack(fill=tk.X)

        results_title = tk.Label(results_header, text="Detected Books",
                                font=('Arial', 11, 'bold'),
                                bg='#1e1e1e', fg='white')
        results_title.pack(side=tk.LEFT, padx=10, pady=5)

        self.results_count = tk.Label(results_header, text="",
                                      font=('Arial', 10),
                                      bg='#1e1e1e', fg='#4CAF50')
        self.results_count.pack(side=tk.RIGHT, padx=10, pady=5)

        self.results_frame = tk.Frame(bottom_frame, bg='#1e1e1e')
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.no_results_label = tk.Label(self.results_frame,
                                         text="Press SCAN to detect books",
                                         font=('Arial', 10),
                                         bg='#1e1e1e', fg='#888888')
        self.no_results_label.pack(expand=True)

    def servo_forward(self):
        self.servo.forward()
        self.status_label.config(text="FWD", fg='#2196F3')

    def servo_reverse(self):
        self.servo.reverse()
        self.status_label.config(text="REV", fg='#2196F3')

    def servo_stop(self):
        self.servo.stop()
        self.status_label.config(text="STOP", fg='#FF9800')

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            if not self.scanning:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (VIDEO_WIDTH, VIDEO_HEIGHT))

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
        self.results_count.config(text="")

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        scanning_label = tk.Label(self.results_frame,
                                 text="Scanning...",
                                 font=('Arial', 10),
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
                               text="No books detected - ensure good lighting and steady camera",
                               font=('Arial', 10),
                               bg='#1e1e1e', fg='#888888')
            no_books.pack(expand=True)
            self.status_label.config(text="No books", fg='orange')
            self.results_count.config(text="")
        else:
            canvas = tk.Canvas(self.results_frame, bg='#1e1e1e', highlightthickness=0, height=70)
            scrollbar = tk.Scrollbar(self.results_frame, orient="horizontal", command=canvas.xview)
            scrollable_frame = tk.Frame(canvas, bg='#1e1e1e')

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(xscrollcommand=scrollbar.set)

            for book in results:
                book_frame = tk.Frame(scrollable_frame, bg='#2d2d2d', relief=tk.RAISED, bd=1)
                book_frame.pack(side=tk.LEFT, padx=3, pady=2, fill=tk.Y)

                pos_label = tk.Label(book_frame, text=f"#{book['position']}",
                                    font=('Arial', 8, 'bold'),
                                    bg='#4CAF50', fg='white',
                                    padx=4, pady=1)
                pos_label.pack(anchor=tk.W, padx=3, pady=(3, 0))

                title_label = tk.Label(book_frame, text=book.get('title', 'Unknown'),
                                      font=('Arial', 9, 'bold'),
                                      bg='#2d2d2d', fg='white',
                                      wraplength=180, justify=tk.LEFT)
                title_label.pack(anchor=tk.W, padx=5, pady=(2, 0))

                author_label = tk.Label(book_frame, text=f"by {book.get('author_full', 'Unknown')}",
                                       font=('Arial', 8),
                                       bg='#2d2d2d', fg='#cccccc')
                author_label.pack(anchor=tk.W, padx=5, pady=(0, 3))

            scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.status_label.config(text="Ready", fg='#4CAF50')
            self.results_count.config(text=f"Found {len(results)}")

        self.scanning = False
        self.scan_button.config(state=tk.NORMAL, bg='#4CAF50')

    def _display_error(self, error_msg):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        error_label = tk.Label(self.results_frame,
                              text=f"Error: {error_msg}",
                              font=('Arial', 10),
                              bg='#1e1e1e', fg='#ff5555')
        error_label.pack(expand=True)

        self.status_label.config(text="Failed", fg='red')
        self.results_count.config(text="")
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