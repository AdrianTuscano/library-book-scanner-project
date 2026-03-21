import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from google.cloud import vision
import requests
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

CREDENTIALS_PATH = "/home/adrian-tuscano/Desktop/adriant-computer-vision-pp-227e26cc3bdb.json"
if os.path.exists(CREDENTIALS_PATH):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
else:
    print(f"WARNING: Credentials file not found at {CREDENTIALS_PATH}")

class BookScanner:
    
    def __init__(self):
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set!")
        self.vision_client = vision.ImageAnnotatorClient()
    
    def smart_book_search(self, texts, call_number=None):
        # Filter noise words
        noise_words = {'press', 'books', 'publishing', 'inc', 'ltd', 'co', 'e'}
        filtered = [t for t in texts if t.lower() not in noise_words]
        
        if not filtered:
            return None
        
        combined_text = " ".join(filtered)
        url = f"https://openlibrary.org/search.json?q={combined_text}"
        
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get('docs') and len(data['docs']) > 0:
                best_match = data['docs'][0]
                return {
                    'title': best_match.get('title', 'Unknown'),
                    'author_full': best_match.get('author_name', ['Unknown'])[0] if best_match.get('author_name') else 'Unknown',
                    'first_publish_year': best_match.get('first_publish_year', 'Unknown'),
                    'call_number': call_number,
                    'confidence': 'high' if data['numFound'] < 10 else 'medium'
                }
        except:
            pass
        
        by_length = sorted(filtered, key=len, reverse=True)
        if by_length:
            title_guess = by_length[0]
            author_guess = by_length[1] if len(by_length) > 1 else None
            
            params = [f"title={title_guess}"]
            if author_guess:
                params.append(f"author={author_guess}")
            
            url = "https://openlibrary.org/search.json?" + "&".join(params)
            try:
                response = requests.get(url, timeout=5)
                data = response.json()
                
                if data.get('docs') and len(data['docs']) > 0:
                    best_match = data['docs'][0]
                    return {
                        'title': best_match.get('title', 'Unknown'),
                        'author_full': best_match.get('author_name', ['Unknown'])[0] if best_match.get('author_name') else 'Unknown',
                        'first_publish_year': best_match.get('first_publish_year', 'Unknown'),
                        'call_number': call_number,
                        'confidence': 'medium'
                    }
            except:
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
        except:
            return None
    
    def get_text_center_x(self, annotation):
        vertices = annotation.bounding_poly.vertices
        x_coords = [v.x for v in vertices]
        return sum(x_coords) / len(x_coords)
    
    def cluster_books_by_gap(self, text_annotations, gap_threshold=20):
        if not text_annotations:
            return []
        
        words = []
        for annotation in text_annotations:
            words.append({
                'text': annotation.description,
                'x': self.get_text_center_x(annotation),
                'annotation': annotation
            })
        
        words.sort(key=lambda w: w['x'])
        
        clusters = []
        current_cluster = [words[0]]
        
        for i in range(1, len(words)):
            gap = words[i]['x'] - words[i-1]['x']
            
            if gap > gap_threshold:
                clusters.append(current_cluster)
                current_cluster = [words[i]]
            else:
                current_cluster.append(words[i])
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def extract_call_number(self, texts):
        for text in texts:
            if re.match(r'^FIC\s+[A-Z]{3}', text):
                return text
            if re.match(r'^\d{3}\.?\d*', text):
                return text
        return None
    
    def scan_books(self, frame):
        detected = self.detect_text_vision(frame)
        
        if not detected or not detected['text_annotations']:
            return []
        
        clusters = self.cluster_books_by_gap(detected['text_annotations'], gap_threshold=20)
        
        if not clusters:
            return []
        
        search_tasks = []
        for cluster in clusters:
            texts = [w['text'] for w in cluster]
            call_number = self.extract_call_number(texts)
            search_tasks.append((texts, call_number))
        
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
                except:
                    pass
        
        results.sort(key=lambda x: x['position'])
        
        return results


class BookScannerGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Book Scanner")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2b2b2b')
        
        try:
            self.scanner = BookScanner()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.root.quit()
            return
        
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
                              font=('Arial', 24, 'bold'),
                              bg='#2b2b2b', fg='white')
        title_label.pack(pady=10)
        
        main_container = tk.Frame(self.root, bg='#2b2b2b')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10)
        
        left_frame = tk.Frame(main_container, bg='#2b2b2b')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        video_frame = tk.Frame(left_frame, bg='black', relief=tk.SUNKEN, bd=2)
        video_frame.pack(pady=5)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack()
        
        right_frame = tk.Frame(main_container, bg='#1e1e1e', relief=tk.SUNKEN, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.config(width=350)
        
        results_title = tk.Label(right_frame, text="Detected Books", 
                                font=('Arial', 14, 'bold'),
                                bg='#1e1e1e', fg='white')
        results_title.pack(pady=10)
        
        self.results_frame = tk.Frame(right_frame, bg='#1e1e1e')
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.no_results_label = tk.Label(self.results_frame, 
                                         text="Press SCAN BOOKS\nto start",
                                         font=('Arial', 12),
                                         bg='#1e1e1e', fg='#888888')
        self.no_results_label.pack(expand=True)
        
        bottom_frame = tk.Frame(self.root, bg='#2b2b2b')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        button_container = tk.Frame(bottom_frame, bg='#2b2b2b')
        button_container.pack()
        
        self.scan_button = tk.Button(button_container, text="SCAN BOOKS", 
                                     command=self.scan_books,
                                     font=('Arial', 16, 'bold'),
                                     bg='#4CAF50', fg='white',
                                     padx=40, pady=15,
                                     relief=tk.RAISED, bd=3)
        self.scan_button.pack(side=tk.LEFT, padx=10)
        
        quit_button = tk.Button(button_container, text="QUIT",
                               command=self.quit_app,
                               font=('Arial', 16),
                               bg='#f44336', fg='white',
                               padx=40, pady=15,
                               relief=tk.RAISED, bd=3)
        quit_button.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(bottom_frame, text="Ready to scan", 
                                     font=('Arial', 12), 
                                     bg='#2b2b2b', fg='#4CAF50')
        self.status_label.pack(pady=5)
    
    def update_video(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                if not self.scanning:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = frame_rgb.shape[:2]
                    max_width = 700
                    if width > max_width:
                        scale = max_width / width
                        frame_rgb = cv2.resize(frame_rgb, 
                                              (int(width * scale), int(height * scale)))
                    
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
        except Exception as e:
            print(f"Video update error: {e}")
        
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
                                 text="Scanning...\nPlease wait",
                                 font=('Arial', 12),
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
        print(f"Displaying {len(results)} results")  
        
        try:
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            
            if not results:
                no_books = tk.Label(self.results_frame, 
                                   text="No books detected\n\nTips:\n• Ensure good lighting\n• Hold camera steady\n• Face book spines to camera",
                                   font=('Arial', 11),
                                   bg='#1e1e1e', fg='#888888',
                                   justify=tk.LEFT)
                no_books.pack(pady=20)
                self.status_label.config(text="No books found", fg='orange')
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
                
                print(f"Creating {len(results)} book cards")  
                
                for idx, book in enumerate(results):
                    print(f"Creating card for book {idx+1}: {book.get('title', 'Unknown')}")  # Debug
                    
                    book_frame = tk.Frame(scrollable_frame, bg='#2d2d2d', relief=tk.RAISED, bd=1)
                    book_frame.pack(fill=tk.X, padx=5, pady=5)
                    
                    position_text = f"#{book['position']}"
                    if book['position'] == 1:
                        position_text += " (Bottom)"
                    elif book['position'] == len(results):
                        position_text += " (Top)"
                    
                    pos_label = tk.Label(book_frame, text=position_text,
                                        font=('Arial', 10, 'bold'),
                                        bg='#4CAF50', fg='white',
                                        padx=5, pady=2)
                    pos_label.pack(anchor=tk.W, padx=5, pady=(5, 0))
                    
                    title_label = tk.Label(book_frame, text=book.get('title', 'Unknown'),
                                          font=('Arial', 11, 'bold'),
                                          bg='#2d2d2d', fg='white',
                                          wraplength=300, justify=tk.LEFT)
                    title_label.pack(anchor=tk.W, padx=10, pady=(5, 0))
                    
                    author_label = tk.Label(book_frame, text=f"by {book.get('author_full', 'Unknown')}",
                                           font=('Arial', 10),
                                           bg='#2d2d2d', fg='#cccccc')
                    author_label.pack(anchor=tk.W, padx=10)
                    
                    year_label = tk.Label(book_frame, text=f"Published: {book.get('first_publish_year', 'Unknown')}",
                                         font=('Arial', 9),
                                         bg='#2d2d2d', fg='#999999')
                    year_label.pack(anchor=tk.W, padx=10, pady=(0, 5))
                    
                    if book.get('call_number'):
                        call_label = tk.Label(book_frame, text=f"Call#: {book['call_number']}",
                                             font=('Arial', 9),
                                             bg='#2d2d2d', fg='#999999')
                        call_label.pack(anchor=tk.W, padx=10, pady=(0, 5))
                
                print("Packing canvas and scrollbar")  
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                self.status_label.config(text=f"Found {len(results)} book(s)", fg='#4CAF50')
            
            print("Setting scanning to False")  
            self.scanning = False
            self.scan_button.config(state=tk.NORMAL, bg='#4CAF50')
            print("Display complete")  
            
        except Exception as e:
            print(f"Error in _display_results: {e}")  
            import traceback
            traceback.print_exc()
            self.scanning = False
            self.scan_button.config(state=tk.NORMAL, bg='#4CAF50')
    
    def _display_error(self, error_msg):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        error_label = tk.Label(self.results_frame, 
                              text=f"Error:\n{error_msg}",
                              font=('Arial', 11),
                              bg='#1e1e1e', fg='#ff5555')
        error_label.pack(pady=20)
        
        self.status_label.config(text="Scan failed", fg='red')
        self.scanning = False
        self.scan_button.config(state=tk.NORMAL, bg='#4CAF50')
        self.update_video()
    
    def quit_app(self):
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = BookScannerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()