import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import arabic_reshaper
from bidi.algorithm import get_display


# Cinstants
PLATE_MODEL_PATH = 'runs_plate/plate_model/weights/best.pt'
CHAR_MODEL_PATH = 'runs_chars/char_model_no_flip/weights/best.pt'


# Mapping
CLASS_MAPPING = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', 
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
    'alif': 'ا', 'baa': 'ب', 'ta': 'ت', 'taa': 'ت', 'thaa': 'ث',
    'jeem': 'ج', '7aa': 'ح', 'khaa': 'خ', 'daal': 'د', 'zaal': 'ذ',
    'raa': 'ر', 'zay': 'ز', 'seen': 'س', 'sheen': 'ش', 'saad': 'ص',
    'daad': 'ض', 'Taa': 'ط', 'Thaa': 'ظ', 'ain': 'ع', 'ghayn': 'غ',
    'faa': 'ف', 'qaaf': 'ق', 'kaaf': 'ك', 'laam': 'ل', 'meem': 'م',
    'noon': 'ن', 'haa': 'ه', 'waw': 'و', 'yaa': 'ي'
}


class EgyptianLPRApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        #Window Setup
        self.title("Egyptian License Plate Recognition")
        self.geometry("900x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        #Load Models at startup
        self.plate_model = YOLO(PLATE_MODEL_PATH)
        self.char_model = YOLO(CHAR_MODEL_PATH)

        #UI Layout
        self.create_widgets()

    def create_widgets(self):
        # 1. Header
        self.lbl_title = ctk.CTkLabel(self, text="Egyptian LPR System", font=("Arial", 24, "bold"))
        self.lbl_title.pack(pady=20)

        # 2. Main Image Display Area
        self.img_label = ctk.CTkLabel(self, text="Please upload an image", width=640, height=360, corner_radius=10, fg_color="#2b2b2b")
        self.img_label.pack(pady=10)

        # 3. Result Text Area
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(pady=20, fill="x", padx=40)

        self.lbl_numbers = ctk.CTkLabel(self.result_frame, text="Numbers: ---", font=("Arial", 20))
        self.lbl_numbers.pack(side="left", padx=20, pady=10)

        self.lbl_letters = ctk.CTkLabel(self.result_frame, text="Letters: ---", font=("Arial", 20))
        self.lbl_letters.pack(side="right", padx=20, pady=10)

        # 4. Upload Button
        self.btn_upload = ctk.CTkButton(self, text="Upload Image", command=self.upload_image, font=("Arial", 16), height=50)
        self.btn_upload.pack(pady=10)

    def enhance_plate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(upscaled)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        # Run detection
        self.process_image(file_path)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        
        # Stage(1): Plate Detection
        results = self.plate_model(img, verbose=False)
        if len(results[0].boxes) == 0:
            self.lbl_numbers.configure(text="No Plate Found")
            return

        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Padding
        h, w, _ = img.shape
        padding = 10
        x1, y1, x2, y2 = max(0, x1-padding), max(0, y1-padding), min(w, x2+padding), min(h, y2+padding)
        
        plate_img = img[y1:y2, x1:x2]
        clean_plate = self.enhance_plate(plate_img)

        # Stage(2): Character Detection
        char_results = self.char_model(clean_plate, verbose=False, conf=0.35, agnostic_nms=True)
        
        numbers_list = []
        letters_list = []
        
        for box in char_results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = self.char_model.names[cls_id]
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])

            # Draw Box on Clean Plate
            cv2.rectangle(clean_plate, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.putText(clean_plate, class_name, (cx1, cy1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if class_name.isdigit():
                numbers_list.append((cx1, class_name))
            else:
                letters_list.append((cx1, class_name))

        # Sort
        numbers_list.sort(key=lambda x: x[0])                # LTR
        letters_list.sort(key=lambda x: x[0], reverse=True)  # RTL

        # Format Text
        final_nums = "".join([CLASS_MAPPING.get(n[1], n[1]) for n in numbers_list])
        raw_letters = " ".join([CLASS_MAPPING.get(l[1], l[1]) for l in letters_list])
        
        # Bidi Logic for GUI
        reshaped_letters = arabic_reshaper.reshape(raw_letters)
        bidi_letters = get_display(reshaped_letters)

        # Update GUI
        self.lbl_numbers.configure(text=f"Numbers: {final_nums}")
        self.lbl_letters.configure(text=f"Letters: {bidi_letters}")

        # Display Image (Convert OpenCV BGR to PIL RGB)
        img_rgb = cv2.cvtColor(clean_plate, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to fit the label
        ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 360))
        self.img_label.configure(image=ctk_img, text="") 


if __name__ == "__main__":
    app = EgyptianLPRApp()
    app.mainloop()