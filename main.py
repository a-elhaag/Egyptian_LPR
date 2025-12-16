import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import arabic_reshaper
from bidi.algorithm import get_display


# Constants
PLATE_MODEL_PATH = 'runs_plate/plate_model/weights/best.pt'
CHAR_MODEL_PATH = 'runs_chars/char_model_no_flip/weights/best.pt' 


# Full Objects Mapping
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


# Image Sharpener
def enhance_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(upscaled)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# Detection
def detect_and_read(image_path):
    print(f"\n[INFO] Loading models...")
    plate_model = YOLO(PLATE_MODEL_PATH)
    char_model = YOLO(CHAR_MODEL_PATH)
    
    print(f"[INFO] Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the path.")
        return



    # Stage(1): Plate Detection
    results = plate_model(img, verbose=False)
    if len(results[0].boxes) == 0:
        print("No license plate found.")
        return

    #yolo gets the coordinates of the best plate
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    #padding
    h, w, _ = img.shape
    padding = 10
    x1, y1, x2, y2 = max(0, x1-padding), max(0, y1-padding), min(w, x2+padding), min(h, y2+padding)
    plate_img = img[y1:y2, x1:x2]
    clean_plate = enhance_plate(plate_img)



    # Stage(2): Character Recognition
    #agnostic
    char_results = char_model(clean_plate, verbose=False, conf=0.35, agnostic_nms=True)
    numbers_list = []
    letters_list = []
    

    #labeling the class id and name on the plate's objects
    for box in char_results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = char_model.names[cls_id]
        cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
        
        cv2.rectangle(clean_plate, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        cv2.putText(clean_plate, class_name, (cx1, cy1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if class_name.isdigit():
            numbers_list.append((cx1, class_name))
        else:
            letters_list.append((cx1, class_name))


    #sorting for arabic letters (right to left)
    numbers_list.sort(key=lambda x: x[0])
    letters_list.sort(key=lambda x: x[0], reverse=True)


    #convert the labels from english to arbic
    final_nums = [CLASS_MAPPING.get(n, n) for _, n in numbers_list]
    final_letters = [CLASS_MAPPING.get(l, l) for _, l in letters_list]
    nums_str = "".join(final_nums)
    letters_str = " ".join(final_letters)
    

    #shaping & bidi for terminal display
    bidi_letters = get_display(arabic_reshaper.reshape(letters_str))
    print(f"\n" + "="*30)
    print(f" 🇪🇬  EGYPTIAN LPR RESULT  🇪🇬")
    print(f"="*30)
    print(f"NUMBERS : {nums_str}")
    print(f"LETTERS : {bidi_letters}")
    print(f"FULL    : {nums_str} || {bidi_letters}")
    print(f"="*30 + "\n")

    cv2.imshow("Final Result", clean_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#python main.py --image "path/to/img.jpg"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Egyptian License Plate Recognition")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image file")
    args = parser.parse_args()
    
    detect_and_read(args.image)