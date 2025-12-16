import arabic_reshaper
import cv2
import numpy as np
import streamlit as st
from bidi.algorithm import get_display
from PIL import Image
from ultralytics import YOLO

# Page Config
st.set_page_config(
    page_title="Egyptian LPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        border-left: 4px solid #1f77b4;
        padding-left: 10px;
        margin: 1rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 10px 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Constants
PLATE_MODEL_PATH = "runs_plate/plate_model/weights/best.pt"
CHAR_MODEL_PATH = "runs_chars/char_model_no_flip/weights/best.pt"

# Character Mapping
CLASS_MAPPING = {
    "0": "٠",
    "1": "١",
    "2": "٢",
    "3": "٣",
    "4": "٤",
    "5": "٥",
    "6": "٦",
    "7": "٧",
    "8": "٨",
    "9": "٩",
    "alif": "ا",
    "baa": "ب",
    "ta": "ت",
    "taa": "ت",
    "thaa": "ث",
    "jeem": "ج",
    "7aa": "ح",
    "khaa": "خ",
    "daal": "د",
    "zaal": "ذ",
    "raa": "ر",
    "zay": "ز",
    "seen": "س",
    "sheen": "ش",
    "saad": "ص",
    "daad": "ض",
    "Taa": "ط",
    "Thaa": "ظ",
    "ain": "ع",
    "ghayn": "غ",
    "faa": "ف",
    "qaaf": "ق",
    "kaaf": "ك",
    "laam": "ل",
    "meem": "م",
    "noon": "ن",
    "haa": "ه",
    "waw": "و",
    "yaa": "ي",
}


@st.cache_resource
def load_models():
    """Load YOLO models (cached for performance)"""
    plate_model = YOLO(PLATE_MODEL_PATH)
    char_model = YOLO(CHAR_MODEL_PATH)
    return plate_model, char_model


def enhance_plate(img):
    """Enhance plate image for better character recognition"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), gray, upscaled, enhanced


def process_image(img, plate_model, char_model):
    """
    Process image through the full LPR pipeline
    Returns all intermediate steps for visualization
    """
    results = {
        "original": img.copy(),
        "plate_detected": None,
        "plate_crop": None,
        "grayscale": None,
        "upscaled": None,
        "enhanced": None,
        "char_detection": None,
        "numbers": "",
        "letters": "",
        "full_plate": "",
        "success": False,
        "message": "",
    }

    # Stage 1: Plate Detection
    plate_results = plate_model(img, verbose=False)

    if len(plate_results[0].boxes) == 0:
        results["message"] = "No license plate detected in the image"
        return results

    # Get plate coordinates
    box = plate_results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    # Add padding
    h, w, _ = img.shape
    padding = 10
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

    # Draw detection on original
    plate_detected = img.copy()
    cv2.rectangle(plate_detected, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        plate_detected,
        f"Plate: {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    results["plate_detected"] = plate_detected

    # Crop plate
    plate_crop = img[y1:y2, x1:x2]
    results["plate_crop"] = plate_crop.copy()

    # Stage 2: Enhancement
    enhanced_plate, gray, upscaled, enhanced = enhance_plate(plate_crop)
    results["grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    results["upscaled"] = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
    results["enhanced"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Stage 3: Character Detection
    char_results = char_model(
        enhanced_plate, verbose=False, conf=0.35, agnostic_nms=True
    )

    numbers_list = []
    letters_list = []
    char_detection = enhanced_plate.copy()

    for box in char_results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = char_model.names[cls_id]
        char_conf = float(box.conf[0])
        cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])

        # Draw bounding box
        color = (255, 0, 0) if class_name.isdigit() else (0, 0, 255)
        cv2.rectangle(char_detection, (cx1, cy1), (cx2, cy2), color, 2)
        cv2.putText(
            char_detection,
            f"{class_name}",
            (cx1, cy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        if class_name.isdigit():
            numbers_list.append((cx1, class_name, char_conf))
        else:
            letters_list.append((cx1, class_name, char_conf))

    results["char_detection"] = char_detection

    # Sort: numbers LTR, letters LTR (bidi will handle RTL display)
    numbers_list.sort(key=lambda x: x[0])
    letters_list.sort(key=lambda x: x[0])

    # Convert to Arabic
    final_nums = "".join([CLASS_MAPPING.get(n[1], n[1]) for n in numbers_list])
    raw_letters = " ".join([CLASS_MAPPING.get(l[1], l[1]) for l in letters_list])

    # Apply Bidi for proper Arabic display
    reshaped_letters = arabic_reshaper.reshape(raw_letters)
    bidi_letters = get_display(reshaped_letters)

    results["numbers"] = final_nums
    results["letters"] = bidi_letters
    results["full_plate"] = f"{bidi_letters} | {final_nums}"
    results["success"] = True
    results["message"] = "License plate recognized successfully!"

    return results


def main():
    # Sidebar with team members
    with st.sidebar:
        st.markdown("### Team Members")
        st.markdown(
            """
        - Ammar Gamal
        - Marwan Hossam
        - Mazen Ashraf
        - Youssif Monir
        - Sohaila Tamer
        - Youstina Adel
        - Roba Tarek
        - Youmna Hesham
        - Anas Ahmed
        """
        )

    # Header
    st.markdown(
        '<h1 class="main-header">Egyptian License Plate Recognition</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Upload an image to detect and recognize Egyptian license plates</p>',
        unsafe_allow_html=True,
    )

    # Load models
    with st.spinner("Loading AI models..."):
        plate_model, char_model = load_models()

    # File uploader with drag and drop
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Drag and drop an image here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process button
        if st.button("Process Image", type="primary", use_container_width=True):
            with st.spinner("Processing image..."):
                results = process_image(img, plate_model, char_model)

            # Show status
            if results["success"]:
                st.success(results["message"])
            else:
                st.error(results["message"])
                st.image(
                    cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB),
                    caption="Original Image",
                    use_container_width=True,
                )
                return

            # Display Results
            st.markdown("---")
            st.markdown("## Recognition Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                <div class="result-box">
                    <strong>Numbers</strong><br>
                    <span style="font-size: 2rem;">{results['numbers']}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="result-box">
                    <strong>Letters</strong><br>
                    <span style="font-size: 2rem;">{results['letters']}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="result-box">
                    <strong>Full Plate</strong><br>
                    <span style="font-size: 2rem;">{results['full_plate']}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Processing Pipeline
            st.markdown("---")
            st.markdown("## Processing Pipeline")

            # Step 1: Original Image
            st.markdown(
                '<p class="step-header">Step 1: Original Image</p>',
                unsafe_allow_html=True,
            )
            st.image(
                cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB),
                caption="Input Image",
                use_container_width=True,
            )

            # Step 2: Plate Detection
            st.markdown(
                '<p class="step-header">Step 2: License Plate Detection (YOLOv8)</p>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    cv2.cvtColor(results["plate_detected"], cv2.COLOR_BGR2RGB),
                    caption="Detected Plate with Bounding Box",
                )
            with col2:
                st.image(
                    cv2.cvtColor(results["plate_crop"], cv2.COLOR_BGR2RGB),
                    caption="Cropped Plate Region",
                )

            # Step 3: Image Enhancement
            st.markdown(
                '<p class="step-header">Step 3: Image Enhancement (Preprocessing)</p>',
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(
                    cv2.cvtColor(results["grayscale"], cv2.COLOR_BGR2RGB),
                    caption="Grayscale Conversion",
                )
            with col2:
                st.image(
                    cv2.cvtColor(results["upscaled"], cv2.COLOR_BGR2RGB),
                    caption="2x Upscaling (Cubic Interpolation)",
                )
            with col3:
                st.image(
                    cv2.cvtColor(results["enhanced"], cv2.COLOR_BGR2RGB),
                    caption="CLAHE Enhancement",
                )

            # Step 4: Character Detection
            st.markdown(
                '<p class="step-header">Step 4: Character Detection & Recognition (YOLOv8)</p>',
                unsafe_allow_html=True,
            )
            st.image(
                cv2.cvtColor(results["char_detection"], cv2.COLOR_BGR2RGB),
                caption="Detected Characters (Blue = Numbers, Red = Letters)",
                use_container_width=True,
            )

            # Pipeline Info
            st.markdown("---")
            st.markdown("## Pipeline Information")
            with st.expander("View Technical Details"):
                st.markdown(
                    """
                ### Processing Steps:
                
                1. **Plate Detection**: YOLOv8 model trained on Egyptian license plates
                2. **Cropping**: Extract plate region with padding
                3. **Grayscale Conversion**: Convert to single channel
                4. **Upscaling**: 2x cubic interpolation for better detail
                5. **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
                6. **Character Detection**: YOLOv8 model trained on Arabic/numeric characters
                7. **Character Sorting**: Numbers sorted LTR, Letters sorted RTL
                8. **Arabic Mapping**: Convert class names to Arabic characters
                """
                )

        else:
            # Show preview
            st.markdown("### Image Preview")
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=f"Uploaded: {uploaded_file.name}",
                use_container_width=True,
            )

    else:
        # Placeholder when no image uploaded
        st.markdown(
            """
        <div style="
            border: 2px dashed #ccc;
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            background-color: #fafafa;
            margin: 20px 0;
        ">
            <h3 style="color: #888;">Upload an Image</h3>
            <p style="color: #aaa;">Drag and drop or click to select a file</p>
            <p style="color: #aaa; font-size: 0.9rem;">Supported: JPG, JPEG, PNG</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #888; padding: 20px;">
        <p>Egyptian License Plate Recognition System</p>
        <p style="font-size: 0.8rem;">Powered by YOLOv8 & Streamlit</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
