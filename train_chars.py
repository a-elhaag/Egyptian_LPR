from ultralytics import YOLO

def train_character_detector():
    model = YOLO('yolov8n.pt') 

    # Train on the character dataset
    results = model.train(
    data='datasets/dataset_characters/data.yaml',
    epochs=50,                  
    imgsz=640,
    project='runs_chars',
    name='char_model_no_flip',
    fliplr=0.0                                 #Disable Left-Right Flip (Default is 0.5)
    )

    print("Training Finished! Best model saved in runs_chars/char_model/weights/best.pt")

if __name__ == '__main__':
    train_character_detector()