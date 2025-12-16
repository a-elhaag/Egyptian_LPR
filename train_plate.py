from ultralytics import YOLO

def train_plate_detector():
    model = YOLO('yolov8n.pt') 

    # Train on the plates dataset
    results = model.train(
        data='datasets/dataset_plates/data.yaml',  
        epochs=30,                                 
        imgsz=640,                                 
        project='runs_plate',                      
        name='plate_model'                         
    )

    print("Training Finished! Best model saved in runs_plate/plate_model/weights/best.pt")

if __name__ == '__main__':
    train_plate_detector()