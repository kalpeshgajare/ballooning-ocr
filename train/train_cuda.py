from ultralytics import YOLO

# Load the pretrained model
model = YOLO('yolo11x.pt') 

# Start training
model.train(
    data='OCR.v1-new_dataset_v1.yolov11/data.yaml',
    epochs=150,
    imgsz=1280,      # Maintain high res for CAD measurements
    device=0,        # <--- Uses your first NVIDIA GPU
    batch=4,         # Adjust based on your VRAM (e.g., 8GB vs 12GB)
    mosaic=1.0,      
    patience=50,
    amp=True         # NVIDIA GPUs benefit from Automatic Mixed Precision (AMP)
)
