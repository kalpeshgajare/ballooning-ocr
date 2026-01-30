from ultralytics import YOLO

# Load the 'x' version for maximum precision
model = YOLO('yolo11x.pt',task='detect') 

# Train on Apple Silicon GPU (MPS)
model.train(
    data='OCR.v1-new_dataset_v1.yolov11/data.yaml',
    epochs=150,
    imgsz=1280,      # High res for thin CAD lines
    device='mps',    # <--- CRUCIAL: Enables Mac GPU acceleration
    batch=4,         # Low batch due to the large 'x' model size
    mosaic=1.0,      
    patience=50,
    amp=False        # Recommendation: Disable AMP if you see stability issues on MPS
)