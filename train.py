from ultralytics import YOLO
import os, random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_MAX_POWER"] = "80"

random.seed(1337)

if __name__ == '__main__':
    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    model = YOLO('yolo11n-obb.pt')
    print(f"Model: {type(model.model)}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters())}")
    
    results = model.train(
        data="./obb.yaml",
        epochs=60,
        imgsz=1280,
        batch=4,
        amp=False,

        # Learning rate
        lr0=0.00001,
        lrf=0.0000001,
        momentum=0.9,
        weight_decay=0.0005,
        
        # Loss
        box=1,
        cls=3,
        dfl=1,
        
        # Data Augmentation
        translate=0,
        mosaic=0,
        degrees=0,
        scale=0,
        perspective=0,
        flipud=0.5,
        hsv_h=0.015,
        hsv_v=0.1,
        copy_paste=0.3,
        
        # Training
        patience=30,
        workers=8,
        
        # 
        project="./result",
        name='test',
        save=True,
        save_period=1,
        val=True,
        plots=True,
    )
