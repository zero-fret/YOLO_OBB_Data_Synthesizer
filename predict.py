from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBPredictor
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == '__main__':
    model = YOLO('./runs/obb/result/test-1/weights/best.pt')
    print(f"Model Type: {type(model.model)}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters())}")
    
    args = dict(
        imgsz=1280,
        conf=0.1,
        model="./runs/obb/result/test-1/weights/best.pt",
        source="./test",
        project="./predict",
    )
    
    predictor = OBBPredictor(overrides=args)
    predictor.predict_cli()
