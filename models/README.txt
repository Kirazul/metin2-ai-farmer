# YOLOv8 Model

Place your trained YOLOv8 model in this directory.

## Required:

**best.pt** - Your trained YOLOv8 model for metin detection

## File Location:

The bot expects the model at: `models/best.pt`

If you have a different model name, either:
1. Rename it to `best.pt`, OR
2. Update the model path in `detector.py` line 12

## Model Requirements:

- YOLOv8 format (Ultralytics)
- Trained to detect Metin2 stones/metins
- Should output bounding boxes for detected metins
- Recommended: YOLOv8n or YOLOv8s for speed

## If you don't have a model:

You'll need to train one using:
1. Collect Metin2 screenshots with metins
2. Label the metins using tools like LabelImg or Roboflow
3. Train using Ultralytics YOLOv8
4. Export the best.pt model

More info: https://docs.ultralytics.com/
