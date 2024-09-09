from ultralytics import YOLO

model = YOLO('models/best_yolo_goalvision.pt')

results = model.predict('10svideo.mp4', save=True)
print(results[0])
print("========================================================")
for box in results[0].boxes:
    print(box)