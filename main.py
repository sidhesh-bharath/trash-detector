from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("best.pt")
img_path = Path("image.png")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

results = model.predict(source=str(img_path), conf=0.5, imgsz=416, save=False)
annotated_img = results[0].plot()
output_path = output_dir / f"{img_path.stem}-output{img_path.suffix}"

cv2.imwrite(str(output_path), annotated_img)
print(f"Detection saved to: {output_path}")

for box in results[0].boxes:
    coords = box.xyxy[0].tolist()
    conf = float(box.conf[0])*100
    cls = int(box.cls[0])
    label = results[0].names[cls]
    print(f"Detected {label}:")
    print(f"  - Bounding Box: {coords}")
    print(f"  - Confidence: {conf:.2f}%\n")

# use list: coords for bounding box coords

cv2.imshow("YOLO Detection Result", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()