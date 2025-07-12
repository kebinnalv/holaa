from ultralytics import YOLO  

model = YOLO("yolov8n.pt")  


model.train(data="baseball-1/data.yaml", epochs=50, imgsz=640)  



