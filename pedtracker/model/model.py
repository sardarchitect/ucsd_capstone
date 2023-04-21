import torch

model = torch.hub.load('ultralytics/yolov', 'yolovs', pretrained=True)

imgs = []

results = model(imgs)
results.print()