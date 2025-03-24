import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import timm # type: ignore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# model = models.resnet50(pretrained=False)
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 7)

model_name = "efficientnet_b0"
model = timm.create_model(model_name, pretrained=True, num_classes=7)
model.to(device)


# model.load_state_dict(torch.load("./Best-model/best_resnet50_model.pth", map_location=device))
# model.to(device)
# model.eval()

model.load_state_dict(torch.load("./Best-model/best_efficientnet_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)  # 0 = webcam หลักของเครื่อง

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

    # with torch.no_grad():
    #     outputs = model(input_tensor)
    #     _, predicted = torch.max(outputs, 1)
    #     pred_class = classes[predicted.item()]

    # cv2.putText(frame, pred_class, (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = frame[y:y+h, x:x+w]  # BGR

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        face_tensor = transform(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_class = classes[predicted.item()]

        cv2.putText(frame, pred_class, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
