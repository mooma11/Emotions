import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import timm  # type: ignore
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === emotion ===
emotion_model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=7)
emotion_model.load_state_dict(torch.load("./Best-model/best_efficientnet_model.pth", map_location=device))
emotion_model.to(device)
emotion_model.eval()
emotion_classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# === gender ===
gender_model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
gender_model.load_state_dict(torch.load("./M-FM/best_efficientnet_model.pth", map_location=device))
gender_model.to(device)
gender_model.eval()
gender_classes = ["Female", "Male"]

# === Image Transform ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

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

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            # Emotion Prediction
            emo_output = emotion_model(face_tensor)
            _, emo_pred = torch.max(emo_output, 1)
            emo_class = emotion_classes[emo_pred.item()]

            # Gender Prediction
            gender_output = gender_model(face_tensor)
            _, gender_pred = torch.max(gender_output, 1)
            gender_class = gender_classes[gender_pred.item()]

        label = f"{gender_class}, {emo_class}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Gender & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
