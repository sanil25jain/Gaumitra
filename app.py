from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import json
import os

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open('/Users/SJ/Documents/lmage-based-breed-recognition-for-cattle-and-buffaloes-of-India/breeds_info.json') as f:
    breeds_info = json.load(f)

classes = list(breeds_info.keys())

NUM_CLASSES = len(classes)

MODEL_PATH = 'best_model_final.pth'

def load_model():
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    prediction_info = None
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            upload_dir = 'static/uploads'
            os.makedirs(upload_dir, exist_ok=True)
            img_path = os.path.join(upload_dir, img_file.filename)
            img_file.save(img_path)

            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            img_tensor = preprocess_image(image)

            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                predicted_idx = torch.argmax(outputs, 1).item()
                prediction = classes[predicted_idx]
                prediction_info = breeds_info.get(prediction, {
                    "description": "No info available.",
                    "primary_purpose": "",
                    "average_milk_yield": "",
                    "region": "",
                    "physical_traits": "",
                    "management_tip": ""
                })

    return render_template('index.html', prediction=prediction, prediction_info=prediction_info)

if __name__ == "__main__":
    app.run(debug=True)

