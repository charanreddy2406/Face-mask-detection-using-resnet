# Face-mask-detection-using-resnet
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

class FaceMaskDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2)  # Adjusting the output layer for binary classification
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        _, predicted = torch.max(outputs, 1)
        if predicted.item() == 0:
            return "With Mask"
        else:
            return "Without Mask"

# Guys, change to ur path..!
if __name__ == "__main__":
    model_path = "path/to/your/pretrained/model.pth"
    face_mask_detector = FaceMaskDetector(model_path)
    result = face_mask_detector.predict("path/to/your/image.jpg")
    print("Mask Status:", result)

# See you again ..!
