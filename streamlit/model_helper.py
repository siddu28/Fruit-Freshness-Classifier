from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

trained_model = None

class_names = ['Fresh_Banana', 'Fresh_Lemon', 'Fresh_Lulo', 'Fresh_Mango', 'Fresh_Orange', 'Fresh_Strawberry', 'Fresh_Tamarillo', 'Fresh_Tomato',
               'Spoiled_Banana', 'Spoiled_Lemon', 'Spoiled_Lulo', 'Spoiled_Mango', 'Spoiled_Orange', 'Spoiled_Strawberry', 'Spoiled_Tamarillo', 'Spoiled_Tomato']

class fruitClassificationResnet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers except layer4
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Image transform (reuse this outside if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to open image: {e}")
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    global trained_model
    if trained_model is None:
        trained_model = fruitClassificationResnet(num_classes=len(class_names))
        trained_model.load_state_dict(torch.load("streamlit/model/ResNet50_model.pth", map_location=torch.device('cpu')))
        trained_model.eval()

    with torch.no_grad():
        outputs = trained_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
