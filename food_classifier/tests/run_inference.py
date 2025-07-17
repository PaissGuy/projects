import torch
from torchvision import transforms
from foodclassifier.model import build_model
from tests.predict_image import predict_image


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=101)
model.load_state_dict(torch.load("../models/resnet_food101.pth", map_location=device))  # Load trained weights
model.to(device)



# Load class names (can be saved from training with: `train_data.classes`)
from torchvision.datasets import Food101
data_root = r"C:\Users\Mashtock\Documents\Courses\M2M\Capstone1\Capstone_Project\foodclassifier\data"
class_names = Food101(root=data_root, split="train").classes

# Run inference
image_path = "example4.jpg"
predict_image(image_path, model, device, class_names)
