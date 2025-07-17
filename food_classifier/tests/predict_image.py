from PIL import Image
import torch
from torchvision import transforms
from foodclassifier.helpers import get_transforms

def predict_image(image_path, model, device, class_names):
    model.eval()

    # ✅ Consistent transforms — same as training!
    transform = get_transforms()

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    print(f"Predicted: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
