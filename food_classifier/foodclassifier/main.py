import torch
from torchvision import datasets, transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
from foodclassifier.model import build_model, train_model, evaluate_model, evaluate_full
from helpers import get_transforms

def main():


    train_data = Food101(root="data", split="train", download=True, transform=get_transforms())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    test_data = Food101(root="data", split="test", download=True, transform=get_transforms())
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(num_classes=len(train_data.classes))
    train_model(model, train_loader, device)

    evaluate_model(model, test_loader, device)
    evaluate_full(model, test_loader, device)

    torch.save(model.state_dict(), "../models/resnet_food101.pth")

if __name__ == "__main__":
    print(torch.__version__)  # Should print 1.10.1
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    main()