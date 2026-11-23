import torch
import torch.nn as nn
from torchvision import models, transforms
from src.config import DEVICE, IMAGE_SIZE

def get_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def build_mobilenet_v2(num_classes, pretrained=True, freeze_features=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    if freeze_features:
        for param in model.features[:15].parameters():
            param.requires_grad = False
            
    in_features = model.last_channel
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)

def load_checkpoint(weights_path):
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    return model, classes, idx_to_class
