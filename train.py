import os
import sys
import time
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_mobilenet_v2(num_classes):
    print(f"[*] Initializing MobileNetV2 with ImageNet pre-trained weights...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze the early feature extraction layers
    print("[*] Freezing early layers (features[:15]) to preserve low-level filters...")
    for param in model.features[:15].parameters():
        param.requires_grad = False
        
    # Replace final classification head
    in_features = model.last_channel
    model.classifier[1] = nn.Linear(in_features, num_classes)
    print(f"[*] Replaced classification head: Linear(in={in_features}, out={num_classes})")
    
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n" + "=" * 60)
    print(f"  STARTING MODEL TRAINING ON: {DEVICE}")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            dataset_size = len(dataloader.dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"  [{phase.upper():5s}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("\n" + "=" * 60)
    print(f"[✓] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"[✓] Best Validation Accuracy: {best_acc*100:.2f}%")
    print("=" * 60)

    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Skin Lesion Classification")
    parser.add_argument("--data_dir", type=str, default="", help="Path to dataset directory (ImageFolder format)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--save_path", type=str, default="trained_model.pth", help="Path to save weights")
    parser.add_argument("--demo", action="store_true", help="Run in quick demonstration mode with synthetic batches")
    args = parser.parse_args()

    classes = ['Eczema', 'Melanoma', 'Psoriasis']

    if args.demo or not args.data_dir or not os.path.exists(args.data_dir):
        print("[!] No data directory provided or --demo selected. Running in Interactive Demo Mode...")
        print("[*] Generating demonstration batches to showcase terminal training pipeline...")
        # Create synthetic batches for live demonstration
        torch.manual_seed(42)
        X_train = torch.randn(64, 3, 224, 224)
        y_train = torch.randint(0, len(classes), (64,))
        X_val = torch.randn(32, 3, 224, 224)
        y_val = torch.randint(0, len(classes), (32,))

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
    else:
        # Standard torchvision ImageFolder pipeline
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        full_train = datasets.ImageFolder(args.data_dir, transform=train_transforms)
        full_val = datasets.ImageFolder(args.data_dir, transform=val_transforms)
        classes = full_train.classes
        
        train_size = int(0.8 * len(full_train))
        indices = torch.randperm(len(full_train)).tolist()
        train_loader = DataLoader(Subset(full_train, indices[:train_size]), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(full_val, indices[train_size:]), batch_size=16, shuffle=False)

    print(f"[*] Target Classes: {classes}")
    model = build_mobilenet_v2(num_classes=len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs
    )

    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'classes': classes
    }, args.save_path)
    print(f"[✓] Checkpoint saved successfully to: {args.save_path}\n")

if __name__ == "__main__":
    main()
