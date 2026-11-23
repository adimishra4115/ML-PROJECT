import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    classes = checkpoint['classes']
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model, classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_cascaded_diagnosis(image_path, gatekeeper_weights="gatekeeper_model.pth", severity_weights="severity_model.pth"):
    print("=" * 60)
    print("  CASCADED DERMATOLOGICAL DIAGNOSIS SYSTEM (TERMINAL)")
    print("=" * 60)
    print(f"[*] Running on device: {DEVICE}")
    print(f"[*] Loading input image: {image_path}")
    
    raw_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    
    # ----------------- STAGE 1: GATEKEEPER -----------------
    print("\n[Stage 1] Loading Gatekeeper Model (Screening)...")
    gatekeeper_model, gatekeeper_classes = load_model(gatekeeper_weights)
    
    with torch.no_grad():
        out1 = gatekeeper_model(input_tensor)
        probs1 = torch.softmax(out1, dim=1)[0]
        pred_idx1 = torch.argmax(probs1).item()
        pred_class1 = gatekeeper_classes[pred_idx1]
        confidence1 = probs1[pred_idx1].item() * 100
        
    print(f"--> Stage 1 Diagnosis: \033[1;32m{pred_class1}\033[0m")
    print(f"--> Confidence: {confidence1:.2f}%")
    print("--> Class Probabilities:")
    for cls, prob in zip(gatekeeper_classes, probs1):
        bar = "█" * int(prob.item() * 20)
        print(f"    - {cls:10s}: {prob.item()*100:5.2f}% {bar}")
        
    # ----------------- STAGE 2: SEVERITY (IF ECZEMA) -----------------
    if pred_class1.lower() == "eczema":
        print("\n" + "-" * 60)
        print("[!] Inflammatory lesion (Eczema) detected.")
        print("[Stage 2] Triggering Specialist Sub-type / Severity Assessor...")
        
        severity_model, severity_classes = load_model(severity_weights)
        with torch.no_grad():
            out2 = severity_model(input_tensor)
            probs2 = torch.softmax(out2, dim=1)[0]
            top3_probs, top3_indices = torch.topk(probs2, min(3, len(severity_classes)))
            
        top_class2 = severity_classes[top3_indices[0].item()]
        top_conf2 = top3_probs[0].item() * 100
        print(f"--> Stage 2 Sub-type: \033[1;36m{top_class2}\033[0m (Confidence: {top_conf2:.2f}%)")
        print("--> Top Diagnoses:")
        for idx, prob in zip(top3_indices, top3_probs):
            cls_name = severity_classes[idx.item()]
            print(f"    - {cls_name:30s}: {prob.item()*100:5.2f}%")
    else:
        print("\n" + "-" * 60)
        print(f"[*] Stage 1 diagnosis is '{pred_class1}'. No Eczema detected; Stage 2 skipped.")
        if pred_class1.lower() == "melanoma":
            print("\033[1;31m[CRITICAL ALERT] Potential Melanoma detected. Immediate dermatological biopsy advised.\033[0m")
            
    print("=" * 60)
    print("Diagnosis completed successfully.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascaded Dermatological Classifier CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to skin lesion image")
    parser.add_argument("--gatekeeper", type=str, default="gatekeeper_model.pth", help="Path to Stage 1 model")
    parser.add_argument("--severity", type=str, default="severity_model.pth", help="Path to Stage 2 model")
    args = parser.parse_args()
    
    run_cascaded_diagnosis(args.image, args.gatekeeper, args.severity)
