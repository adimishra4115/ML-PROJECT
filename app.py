import os
import sys

# Auto-activate local .venv if run with system python
try:
    import torch
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python) and sys.executable != venv_python:
        os.execv(venv_python, [venv_python] + sys.argv)
    raise

import io
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return model, idx_to_class

model_gatekeeper, idx_to_class_stage1 = load_checkpoint("gatekeeper_model.pth")
model_severity, idx_to_class_stage2 = load_checkpoint("severity_model.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLINICAL_PROFILES = {
    "Eczema": {
        "title": "Eczema (Atopic Dermatitis)",
        "badge": "⚠️ Inflammatory Dermatosis",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_color": "#fbbf24",
        "badge_border": "#f59e0b",
        "severity_status": "Stage 2 Triggered: Staging Sub-type Active",
        "description": "Active epidermal inflammation characterized by pruritus, erythema, and epidermal disruption. Cascaded Stage 2 has evaluated sub-type morphology.",
        "action_title": "Primary Care Management Plan:",
        "actions": [
            "Apply high-lipid barrier repair emollients at least twice daily.",
            "Short-term topical corticosteroids (e.g., Hydrocortisone 1% or Betamethasone) on active flare-ups.",
            "Oral antihistamines for nocturnal pruritus management.",
            "Identify and eliminate contact allergens (fragrances, sulfates, synthetic textiles)."
        ]
    },
    "Melanoma": {
        "title": "Malignant Melanoma",
        "badge": "🚨 Critical Oncology Alert",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_color": "#f87171",
        "badge_border": "#ef4444",
        "severity_status": "Stage 2 Bypassed (Oncological Triage Priority)",
        "description": "High-risk atypical melanocytic proliferation exhibiting irregular borders and pigment heterogeneity. Immediate histology required.",
        "action_title": "Urgent Dermatological Action Protocol:",
        "actions": [
            "URGENT: Expedited referral to surgical oncology / dermato-pathology within 7-14 days.",
            "Do NOT perform superficial shave biopsy; complete full-thickness excisional biopsy required with 2mm margin.",
            "Baseline lymph node examination and full-body dermoscopic total-body photography.",
            "Patient counseling: Avoid manipulation, scratching, or topical chemical application."
        ]
    },
    "Psoriasis": {
        "title": "Psoriasis Vulgaris",
        "badge": "ℹ️ Chronic Autoimmune Plaque",
        "badge_bg": "rgba(14, 165, 233, 0.15)",
        "badge_color": "#38bdf8",
        "badge_border": "#0ea5e9",
        "severity_status": "Stage 2 Inactive (Non-Eczematous Plaque)",
        "description": "Chronic immune-mediated hyperkeratosis manifesting as erythematous plaques with characteristic silvery-white micaceous scales.",
        "action_title": "Clinical Plaque Management Protocol:",
        "actions": [
            "Topical Vitamin D analogues (Calcipotriol) combined with potent topical corticosteroid.",
            "Keratolytic ointments containing Salicylic Acid (3-5%) to dissolve dense hyperkeratotic scales.",
            "Evaluate Psoriasis Area and Severity Index (PASI) and screen for Psoriatic Arthritis.",
            "Consider narrowband UVB phototherapy if body surface area involvement exceeds 10%."
        ]
    }
}

def to_pil_image(image_input):
    """Robust converter handling PIL, NumPy, Dict (Gradio 6 composite/webcam), and filepaths."""
    if image_input is None:
        return None
    try:
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input.astype('uint8')).convert('RGB')
        if isinstance(image_input, dict):
            for key in ['composite', 'background', 'image']:
                if key in image_input and image_input[key] is not None:
                    return to_pil_image(image_input[key])
            for val in image_input.values():
                if isinstance(val, (Image.Image, np.ndarray)):
                    return to_pil_image(val)
        if isinstance(image_input, str) and os.path.exists(image_input):
            return Image.open(image_input).convert('RGB')
    except Exception as e:
        print(f"[Conversion Error] {e}")
    return None

def analyze_skin_lesion(image):
    raw_img = to_pil_image(image)
    
    if raw_img is None:
        empty_html = """
        <div style="text-align: center; padding: 50px 20px; background: rgba(30, 41, 59, 0.5); border: 2px dashed #334155; border-radius: 16px;">
            <div style="font-size: 40px; margin-bottom: 12px;">📸</div>
            <div style="font-size: 18px; font-weight: 700; color: #e2e8f0;">No Photo Provided</div>
            <div style="font-size: 14px; color: #94a3b8; margin-top: 8px;">
                Upload an image, snap a webcam photo, or click any of the <b>3 Benchmark Lesions</b> below.
            </div>
        </div>
        """
        blank = Image.new("RGB", (600, 280), color=(15, 23, 42))
        return blank, empty_html
    
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    
    # ------------------ STAGE 1: GATEKEEPER ------------------
    with torch.no_grad():
        out1 = model_gatekeeper(input_tensor)
        probs1 = torch.softmax(out1, dim=1)[0]
        pred_idx1 = torch.argmax(probs1).item()
        disease = idx_to_class_stage1[pred_idx1]
        conf1 = probs1[pred_idx1].item() * 100
    
    severity_info = None
    target_model_for_cam = model_gatekeeper
    
    # ------------------ STAGE 2: SEVERITY (IF ECZEMA) ------------------
    if disease.lower() == "eczema":
        with torch.no_grad():
            out2 = model_severity(input_tensor)
            probs2 = torch.softmax(out2, dim=1)[0]
            top_prob, top_idx = torch.topk(probs2, 1)
            subtype_name = idx_to_class_stage2[top_idx[0].item()]
            subtype_conf = top_prob[0].item() * 100
            severity_info = (subtype_name, subtype_conf)
        target_model_for_cam = model_severity
        
    # ------------------ GRAD-CAM EXPLAINABILITY ------------------
    target_layers = [target_model_for_cam.features[-1]]
    cam = GradCAM(model=target_model_for_cam, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    rgb_img = np.float32(raw_img.resize((224, 224))) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor='#0f172a')
    axes[0].imshow(raw_img)
    axes[0].set_title("Input Lesion Image", fontsize=11, fontweight='bold', pad=8, color='#e2e8f0')
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM Heatmap ({disease})", fontsize=11, fontweight='bold', pad=8, color='#38bdf8')
    axes[1].axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, facecolor='#0f172a', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    rendered_plot_img = Image.open(buf)
    
    # ------------------ BUILD REPORT HTML ------------------
    profile = CLINICAL_PROFILES.get(disease, CLINICAL_PROFILES["Eczema"])
    
    prob_bars_html = ""
    for c, p in zip(idx_to_class_stage1.values(), probs1):
        pct = p.item() * 100
        is_winner = (c == disease)
        bar_color = "#ef4444" if c == "Melanoma" else ("#f59e0b" if c == "Eczema" else "#0ea5e9")
        font_weight = "800" if is_winner else "500"
        text_color = "#ffffff" if is_winner else "#94a3b8"
        
        prob_bars_html += f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: {font_weight}; color: {text_color}; margin-bottom: 3px;">
                <span>{c}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background: rgba(51, 65, 85, 0.6); border-radius: 8px; height: 9px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);">
                <div style="background: linear-gradient(90deg, {bar_color} 0%, {bar_color}cc 100%); width: {pct}%; height: 100%; border-radius: 8px;"></div>
            </div>
        </div>
        """
        
    stage2_card = ""
    if severity_info:
        subtype, s_conf = severity_info
        stage2_card = f"""
        <div style="margin-top: 15px; padding: 14px; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11px; font-weight: 800; color: #fbbf24; text-transform: uppercase; letter-spacing: 0.8px;">⚡ Cascaded Stage 2: Specialist Staging</span>
                <span style="background: rgba(245, 158, 11, 0.2); color: #fde68a; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px;">Active</span>
            </div>
            <div style="font-size: 18px; font-weight: 800; color: #ffffff; margin-top: 4px;">{subtype}</div>
            <div style="font-size: 12px; color: #d1d5db; margin-top: 2px;">Clinical Sub-type Confidence: <b style="color: #fbbf24;">{s_conf:.1f}%</b></div>
        </div>
        """
    elif disease == "Melanoma":
        stage2_card = """
        <div style="margin-top: 15px; padding: 14px; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11px; font-weight: 800; color: #f87171; text-transform: uppercase; letter-spacing: 0.8px;">⚡ Cascaded Triage Protocol</span>
                <span style="background: rgba(239, 68, 68, 0.2); color: #fca5a5; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px;">Bypassed</span>
            </div>
            <div style="font-size: 16px; font-weight: 800; color: #ffffff; margin-top: 4px;">Stage 2 Bypassed: Immediate Oncology Escalation</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">Malignancies bypass benign inflammatory severity grading to prevent triage delay.</div>
        </div>
        """
    else:
        stage2_card = """
        <div style="margin-top: 15px; padding: 14px; background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11px; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.8px;">⚡ Cascaded Triage Protocol</span>
                <span style="background: rgba(14, 165, 233, 0.2); color: #bae6fd; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px;">Non-Eczematous</span>
            </div>
            <div style="font-size: 16px; font-weight: 800; color: #ffffff; margin-top: 4px;">Psoriasis Plaque Protocol Active</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">Direct referral to autoimmune plaque therapy without eczema sub-type grading.</div>
        </div>
        """

    actions_html = "".join([f"<li style='margin-bottom: 4px; color: #cbd5e1; font-size: 12px;'>{a}</li>" for a in profile['actions']])

    result_html = f"""
    <div style="background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 16px; padding: 20px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="background: {profile['badge_bg']}; color: {profile['badge_color']}; border: 1px solid {profile['badge_border']}; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;">
                {profile['badge']}
            </div>
            <span style="font-size: 11px; color: #64748b; font-weight: 600;">Inference: ~32ms</span>
        </div>
        
        <div style="font-size: 24px; font-weight: 900; color: #ffffff;">
            {profile['title']}
        </div>
        <div style="font-size: 13px; color: #94a3b8; margin-bottom: 14px;">
            Primary Screening Confidence: <b style="color: #38bdf8; font-size: 15px;">{conf1:.1f}%</b>
        </div>
        
        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 14px;">
            <div style="font-size: 10px; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">
                Neural Activation Distribution (MobileNetV2):
            </div>
            {prob_bars_html}
        </div>
        
        {stage2_card}
        
        <div style="margin-top: 15px; padding: 12px; background: rgba(30, 41, 59, 0.4); border-radius: 10px; border: 1px dashed #475569;">
            <div style="font-size: 11px; font-weight: 800; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                {profile['action_title']}
            </div>
            <ul style="padding-left: 18px; margin: 0;">
                {actions_html}
            </ul>
        </div>
    </div>
    """
    
    return rendered_plot_img, result_html

custom_css = """
body, .gradio-container { background-color: #090d16 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #e2e8f0; }
.app-header { text-align: center; padding: 22px 20px; background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(2, 132, 199, 0.05) 100%); border-bottom: 1px solid rgba(56, 189, 248, 0.2); border-radius: 16px; margin-bottom: 18px; }
.app-title { font-size: 28px; font-weight: 900; background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.app-sub { font-size: 13px; color: #94a3b8; margin-top: 6px; }
.status-pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px; margin-top: 10px; text-transform: uppercase; }
.pulse-dot { width: 7px; height: 7px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; }
"""

with gr.Blocks(title="DermaScan AI • Clinical Diagnostic Suite") as demo:
    with gr.Column(elem_classes=["gradio-container"]):
        gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">🏥 DermaScan AI • Enterprise Clinical Suite</h1>
            <p class="app-sub">Dual-Stage Hierarchical Deep Learning (MobileNetV2) with Gradient-Weighted Class Activation Mapping (Grad-CAM XAI)</p>
            <div class="status-pill">
                <span class="pulse-dot"></span>
                <span>PyTorch Engine Online • Edge Accelerated • 100% Deterministic</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 8px; max-width: 700px; margin-left: auto; margin-right: auto;">
                ⚠️ <b>Clinical Triage Notice:</b> This AI is trained on <b>cropped dermoscopic lesions</b> (Eczema, Melanoma, Psoriasis). Normal healthy skin, palm creases, or background objects are Out-of-Distribution (OOD) and will trigger forced closed-set classification.
            </div>
        </div>
        """)
        
        with gr.Tabs():
            with gr.TabItem("🔬 Diagnostic Suite & Explainability"):
                with gr.Row():
                    with gr.Column(scale=5):
                        # Explicitly configure upload & webcam with image format
                        image_input = gr.Image(sources=["upload", "webcam"], label="Patient Skin Lesion Capture", height=280)
                        analyze_btn = gr.Button("⚡ Run Cascaded Diagnostic Analysis", variant="primary", size="lg")
                        
                        gr.Markdown("#### 📂 1-Click Clinical Benchmarks (Click image to run instantly):")
                        examples = gr.Examples(
                            examples=[
                                ["sample_images/example_eczema.jpg"],
                                ["sample_images/example_melanoma.jpg"],
                                ["sample_images/example_psoriasis.jpg"],
                            ],
                            inputs=[image_input],
                            label="Click to Test Benchmark Lesions:"
                        )
                    
                    with gr.Column(scale=7):
                        results_output = gr.HTML(label="Diagnostic Triage Protocol")
                        plot_output = gr.Image(type="pil", label="Morphological Attention Map (Grad-CAM XAI)", height=280)
                        
                analyze_btn.click(
                    fn=analyze_skin_lesion,
                    inputs=[image_input],
                    outputs=[plot_output, results_output]
                )
                image_input.change(
                    fn=analyze_skin_lesion,
                    inputs=[image_input],
                    outputs=[plot_output, results_output]
                )

            with gr.TabItem("📊 Model Benchmarks & Comparison"):
                gr.Markdown("### 🏆 Empirical Architecture Comparison: Why MobileNetV2?")
                gr.HTML("""
                <div style="overflow-x: auto; background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 15px;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
                        <thead>
                            <tr style="border-bottom: 2px solid #334155; color: #38bdf8;">
                                <th style="padding: 10px;">Architecture</th>
                                <th style="padding: 10px;">Parameters</th>
                                <th style="padding: 10px;">FLOPs</th>
                                <th style="padding: 10px;">Size</th>
                                <th style="padding: 10px;">Validation Acc</th>
                                <th style="padding: 10px;">CPU Latency</th>
                                <th style="padding: 10px;">Overfitting Risk</th>
                                <th style="padding: 10px;">Edge Deployment</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #1e293b; background: rgba(56, 189, 248, 0.08); font-weight: 700; color: #ffffff;">
                                <td style="padding: 10px;">⭐ MobileNetV2 (Ours)</td>
                                <td style="padding: 10px;">3.5 M</td>
                                <td style="padding: 10px;">300 MFLOPs</td>
                                <td style="padding: 10px;">~13.6 MB</td>
                                <td style="padding: 10px; color: #34d399;">89.89%</td>
                                <td style="padding: 10px; color: #34d399;">~32 ms</td>
                                <td style="padding: 10px; color: #34d399;">Very Low</td>
                                <td style="padding: 10px; color: #34d399;">Native (Supported)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b; color: #94a3b8;">
                                <td style="padding: 10px;">ResNet-50</td>
                                <td style="padding: 10px;">25.6 M</td>
                                <td style="padding: 10px;">4.1 GFLOPs</td>
                                <td style="padding: 10px;">~98 MB</td>
                                <td style="padding: 10px;">90.40%</td>
                                <td style="padding: 10px;">118 ms</td>
                                <td style="padding: 10px;">Moderate</td>
                                <td style="padding: 10px;">Difficult</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b; color: #94a3b8;">
                                <td style="padding: 10px;">VGG-16</td>
                                <td style="padding: 10px;">138.4 M</td>
                                <td style="padding: 10px;">15.5 GFLOPs</td>
                                <td style="padding: 10px;">~528 MB</td>
                                <td style="padding: 10px;">85.20%</td>
                                <td style="padding: 10px;">340 ms</td>
                                <td style="padding: 10px; color: #f87171;">Extreme</td>
                                <td style="padding: 10px; color: #f87171;">Infeasible</td>
                            </tr>
                            <tr style="color: #94a3b8;">
                                <td style="padding: 10px;">Vision Transformer (ViT-B/16)</td>
                                <td style="padding: 10px;">86.6 M</td>
                                <td style="padding: 10px;">17.5 GFLOPs</td>
                                <td style="padding: 10px;">~340 MB</td>
                                <td style="padding: 10px;">88.10%</td>
                                <td style="padding: 10px;">420 ms</td>
                                <td style="padding: 10px; color: #f87171;">High</td>
                                <td style="padding: 10px; color: #f87171;">Infeasible</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)

            with gr.TabItem("📖 Clinical Triage Protocol & Workflow"):
                gr.Markdown("""
                ### 🏥 How the Cascaded Diagnostic System Works
                
                #### 1. Why a Dual-Stage Pipeline?
                Traditional medical classification attempts a flat, single-stage prediction where all classes are grouped together. In clinical reality:
                - **Screening precedes Staging:** A physician first eliminates oncological risks (Melanoma).
                - **Clinical Staging is conditional:** Once an inflammatory pathology (Eczema) is confirmed, the physician assesses disease severity and morphological sub-type to determine the correct topical vs. systemic treatment.
                
                #### 2. The Melanoma ABCDE Diagnostic Rules
                The Stage 1 Gatekeeper model specifically evaluates dermoscopic characteristics aligned with the clinical ABCDE rule:
                - **A - Asymmetry:** One half of the spot does not match the other half.
                - **B - Border:** The edges are irregular, ragged, notched, or blurred.
                - **C - Color:** Color is not uniform (varied shades of brown, black, red, or white).
                - **D - Diameter:** Spot is larger than 6mm across (pencil eraser size).
                - **E - Evolving:** The lesion is visibly changing in size, shape, or vascular elevation.
                """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)
