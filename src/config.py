import os
import torch

RANDOM_STATE = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
GATEKEEPER_MODEL_PATH = os.path.join(BASE_DIR, "gatekeeper_model.pth")
SEVERITY_MODEL_PATH = os.path.join(BASE_DIR, "severity_model.pth")

# Reports paths
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")

# Sample images directory
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")

# Class Names
STAGE1_CLASSES = ['Eczema', 'Melanoma', 'Psoriasis']
STAGE2_CLASSES = [
    'Atopic dermatitis childhood phase', 'Atopic dermatitis feet', 'Dyshidrosis', 
    'Eczema areola', 'Eczema asteatotic', 'Eczema chronic', 'Eczema fingertips', 
    'Eczema foot', 'Eczema hand', 'Ichthyosis', 'Keratolysis exfoliativa', 
    'Keratosis pilaris', 'Lichen simplex chronicus', 'Neurotic excoriations', 
    'Pompholyx', 'Prurigo nodularis', 'Stasis dermatitis'
]

# Clinical Metadata and Recommendation
CLINICAL_INFO = {
    "Eczema": {
        "type": "Inflammatory Dermatosis",
        "urgency": "Routine to Moderate",
        "action": "Triggers Stage 2 Severity Assessor. Prescribe topical emollients, topical corticosteroids, antihistamines.",
        "badge_color": "#f59e0b"
    },
    "Melanoma": {
        "type": "Malignant Skin Neoplasm",
        "urgency": "Critical / High Priority",
        "action": "Urgent dermatological biopsy and histological excision. Stage 2 bypassed.",
        "badge_color": "#ef4444"
    },
    "Psoriasis": {
        "type": "Chronic Autoimmune Epidermal Plaque",
        "urgency": "Specialist Consultation",
        "action": "Prescribe topical keratolytics, Vitamin D analogues, or systemic biologics.",
        "badge_color": "#0ea5e9"
    }
}
