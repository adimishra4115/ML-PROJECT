#!/usr/bin/env python3
"""
=============================================================================
  Cascaded Dermatological Diagnosis & Severity Grading System
  run_all.py: Master pipeline script that rebuilds EDA, metrics, explainability, 
              comparison tables, and prints summary report.
=============================================================================
"""

import os
import sys

# Auto-activate local .venv if run with system python
try:
    import pandas
except ImportError:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python) and sys.executable != venv_python:
        os.execv(venv_python, [venv_python] + sys.argv)
    raise

import time
import json

from src.config import (
    BASE_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_PATH, 
    COMPARISON_PATH, GATEKEEPER_MODEL_PATH, SEVERITY_MODEL_PATH, DEVICE
)
from src.eda import run_eda
from src.evaluate import evaluate_models
from src.explain import generate_sample_gradcams

def main():
    start_time = time.time()
    print("=" * 75)
    print("  🏥 CASCADED DERMATOLOGICAL DIAGNOSIS: MASTER PIPELINE RUNNER")
    print("=" * 75)
    print(f"[*] Execution Directory : {BASE_DIR}")
    print(f"[*] Hardware Device      : {DEVICE}")
    print(f"[*] Stage 1 Checkpoint   : {os.path.basename(GATEKEEPER_MODEL_PATH)} ({'Found' if os.path.exists(GATEKEEPER_MODEL_PATH) else 'Missing'})")
    print(f"[*] Stage 2 Checkpoint   : {os.path.basename(SEVERITY_MODEL_PATH)} ({'Found' if os.path.exists(SEVERITY_MODEL_PATH) else 'Missing'})")
    print("=" * 75)

    # Step 1: Exploratory Data Analysis & Visualizations
    print("\n[Step 1/3] Generating Exploratory Data Analysis (EDA) Figures...")
    run_eda()

    # Step 2: Model Evaluation & Metric Generation
    print("\n[Step 2/3] Evaluating Models & Generating Confusion Matrix & Comparison Table...")
    evaluate_models()

    # Step 3: Explainable AI (Grad-CAM) Visualizations
    print("\n[Step 3/3] Generating Grad-CAM Explainability Attention Maps...")
    generate_sample_gradcams()

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"[✓] MASTER PIPELINE COMPLETED IN {elapsed:.1f} SECONDS!")
    print("=" * 75)

    # Print summary tables in terminal
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        s1 = m.get("stage1_gatekeeper", {})
        print("\n📈 STAGE 1 (GATEKEEPER) CLASSIFICATION METRICS:")
        print(f"   - Architecture           : {s1.get('model_architecture')}")
        print(f"   - Overall Accuracy       : {s1.get('overall_accuracy')*100:.2f}%")
        print(f"   - Macro Recall (Safety)  : {s1.get('macro_recall')*100:.2f}%")
        print(f"   - Macro Precision        : {s1.get('macro_precision')*100:.2f}%")
        print(f"   - Macro F1-Score         : {s1.get('macro_f1')*100:.2f}%")
        print(f"   - Inference Latency      : {s1.get('inference_latency_cpu_ms')} ms (CPU)")

    if os.path.exists(COMPARISON_PATH):
        print("\n📊 MODEL ARCHITECTURE BENCHMARK COMPARISON:")
        comp_df = pd.read_csv(COMPARISON_PATH)
        print(comp_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("  Artifacts Generated:")
    print(f"   ├── Figures   : {FIGURES_DIR}")
    print(f"   ├── Metrics   : {METRICS_PATH}")
    print(f"   └── Comparison: {COMPARISON_PATH}")
    print("=" * 75)
    print("\n👉 To launch the interactive Web Dashboard: python app.py")
    print("👉 To run terminal single-image prediction: python predict.py --image <path>\n")

if __name__ == "__main__":
    main()
