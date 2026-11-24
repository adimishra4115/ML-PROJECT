import os
import base64
import subprocess
from pathlib import Path

def get_base64_img(path):
    if not os.path.exists(path):
        return ""
    ext = Path(path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

if __name__ == "__main__":
    imgs = {
        "__IMG_DIST__": get_base64_img("reports/figures/01_class_distribution.png"),
        "__IMG_BOX__": get_base64_img("reports/figures/02_boxplot_brightness_variance.png"),
        "__IMG_RGB__": get_base64_img("reports/figures/03_rgb_channel_histograms.png"),
        "__IMG_CM__": get_base64_img("reports/figures/04_confusion_matrix.png"),
        "__IMG_GRADCAM__": get_base64_img("reports/figures/05_gradcam_overlay.png"),
        "__IMG_ECZEMA__": get_base64_img("reports/figures/example_eczema.jpg"),
        "__IMG_MELANOMA__": get_base64_img("reports/figures/example_melanoma.jpg"),
        "__IMG_PSORIASIS__": get_base64_img("reports/figures/example_psoriasis.jpg"),
        "__IMG_PALM_APP__": get_base64_img("reports/figures/app_screenshot_palmprediction.jpg"),
        "__IMG_WEBCAM_APP__": get_base64_img("reports/figures/app_screenshot_webcam.jpg"),
    }

    template_path = "reports/full_report_template.html"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
        for k, v in imgs.items():
            html = html.replace(k, v)
        with open("reports/MSRIT_Dermatology_Report_Full.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("Updated reports/MSRIT_Dermatology_Report_Full.html")

    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome_bin):
        for name in ["MSRIT_Dermatology_Report_4Page", "MSRIT_Dermatology_Report_Full"]:
            html_file = os.path.abspath(f"reports/{name}.html")
            pdf_file = os.path.abspath(f"reports/{name}.pdf")
            cmd = [chrome_bin, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_file}", html_file]
            subprocess.run(cmd, check=True)
            print(f"Generated {pdf_file}")
