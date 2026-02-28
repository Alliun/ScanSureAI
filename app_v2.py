"""
ScanSure AI v2 — app_v2.py
Streamlit UI with BraTS-trained model support.

Run:
    streamlit run app_v2.py

If no trained weights found, falls back to random weights with a warning.
"""

import io
import os
import sys
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# ── Allow imports from v1 folder ──────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from unet_model import LightUNet, mc_inference
from utils import preprocess_image, tensor_to_numpy, get_device, Timer


# ══════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ScanSure AI v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080d14;
    color: #c8d6e8;
  }
  section[data-testid="stSidebar"] {
    background: #0c1525;
    border-right: 1px solid #1d2f45;
  }
  .hero {
    padding: 2rem 0 1.4rem;
    border-bottom: 1px solid #1d3050;
    margin-bottom: 2rem;
  }
  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    color: #e0eeff;
    margin: 0;
    line-height: 1;
  }
  .hero h1 span { color: #3b9eff; }
  .hero .v2tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #080d14;
    background: #3b9eff;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 10px;
    vertical-align: middle;
    letter-spacing: 0.08em;
  }
  .hero p {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #5a7a9e;
    margin: 0.4rem 0 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .metric-card {
    background: #0c1a2e;
    border: 1px solid #1d3050;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #4a6a8e;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
  }
  .metric-card .value { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #3b9eff; }
  .metric-card .value.ok   { color: #30e88e; }
  .metric-card .value.warn { color: #f5a623; }
  .metric-card .value.bad  { color: #ff4b4b; }
  .panel-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #3b9eff;
    border-bottom: 1px solid #1d3050;
    padding-bottom: 6px;
    margin-bottom: 10px;
  }
  .finding-box {
    background: #0a1520;
    border: 1px solid #1d3050;
    border-left: 3px solid #3b9eff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.8;
    color: #8aaac8;
  }
  .finding-box .finding-title {
    color: #3b9eff;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
  }
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
  }
  .tag-trained  { background: #30e88e22; color: #30e88e; border: 1px solid #30e88e44; }
  .tag-random   { background: #f5a62322; color: #f5a623; border: 1px solid #f5a62344; }
  .tag-brats    { background: #3b9eff22; color: #3b9eff; border: 1px solid #3b9eff44; }
  div.stButton > button {
    background: #3b9eff; color: #080d14; border: none; border-radius: 6px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.05em; padding: 0.55rem 1.6rem; width: 100%;
    transition: background 0.2s, transform 0.1s;
  }
  div.stButton > button:hover  { background: #5cb3ff; transform: translateY(-1px); }
  div.stButton > button:active { transform: translateY(0); }
  hr { border-color: #1d3050; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Cached model loader — checks for trained weights
# ══════════════════════════════════════════════════════════════

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")

@st.cache_resource(show_spinner=False)
def load_model_and_device():
    device = get_device()
    model  = LightUNet(in_channels=1, dropout_p=0.2).to(device)

    trained = False
    ckpt_info = {}

    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            trained   = True
            ckpt_info = {
                "epoch":    ckpt.get("epoch", "?"),
                "val_loss": ckpt.get("val_loss", "?"),
                "val_dice": ckpt.get("val_dice", "?"),
            }
            print(f"[ScanSure AI v2] ✅  Loaded trained weights from {CHECKPOINT_PATH}")
            print(f"  Epoch: {ckpt_info['epoch']}  "
                  f"Val Loss: {ckpt_info['val_loss']:.4f}  "
                  f"Val Dice: {ckpt_info['val_dice']:.4f}")
        except Exception as e:
            print(f"[ScanSure AI v2] ⚠  Failed to load checkpoint: {e}")
    else:
        print(f"[ScanSure AI v2] ⚠  No checkpoint found at {CHECKPOINT_PATH} — using random weights")

    model.eval()
    return model, device, trained, ckpt_info


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def apply_colormap(arr: np.ndarray, cmap_name: str = "plasma") -> Image.Image:
    normed = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    rgba   = cm.get_cmap(cmap_name)(normed)
    rgb    = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def mask_to_image(mask_arr: np.ndarray, threshold: float = 0.5) -> Image.Image:
    normed = (mask_arr - mask_arr.min()) / (mask_arr.max() - mask_arr.min() + 1e-8)
    binary = (normed > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L").convert("RGB")


def generate_findings(lesion_coverage: float, mean_conf: float, is_trained: bool) -> str:
    """Generate a simple findings summary based on output metrics."""
    if not is_trained:
        return "⚠ Model not trained on BraTS data. Findings below are illustrative only."

    lines = []
    if lesion_coverage < 1.0:
        lines.append("• No significant lesion regions detected.")
    elif lesion_coverage < 5.0:
        lines.append(f"• Small region of interest detected ({lesion_coverage:.1f}% coverage).")
        lines.append("• Recommend clinical review of highlighted area.")
    elif lesion_coverage < 20.0:
        lines.append(f"• Moderate lesion region detected ({lesion_coverage:.1f}% coverage).")
        lines.append("• Possible edema or tumor mass — further imaging advised.")
    else:
        lines.append(f"• Large region of abnormality detected ({lesion_coverage:.1f}% coverage).")
        lines.append("• Significant pathology indicated — urgent clinical review recommended.")

    conf_str = f"{mean_conf*100:.1f}%"
    if mean_conf > 0.85:
        lines.append(f"• Model confidence: HIGH ({conf_str}) — prediction is reliable.")
    elif mean_conf > 0.70:
        lines.append(f"• Model confidence: MODERATE ({conf_str}) — results should be verified.")
    else:
        lines.append(f"• Model confidence: LOW ({conf_str}) — high uncertainty, do not rely on this prediction.")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️  Settings")
    st.divider()
    n_passes  = st.slider("MC Dropout Passes", 5, 30, 10, 1)
    threshold = st.slider("Mask Threshold", 0.1, 0.9, 0.5, 0.05)
    conf_cmap = st.selectbox("Confidence Colormap",
                             ["plasma", "magma", "inferno", "viridis", "hot"], index=0)
    st.divider()
    st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#3a5a78;line-height:1.8'>
MODEL v2<br>
LightUNet · BraTS 2023<br>
FLAIR modality · Binary mask<br>
BCE + Dice Loss · MC Dropout<br><br>
VERSION 0.2 · HACKATHON BUILD
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Main layout
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>Scan<span>Sure</span> AI <span class="v2tag">V2</span></h1>
  <p>BraTS-Trained Segmentation · Monte Carlo Confidence · Clinical Findings</p>
</div>
""", unsafe_allow_html=True)

model, device, is_trained, ckpt_info = load_model_and_device()

# ── Device + model status badges ─────────────────────────────
device_color  = "#30e88e" if device.type == "cuda" else "#f5a623"
device_label  = f"{'GPU · CUDA/ROCm' if device.type == 'cuda' else 'CPU'}"
trained_label = "BraTS Trained" if is_trained else "Random Weights"
trained_color = "#30e88e" if is_trained else "#f5a623"

col_b1, col_b2, col_b3 = st.columns([1, 1, 4])
with col_b1:
    st.markdown(f"""
<div style='font-family:Space Mono,monospace;font-size:0.72rem;color:{device_color};
            padding:6px 12px;background:#0c1a2e;border:1px solid {device_color}33;
            border-radius:6px;text-align:center'>◉ {device_label}</div>
""", unsafe_allow_html=True)
with col_b2:
    st.markdown(f"""
<div style='font-family:Space Mono,monospace;font-size:0.72rem;color:{trained_color};
            padding:6px 12px;background:#0c1a2e;border:1px solid {trained_color}33;
            border-radius:6px;text-align:center'>◉ {trained_label}</div>
""", unsafe_allow_html=True)

# ── Checkpoint info (if trained) ──────────────────────────────
if is_trained and ckpt_info:
    st.markdown(f"""
<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#4a6a8e;margin-bottom:1rem'>
  Checkpoint → Epoch {ckpt_info['epoch']} · 
  Val Loss: {ckpt_info['val_loss']:.4f} · 
  Val Dice: {ckpt_info['val_dice']:.4f}
</div>
""", unsafe_allow_html=True)
elif not is_trained:
    st.warning("⚠ No trained weights found. Run `train.py` with BraTS data first for meaningful results.")

# ── Upload ────────────────────────────────────────────────────
up_col, _ = st.columns([2, 3])
with up_col:
    st.markdown('<div class="panel-header">📂 Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload axial MRI slice (PNG/JPG/TIFF)",
                                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                                label_visibility="collapsed")
    run_btn = st.button("▶  Analyse Scan", disabled=(uploaded is None))

result_area = st.empty()

# ══════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════

if uploaded and run_btn:
    pil_image    = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    input_tensor = preprocess_image(pil_image)

    with st.spinner("Running Monte Carlo inference…"):
        with Timer() as t:
            mean_pred, uncertainty = mc_inference(
                model, input_tensor, n_passes=n_passes, device=device
            )

    mask_np = tensor_to_numpy(mean_pred)
    unc_np  = tensor_to_numpy(uncertainty)

    orig_disp = pil_image.resize((256, 256))
    mask_disp = mask_to_image(mask_np, threshold)
    conf_disp = apply_colormap(unc_np, conf_cmap)

    mean_conf       = float(1 - unc_np.mean())
    mask_normed     = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
    lesion_coverage = float((mask_normed > threshold).mean() * 100)

    with result_area.container():
        st.divider()

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="label">Inference Time</div>
            <div class="value">{t.elapsed_str}</div></div>""", unsafe_allow_html=True)
        with m2:
            cc = "ok" if mean_conf > 0.8 else ("warn" if mean_conf > 0.65 else "bad")
            st.markdown(f"""<div class="metric-card"><div class="label">Mean Confidence</div>
            <div class="value {cc}">{mean_conf*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with m3:
            lc = "ok" if lesion_coverage < 5 else ("warn" if lesion_coverage < 20 else "bad")
            st.markdown(f"""<div class="metric-card"><div class="label">Lesion Coverage</div>
            <div class="value {lc}">{lesion_coverage:.1f}%</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="label">MC Passes</div>
            <div class="value">{n_passes}</div></div>""", unsafe_allow_html=True)

        st.divider()

        # Images
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="panel-header">Original MRI</div>', unsafe_allow_html=True)
            st.image(orig_disp, use_container_width=True)
        with c2:
            st.markdown('<div class="panel-header">Segmentation Mask</div>', unsafe_allow_html=True)
            st.image(mask_disp, use_container_width=True)
        with c3:
            st.markdown('<div class="panel-header">Confidence Heatmap</div>', unsafe_allow_html=True)
            st.image(conf_disp, use_container_width=True)

        # Findings
        st.divider()
        st.markdown('<div class="panel-header">🩺 AI Findings Summary</div>', unsafe_allow_html=True)
        findings = generate_findings(lesion_coverage, mean_conf, is_trained)
        st.markdown(f"""
<div class="finding-box">
  <div class="finding-title">Automated Analysis Report</div>
  {findings.replace(chr(10), '<br>')}
  <br><br>
  <span style='color:#2a4a6e;font-size:0.62rem'>
  ⚠ This is an AI-assisted tool. Results must be reviewed by a qualified clinician.
  Not for diagnostic use without clinical validation.
  </span>
</div>
""", unsafe_allow_html=True)

        # Distribution chart
        st.divider()
        st.markdown('<div class="panel-header">Prediction Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 2.2), facecolor="#0c1a2e")
        ax.set_facecolor("#080d14")
        ax.hist(mask_np.ravel(), bins=80, color="#3b9eff", alpha=0.85, edgecolor="none")
        ax.axvline(threshold, color="#f5a623", linewidth=1.2, linestyle="--",
                   label=f"Threshold = {threshold}")
        ax.set_xlabel("Predicted Probability", color="#5a7a9e", fontfamily="monospace", fontsize=8)
        ax.set_ylabel("Pixel Count",           color="#5a7a9e", fontfamily="monospace", fontsize=8)
        ax.tick_params(colors="#3a5a78", labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor("#1d3050")
        ax.legend(fontsize=7, facecolor="#0c1a2e", edgecolor="#1d3050", labelcolor="#c8d6e8")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

elif not uploaded:
    with result_area.container():
        st.markdown("""
<div style='margin-top:3rem;text-align:center;padding:4rem 2rem;
            border:1px dashed #1d3050;border-radius:12px;'>
  <div style='font-size:3rem;margin-bottom:1rem'>🧠</div>
  <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:600;color:#2a4a6e'>
    Upload an MRI scan to begin analysis
  </div>
  <div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#3a5a78;margin-top:0.6rem'>
    Best results with axial FLAIR slices from BraTS-format data
  </div>
</div>
""", unsafe_allow_html=True)
