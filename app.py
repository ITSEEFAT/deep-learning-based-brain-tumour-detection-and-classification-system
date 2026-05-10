import os
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #060c1a !important;
    color: #c8d8f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1a3a6a;
    padding-bottom: 12px;
    margin-bottom: 24px;
}
.top-bar-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 22px;
    color: #4ab3ff;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.top-bar-status {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #2a6aaa;
    letter-spacing: 2px;
}

.panel {
    background: #080f20;
    border: 1px solid #1a3a6a;
    padding: 16px;
    margin-bottom: 16px;
}
.panel-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #2a6aaa;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid #1a3a6a;
    padding-bottom: 6px;
}

.status-detected {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #ff4444;
    letter-spacing: 2px;
}
.status-normal {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #44ff88;
    letter-spacing: 2px;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #0d1f3a;
    font-size: 13px;
}
.metric-label { color: #4a7ab0; font-size: 12px; }
.metric-value { color: #c8d8f0; font-family: 'Share Tech Mono', monospace; font-size: 12px; }

[data-testid="stFileUploader"] {
    background: #080f20 !important;
    border: 1px dashed #1a3a6a !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"] label {
    color: #4a7ab0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
}

[data-testid="stButton"] button {
    background: #0d2a5a !important;
    color: #4ab3ff !important;
    border: 1px solid #1a5aaa !important;
    border-radius: 0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 12px !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    background: #1a4a8a !important;
    border-color: #4ab3ff !important;
}

[data-testid="stImage"] img {
    border: 1px solid #1a3a6a;
    filter: brightness(0.9) contrast(1.1);
}

hr { border-color: #1a3a6a !important; }
[data-testid="stSpinner"] { color: #4ab3ff !important; }
[data-testid="stAlert"] {
    background: #080f20 !important;
    border: 1px solid #1a3a6a !important;
    border-radius: 0 !important;
    color: #4a7ab0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = {
    'glioma':     'GLIOMA TUMOR',
    'meningioma': 'MENINGIOMA',
    'notumor':    'NO TUMOR DETECTED',
    'pituitary':  'PITUITARY TUMOR',
}

# ── Model Architecture ────────────────────────────────────────
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256),         nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

EVAL_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ── Model loader — hard stop if missing ──────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists('model2.pth'):
        return None, "model2.pth not found — run train.py first"
    try:
        m = BrainTumorCNN(num_classes=4)
        m.load_state_dict(torch.load('model.pth', map_location='cpu', weights_only=True))
        m.eval()
        return m, None
    except Exception as e:
        return None, f"Failed to load model.pth: {e}"

model, model_error = load_model()

# ── Block entire app if model not available ───────────────────
if model_error:
    st.markdown(f"""
    <div style="border:1px solid #ff4444; background:#1a0000; padding:40px;
                font-family:'Share Tech Mono',monospace; text-align:center; margin-top:60px;">
        <div style="color:#ff4444; font-size:22px; letter-spacing:4px; margin-bottom:16px;">
            &#9888; MODEL NOT LOADED
        </div>
        <div style="color:#aa3333; font-size:13px; margin-bottom:20px;">
            {model_error}
        </div>
        <div style="color:#4a7ab0; font-size:11px; letter-spacing:2px; line-height:2;">
            Step 1 — Train the model:<br>
            <span style="color:#4ab3ff;">python train.py</span><br><br>
            Step 2 — Then launch the app:<br>
            <span style="color:#4ab3ff;">streamlit run app.py</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Model info ────────────────────────────────────────────────
num_params   = f"{sum(p.numel() for p in model.parameters()):,}"
device_label = 'CUDA' if torch.cuda.is_available() else 'CPU'

# ── Top bar ───────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="top-bar-title">&#11041; Brain Tumor Detection System</div>
    <div class="top-bar-status">CNN &middot; PyTorch &middot; &nbsp;</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 1.4, 1])

# ─── LEFT COLUMN ─────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel"><div class="panel-label">&#9658; MRI Upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        if 'last_uploaded' not in st.session_state or st.session_state['last_uploaded'] != uploaded.name:
            if 'result' in st.session_state:
                del st.session_state['result']
        st.session_state['last_uploaded'] = uploaded.name

        img = Image.open(uploaded).convert('RGB')
        st.markdown('<div class="panel"><div class="panel-label">&#9658; Input Image</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="panel">
            <div class="panel-label">&#9658; Image Info</div>
            <div class="metric-row"><span class="metric-label">FILENAME</span><span class="metric-value">{uploaded.name[:18]}</span></div>
            <div class="metric-row"><span class="metric-label">SIZE</span><span class="metric-value">{img.size[0]} x {img.size[1]} px</span></div>
            <div class="metric-row"><span class="metric-label">MODE</span><span class="metric-value">{img.mode}</span></div>
            <div class="metric-row"><span class="metric-label">INPUT TENSOR</span><span class="metric-value">128 x 128 x 3</span></div>
        </div>
        """, unsafe_allow_html=True)

# ─── MIDDLE COLUMN ───────────────────────────────────────────
with col_mid:
    if uploaded:
        run = st.button("RUN ANALYSIS", use_container_width=True)

        if run:
            with st.spinner("PROCESSING..."):
                time.sleep(1.0)

                tensor = EVAL_TF(img).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs   = F.softmax(outputs, dim=1)[0]

                pred_idx   = probs.argmax().item()
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx].item()
                probs_dict = {c: probs[i].item() for i, c in enumerate(CLASS_NAMES)}

                st.session_state['result'] = {
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'probs':      probs_dict
                }

        if 'result' in st.session_state:
            r        = st.session_state['result']
            pred     = r['pred_class']
            conf     = r['confidence']
            is_tumor = pred != 'notumor'

            status_class = "status-detected" if is_tumor else "status-normal"
            status_text  = f"DETECTED : {CLASS_LABELS[pred]}" if is_tumor else "STATUS : NO TUMOR DETECTED"

            st.markdown(f"""
            <div class="panel">
                <div class="panel-label">&#9658; Prediction Result</div>
                <div class="{status_class}" style="font-size:16px; margin:8px 0;">{status_text}</div>
                <div class="metric-row" style="margin-top:10px">
                    <span class="metric-label">CONFIDENCE</span>
                    <span class="metric-value" style="color:#4ab3ff; font-size:18px;">{conf*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence bars via components.html (avoids Streamlit markdown parser) ──
            bars_html = """
<style>
  body { margin:0; padding:0; background:#080f20; }
  .panel { background:#080f20; border:1px solid #1a3a6a; padding:16px; }
  .panel-label { font-family:'Share Tech Mono',monospace; font-size:10px; color:#2a6aaa;
                 letter-spacing:3px; text-transform:uppercase;
                 border-bottom:1px solid #1a3a6a; padding-bottom:6px; margin-bottom:12px; }
  .conf-row { margin:8px 0; }
  .conf-label { display:flex; justify-content:space-between;
                font-family:'Share Tech Mono',monospace; font-size:11px;
                color:#4a7ab0; margin-bottom:4px; }
  .conf-label .top { color:#c8d8f0; }
  .conf-bar-bg { background:#0d1f3a; height:6px; width:100%; }
  .conf-bar-fill { height:6px; background:linear-gradient(90deg,#1a5aaa,#4ab3ff); }
  .conf-bar-fill.high { background:linear-gradient(90deg,#aa1a1a,#ff4444); }
</style>
<div class="panel">
  <div class="panel-label">&#9658; Class Probabilities</div>
"""
            for cls in CLASS_NAMES:
                p         = r['probs'][cls]
                is_top    = cls == pred
                fill      = "high" if (is_top and is_tumor) else ""
                prefix    = "&#9658; " if is_top else ""
                cls_class = "top" if is_top else ""
                bars_html += f"""
  <div class="conf-row">
    <div class="conf-label">
      <span class="{cls_class}">{prefix}{cls.upper()}</span>
      <span class="{cls_class}">{p*100:.1f}%</span>
    </div>
    <div class="conf-bar-bg">
      <div class="conf-bar-fill {fill}" style="width:{p*100:.1f}%"></div>
    </div>
  </div>"""

            bars_html += "\n</div>"
            components.html(bars_html, height=len(CLASS_NAMES) * 52 + 60, scrolling=False)

            # ── Bar chart ──
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor('#080f20')
            ax.set_facecolor('#060c1a')
            colors = ['#ff4444' if c == pred and is_tumor else '#4ab3ff' for c in CLASS_NAMES]
            vals   = [r['probs'][c] * 100 for c in CLASS_NAMES]
            bars   = ax.bar(CLASS_NAMES, vals, color=colors, width=0.5,
                            edgecolor='#1a3a6a', linewidth=0.8)
            ax.set_ylim(0, 105)
            ax.tick_params(colors='#4a7ab0', labelsize=8)
            ax.spines[:].set_color('#1a3a6a')
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.set_ylabel('Confidence %', color='#4a7ab0', fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f'{val:.1f}', ha='center', va='bottom',
                        color='#c8d8f0', fontsize=7, fontfamily='monospace')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        else:
            st.markdown("""
            <div class="panel" style="text-align:center; padding:40px 16px;">
                <div style="font-family:'Share Tech Mono',monospace; color:#1a3a6a;
                            font-size:12px; letter-spacing:3px;">
                    AWAITING ANALYSIS
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel" style="text-align:center; padding:60px 16px;">
            <div style="font-family:'Share Tech Mono',monospace; color:#1a3a6a;
                        font-size:11px; letter-spacing:3px; line-height:2;">
                NO IMAGE LOADED<br>
                UPLOAD MRI SCAN TO BEGIN
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── RIGHT COLUMN ─────────────────────────────────────────────
with col_right:
    st.markdown(f"""
    <div class="panel">
        <div class="panel-label">&#9658; System Status</div>
        <div class="metric-row"><span class="metric-label">MODEL</span>
            <span class="metric-value" style="color:#44ff88;">LOADED &#10003;</span></div>
        <div class="metric-row"><span class="metric-label">DEVICE</span>
            <span class="metric-value">{device_label}</span></div>
        <div class="metric-row"><span class="metric-label">PARAMETERS</span>
            <span class="metric-value">{num_params}</span></div>
        <div class="metric-row"><span class="metric-label">INPUT SIZE</span>
            <span class="metric-value">128 x 128</span></div>
        <div class="metric-row"><span class="metric-label">CLASSES</span>
            <span class="metric-value">4</span></div>
        <div class="metric-row"><span class="metric-label">FRAMEWORK</span>
            <span class="metric-value">PyTorch</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-label">&#9658; Target Classes</div>
        <div class="metric-row"><span class="metric-label">01</span><span class="metric-value">GLIOMA</span></div>
        <div class="metric-row"><span class="metric-label">02</span><span class="metric-value">MENINGIOMA</span></div>
        <div class="metric-row"><span class="metric-label">03</span><span class="metric-value">NO TUMOR</span></div>
        <div class="metric-row"><span class="metric-label">04</span><span class="metric-value">PITUITARY</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-label">&#9658; Preprocessing</div>
        <div class="metric-row"><span class="metric-label">RESIZE</span><span class="metric-value">128x128</span></div>
        <div class="metric-row"><span class="metric-label">NORMALIZE</span><span class="metric-value">u=0.5 o=0.5</span></div>
        <div class="metric-row"><span class="metric-label">FORMAT</span><span class="metric-value">RGB TENSOR</span></div>
    </div>
    """, unsafe_allow_html=True)

    if 'result' in st.session_state and uploaded:
        r    = st.session_state['result']
        pred = r['pred_class']
        conf = r['confidence']

        risk_map = {
            'glioma':     ('HIGH',     '#ff4444'),
            'meningioma': ('MODERATE', '#ffaa44'),
            'pituitary':  ('MODERATE', '#ffaa44'),
            'notumor':    ('NONE',     '#44ff88'),
        }
        risk_label, risk_color = risk_map[pred]

        st.markdown(f"""
        <div class="panel">
            <div class="panel-label">&#9658; Assessment</div>
            <div class="metric-row">
                <span class="metric-label">RISK LEVEL</span>
                <span class="metric-value" style="color:{risk_color};">{risk_label}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">CONFIDENCE</span>
                <span class="metric-value">{conf*100:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">DIAGNOSIS</span>
                <span class="metric-value">{CLASS_LABELS[pred]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)