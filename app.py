import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import streamlit as st
import io

# ── Import standalone chatbot module ──
from chatbot import HydraBot, DEHYDRATION_SYMPTOMS

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="HydraVision XAI", page_icon="💧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0f1e; color: #e8eaf6; }
.main { background-color: #0a0f1e; }
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #0a2a4a 50%, #0d3b6e 100%);
    border: 1px solid #1e3a5f; border-radius: 20px; padding: 3rem 2.5rem;
    margin-bottom: 2rem; position: relative; overflow: hidden; text-align: center;
}
.hero::before { content: "💧"; position: absolute; font-size: 200px; opacity: 0.04; right: -20px; top: -30px; }
.hero h1 {
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(90deg, #4fc3f7, #81d4fa, #b3e5fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.5rem 0;
}
.hero p { color: #90caf9; font-family: 'DM Mono', monospace; font-size: 0.95rem; margin: 0; }
.option-card {
    background: linear-gradient(135deg, #0d1b3e, #0a2a4a);
    border: 1px solid #1e3a5f; border-radius: 16px; padding: 2.5rem 2rem;
    text-align: center;
}
.option-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.option-title { font-size: 1.5rem; font-weight: 800; color: #4fc3f7; margin-bottom: 0.5rem; }
.option-desc { font-family: 'DM Mono', monospace; font-size: 0.82rem; color: #546e8a; }
.chat-container {
    background: #080d1a; border: 1px solid #1e3a5f; border-radius: 16px;
    padding: 1.5rem; max-height: 500px; overflow-y: auto; margin-bottom: 1rem;
}
.chat-msg-user {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    border-radius: 12px 12px 2px 12px; padding: 0.8rem 1.2rem;
    margin: 0.5rem 0 0.5rem 20%; color: white; font-size: 0.9rem;
}
.chat-msg-bot {
    background: #0d1b3e; border: 1px solid #1e3a5f;
    border-radius: 12px 12px 12px 2px; padding: 0.8rem 1.2rem;
    margin: 0.5rem 20% 0.5rem 0; color: #e8eaf6; font-size: 0.9rem;
}
.chat-role { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #546e8a; margin-bottom: 0.3rem; }
.card-title { font-size: 0.75rem; font-family: 'DM Mono', monospace; color: #4fc3f7;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.6rem; }
.pred-hydrated { background: linear-gradient(135deg, #0d3b2e, #1b5e3a); border: 1px solid #2e7d52;
    border-radius: 10px; padding: 1.2rem 1.6rem; text-align: center; }
.pred-dehydrated { background: linear-gradient(135deg, #3b1a0d, #5e2a1b); border: 1px solid #7d3a2e;
    border-radius: 10px; padding: 1.2rem 1.6rem; text-align: center; }
.pred-label { font-size: 1.8rem; font-weight: 800; margin: 0; }
.pred-conf { font-family: 'DM Mono', monospace; font-size: 0.85rem; opacity: 0.8; margin-top: 0.2rem; }
.info-box { background: #0a1628; border-left: 3px solid #4fc3f7; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; font-family: 'DM Mono', monospace; font-size: 0.82rem; color: #90caf9; margin: 0.5rem 0; }
.metric-row { display: flex; gap: 1rem; margin-top: 0.8rem; }
.metric-box { flex: 1; background: #0a1628; border: 1px solid #1e3a5f; border-radius: 8px; padding: 0.8rem; text-align: center; }
.metric-val { font-size: 1.4rem; font-weight: 800; color: #4fc3f7; }
.metric-lbl { font-size: 0.7rem; font-family: 'DM Mono', monospace; color: #546e8a; text-transform: uppercase; letter-spacing: 0.1em; }
section[data-testid="stSidebar"] { background-color: #080d1a !important; border-right: 1px solid #1e3a5f; }
section[data-testid="stSidebar"] * { color: #b0c4de !important; }
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1); color: white !important;
    border: none; border-radius: 8px; font-family: 'Syne', sans-serif;
    font-weight: 600; padding: 0.55rem 1.4rem; width: 100%;
}
hr { border-color: #1e3a5f; }
.dehy-banner {
    background: linear-gradient(135deg, #3b1a0d, #5e2a1b);
    border: 1px solid #ef5350; border-radius: 16px;
    padding: 1.5rem 2rem; margin: 2rem 0; text-align: center;
}
.dehy-banner h2 { color: #ff8a65; font-size: 1.6rem; margin: 0 0 0.5rem 0; }
.dehy-banner p { color: #ffccbc; font-family: 'DM Mono', monospace; font-size: 0.88rem; margin: 0; }
.hydrated-banner {
    background: linear-gradient(135deg, #0d3b2e, #1b5e3a);
    border: 1px solid #69f0ae; border-radius: 16px;
    padding: 1.5rem 2rem; margin: 2rem 0; text-align: center;
}
.hydrated-banner h2 { color: #69f0ae; font-size: 1.6rem; margin: 0 0 0.5rem 0; }
.hydrated-banner p { color: #b9f6ca; font-family: 'DM Mono', monospace; font-size: 0.88rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ["Dehydrated", "Hydrated"]

# ──────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────
for key, default in [
    ("page", "home"),
    ("groq_key", ""),
    ("general_bot", None),
    ("general_msgs", []),
    ("post_bot", None),
    ("post_msgs", []),
    ("post_chat_started", False),
    ("analysis_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_face_detector():
    import cv2
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def crop_face_from_pil(pil_img, padding=0.25):
    import cv2
    detector = load_face_detector()
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    h_img, w_img = img_bgr.shape[:2]
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        pad_x = int(w * padding); pad_y = int(h * padding)
        x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
        x2 = min(w_img, x+w+pad_x); y2 = min(h_img, y+h+pad_y)
        cropped = Image.fromarray(img_rgb[y1:y2, x1:x2])
        return cropped, True, len(faces)
    else:
        side = min(h_img, w_img)
        y1 = (h_img-side)//2; x1 = (w_img-side)//2
        cropped = Image.fromarray(img_rgb[y1:y1+side, x1:x1+side])
        return cropped, False, 0

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return img_to_array(img) / 255.0

def make_predict_fn(model):
    def predict_fn(images):
        preds = model.predict(images, verbose=0)
        return np.hstack([1 - preds, preds])
    return predict_fn

def run_lime_red_zones(model, img_array, num_samples=300, num_features=5, is_hydrated=False):
    from lime import lime_image
    from skimage.segmentation import quickshift

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array, make_predict_fn(model),
        top_labels=2, hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: quickshift(x, kernel_size=3, max_dist=200, ratio=0.2)
    )

    bg = "#0a0f1e"

    if is_hydrated:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=bg)
        for ax in axes:
            ax.set_facecolor(bg); ax.axis("off")
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image", color="#90caf9", fontsize=12, pad=10)
        axes[1].imshow(img_array)
        axes[1].set_title("✅ Skin is Hydrated — No Dehydration Zones", color="#69f0ae", fontsize=12, pad=10)
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#69f0ae"); spine.set_linewidth(3); spine.set_visible(True)
    else:
        # ── Build a WEIGHTED heatmap from ALL LIME superpixel weights ──
        dehy_label = 0  # index 0 = Dehydrated

        # Get the segments map (same segmentation used internally by LIME)
        segments = quickshift(img_array, kernel_size=3, max_dist=200, ratio=0.2)

        # Extract per-segment weights for the dehydration label
        local_exp = explanation.local_exp.get(dehy_label, [])
        # local_exp is list of (segment_id, weight)
        seg_weights = dict(local_exp)

        # Build continuous weight map across all pixels
        weight_map = np.zeros(img_array.shape[:2], dtype=float)
        for seg_id, w in seg_weights.items():
            weight_map[segments == seg_id] = w

        # Keep only positive weights (regions that push TOWARD dehydration)
        pos_map = np.clip(weight_map, 0, None)

        # Normalize 0→1
        max_w = pos_map.max()
        if max_w > 0:
            pos_map = pos_map / max_w
        else:
            pos_map = np.zeros_like(pos_map)

        # ── Apply smooth heatmap: red tint scaled by weight ──
        # Deep red = strong dehydration signal, subtle orange = mild signal
        overlay = img_array.copy()
        # Red channel: boost; Green+Blue: suppress proportionally to weight
        strength = pos_map * 0.75  # max 75% overlay strength
        overlay[:, :, 0] = np.clip(img_array[:, :, 0] + strength * (1 - img_array[:, :, 0]), 0, 1)
        overlay[:, :, 1] = np.clip(img_array[:, :, 1] * (1 - strength * 0.8), 0, 1)
        overlay[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - strength * 0.8), 0, 1)
        colored = overlay

        # Build Reds colormap heatmap blended over original image
        import matplotlib.cm as cm
        cmap = cm.get_cmap("Reds")
        heat_rgba = cmap(pos_map)
        heat_rgb  = heat_rgba[:, :, :3]
        alpha_map = np.expand_dims(np.clip(pos_map * 0.85, 0, 0.85), axis=2)
        blended = img_array * (1 - alpha_map) + heat_rgb * alpha_map
        blended = np.clip(blended, 0, 1)

        fig = plt.figure(figsize=(18, 5.5), facecolor=bg)
        gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.05)
        ax0 = fig.add_subplot(gs[0]); ax0.set_facecolor(bg); ax0.axis("off")
        ax1 = fig.add_subplot(gs[1]); ax1.set_facecolor(bg); ax1.axis("off")
        ax_cb = fig.add_subplot(gs[2])
        ax0.imshow(img_array)
        ax0.set_title("Original Image", color="#90caf9", fontsize=12, pad=10)
        ax1.imshow(blended)
        ax1.set_title("Dehydration Heatmap  (deeper red = stronger signal)", color="#ff8a65", fontsize=11, pad=10)
        norm = plt.Normalize(vmin=0, vmax=1)
        sm   = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cb)
        cbar.set_label("Dehydration Signal Strength", color="#90caf9", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#90caf9", labelcolor="#90caf9")
        cbar.outline.set_edgecolor("#1e3a5f")
        ax_cb.set_facecolor(bg)

    plt.suptitle("HydraVision Skin Analysis Map", color="#4fc3f7", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return buf

def render_chat(messages):
    if not messages:
        return
    chat_html = '<div class="chat-container">'
    for msg in messages:
        content = msg["content"].replace("\n", "<br>")
        if msg["role"] == "user":
            chat_html += f'<div class="chat-msg-user"><div class="chat-role">You</div>{content}</div>'
        else:
            chat_html += f'<div class="chat-msg-bot"><div class="chat-role">💧 HydraBot</div>{content}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

def get_or_create_bot(key: str, mode: str) -> HydraBot:
    """Get existing bot from session state or create a fresh one."""
    if st.session_state.get(key) is None or not isinstance(st.session_state[key], HydraBot):
        st.session_state[key] = HydraBot(api_key=st.session_state.groq_key, mode=mode)
    else:
        # Update key in case user changed it in sidebar
        st.session_state[key].set_api_key(st.session_state.groq_key)
    return st.session_state[key]

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💧 HydraVision")
    st.markdown("---")
    if st.button("🏠 Home"):
        st.session_state.page = "home"
        st.rerun()
    if st.button("💬 Skin Chatbot"):
        st.session_state.page = "chatbot"
        st.session_state.chat_messages = []
        st.rerun()
    if st.button("🔬 Skin Analysis"):
        st.session_state.page = "analysis"
        st.session_state.post_chat_started = False
        st.session_state.post_analysis_chat = []
        st.rerun()
    st.markdown("---")

    st.markdown("### 🔑 Groq API Key")
    st.markdown('<a href="https://console.groq.com/keys" target="_blank" style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4fc3f7;">Get free key → console.groq.com</a>', unsafe_allow_html=True)
    groq_input = st.text_input("Paste key here", value=st.session_state.groq_key, type="password", label_visibility="collapsed", placeholder="gsk_...")
    if groq_input != st.session_state.groq_key:
        st.session_state.groq_key = groq_input
    if st.session_state.groq_key:
        st.markdown('<span style="color:#69f0ae;font-size:0.8rem;">✅ Key entered</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ff8a65;font-size:0.8rem;">⚠️ Key required for chatbot</span>', unsafe_allow_html=True)

    if st.session_state.page == "analysis":
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")
        model_path   = st.text_input("Model path (.h5)", value="hydration_model.h5")
        lime_samples = st.slider("LIME samples", 100, 800, 500, 100)
        lime_feats   = st.slider("Regions to highlight", 3, 10, 8)
    else:
        model_path = "hydration_model.h5"
        lime_samples = 500
        lime_feats = 8

# ══════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
      <h1>💧 HydraVision XAI</h1>
      <p>AI-Powered Skin Hydration Detection · MobileNetV2 · LIME Explainability · Personalized Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Choose Your Experience")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="option-card">
            <div class="option-icon">💬</div>
            <div class="option-title">Skin Chatbot</div>
            <div class="option-desc">Ask any question about skin health, hydration, skincare routines, and more. Our AI dermatology specialist answers only skin-related questions.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Skin Chatbot →", key="home_chat"):
            st.session_state.page = "chatbot"
            st.session_state.chat_messages = []
            st.rerun()

    with col2:
        st.markdown("""
        <div class="option-card">
            <div class="option-icon">🔬</div>
            <div class="option-title">Skin Analysis</div>
            <div class="option-desc">Upload a photo or use your camera. AI detects hydration levels and highlights dehydrated zones in red. Get personalized recommendations after.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Skin Analysis →", key="home_analysis"):
            st.session_state.page = "analysis"
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    💡 <b>How HydraVision works:</b> Our MobileNetV2 model analyzes facial skin to detect hydration status. 
    LIME explainability highlights exactly which regions show dehydration signals (shown in red). 
    After analysis, our AI chatbot guides you through personalized recommendations based on your specific symptoms.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: SKIN CHATBOT
# ══════════════════════════════════════════════
elif st.session_state.page == "chatbot":
    st.markdown("""
    <div class="hero" style="padding: 2rem;">
      <h1 style="font-size: 2rem;">💬 Skin Health Chatbot</h1>
      <p>Your personal AI dermatology assistant — skin questions only</p>
    </div>
    """, unsafe_allow_html=True)

    bot = get_or_create_bot("general_bot", "general")

    # Init greeting
    if not st.session_state.general_msgs:
        st.session_state.general_msgs = [
            {"role": "assistant", "content": bot.get_opening_message()}
        ]

    render_chat(st.session_state.general_msgs)

    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input("Ask about your skin...", key="main_chat_input",
                                   label_visibility="collapsed",
                                   placeholder="e.g. How do I know if my skin is dehydrated?",
                                   value=st.session_state.get("main_chat_input_val", ""))
    with col_btn:
        send = st.button("Send ›", key="main_chat_send")

    if send and user_input.strip():
        msg = user_input.strip()
        st.session_state["main_chat_input_val"] = ""
        with st.spinner(""):
            response = bot.chat(msg)
        st.session_state.general_msgs.append({"role": "user", "content": msg})
        st.session_state.general_msgs.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("🗑️ Clear Chat", key="clear_main_chat"):
        bot.reset()
        st.session_state.general_msgs = []
        st.rerun()

# ══════════════════════════════════════════════
#  PAGE: SKIN ANALYSIS
# ══════════════════════════════════════════════
elif st.session_state.page == "analysis":
    st.markdown("""
    <div class="hero" style="padding: 2rem;">
      <h1 style="font-size: 2rem;">🔬 Skin Hydration Analysis</h1>
      <p>Upload or capture a photo for AI-powered hydration detection</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model = None
    with st.spinner("Loading AI model..."):
        try:
            model = load_model(model_path)
            st.success(f"✅ Model loaded: `{model_path}`")
        except Exception as e:
            st.error(f"❌ Could not load model: {e}")
            st.info("Ensure `hydration_model.h5` is in the same folder as `app.py`")
            st.stop()

    st.markdown("---")

    col_up, col_info = st.columns([1, 1])
    with col_up:
        st.markdown('<div class="card-title">📤 Upload or Capture Image</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📁 Upload File", "📷 Camera"])
        uploaded = None
        with tab1:
            uploaded_file = st.file_uploader("Drop any JPG/PNG", type=["jpg","jpeg","png"], label_visibility="collapsed")
            if uploaded_file:
                uploaded = uploaded_file
        with tab2:
            camera_img = st.camera_input("Take a photo", label_visibility="collapsed")
            if camera_img:
                uploaded = camera_img

    with col_info:
        st.markdown('<div class="card-title">ℹ️ Analysis Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        1. 📸 Face auto-detected and cropped<br>
        2. 🤖 MobileNetV2 predicts hydration status<br>
        3. 🗺️ LIME highlights dehydrated zones in <b style="color:#ff8a65">red</b><br>
        4. 💬 AI chatbot gives personalized advice<br><br>
        <b>🔴 Red = dehydration signals detected</b><br>
        <b>✅ No red = skin is well hydrated</b>
        </div>
        """, unsafe_allow_html=True)

    if uploaded:
        pil_img_raw = Image.open(uploaded)
        face_cropped, face_found, n_faces = crop_face_from_pil(pil_img_raw, padding=0.25)

        st.markdown("---")
        if face_found:
            st.success(f"✅ Face detected ({n_faces} found) — analysing face region")
        else:
            st.warning("⚠️ No face detected — using center crop as fallback")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card-title">📷 Original Upload</div>', unsafe_allow_html=True)
            st.image(pil_img_raw, use_container_width=True)
        with c2:
            st.markdown('<div class="card-title">✂️ Face Region (used for analysis)</div>', unsafe_allow_html=True)
            st.image(face_cropped, use_container_width=True)

        pil_img   = face_cropped
        img_array = preprocess(pil_img)

        st.markdown("---")

        batch = np.expand_dims(img_array, 0)
        prob  = float(model.predict(batch, verbose=0)[0][0])
        pred_idx   = int(prob >= 0.5)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = prob if pred_idx == 1 else 1 - prob
        is_hydrated = pred_label == "Hydrated"

        # ── Always persist result so post-analysis section reads stable values ──
        st.session_state.analysis_result = {
            "prob": prob, "pred_label": pred_label,
            "confidence": confidence, "is_hydrated": is_hydrated
        }
        # Reset post-chat if a NEW image was just analyzed
        if st.session_state.get("last_pred_label") != pred_label:
            st.session_state.post_chat_started = False
            st.session_state.post_msgs = []
            st.session_state.post_bot  = None
        st.session_state["last_pred_label"] = pred_label

        r1, r2, r3 = st.columns([1, 1.2, 1])

        with r1:
            st.markdown('<div class="card-title">🖼 Analyzed Image</div>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

        with r2:
            st.markdown('<div class="card-title">🔬 Result</div>', unsafe_allow_html=True)
            cls   = "pred-hydrated" if is_hydrated else "pred-dehydrated"
            icon  = "💧" if is_hydrated else "⚠️"
            color = "#69f0ae" if is_hydrated else "#ff8a65"
            st.markdown(f"""
            <div class="{cls}">
                <div class="pred-label" style="color:{color}">{icon} {pred_label}</div>
                <div class="pred-conf">Confidence: {confidence:.1%}</div>
            </div>
            <div class="metric-row">
                <div class="metric-box"><div class="metric-val">{prob:.2f}</div><div class="metric-lbl">Raw Score</div></div>
                <div class="metric-box"><div class="metric-val">{confidence:.0%}</div><div class="metric-lbl">Certainty</div></div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="card-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
            fig_b, ax = plt.subplots(figsize=(3.5, 2.8), facecolor="#0a0f1e")
            ax.set_facecolor("#0a0f1e")
            bars = ax.barh(CLASS_NAMES, [1-prob, prob], color=["#ef5350","#42a5f5"], height=0.5)
            ax.set_xlim(0,1); ax.tick_params(colors="#90caf9", labelsize=9)
            for spine in ax.spines.values(): spine.set_edgecolor("#1e3a5f")
            for bar, val in zip(bars, [1-prob, prob]):
                ax.text(val+0.02, bar.get_y()+bar.get_height()/2, f"{val:.1%}", va="center", color="#e8eaf6", fontsize=9)
            plt.tight_layout()
            buf_b = io.BytesIO()
            plt.savefig(buf_b, format="png", dpi=120, bbox_inches="tight", facecolor="#0a0f1e")
            plt.close(); buf_b.seek(0)
            st.image(buf_b, use_container_width=True)

        # ── LIME MAP ──
        st.markdown("---")
        st.markdown("### 🗺️ Skin Analysis Map")
        if not is_hydrated:
            st.markdown("""
            <div class="info-box">
            🔴 <b>Red zones</b> show regions where the AI detected dehydration signals. 
            These are the areas of your skin showing characteristics associated with dehydration — like dullness, texture changes, or loss of plumpness.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            ✅ No significant dehydration zones detected. Your skin map shows healthy hydration throughout!
            </div>
            """, unsafe_allow_html=True)

        with st.spinner(f"Generating skin map ({lime_samples} samples)..."):
            try:
                buf_lime = run_lime_red_zones(model, img_array, num_samples=lime_samples, num_features=lime_feats, is_hydrated=is_hydrated)
                st.image(buf_lime, use_container_width=True)
                st.download_button("⬇️ Download Analysis Map", buf_lime, "skin_analysis_map.png", "image/png")
            except ImportError:
                st.error("LIME not installed. Run: `pip install lime` then restart.")
            except Exception as e:
                st.error(f"Map generation failed: {e}")

        # ══════════════════════════════════════════
        #  POST-ANALYSIS CHATBOT
        # ══════════════════════════════════════════
        st.markdown("---")

        # Always read from session_state so reruns don't flip the value
        result      = st.session_state.analysis_result
        is_hydrated = result["is_hydrated"]
        post_mode   = "hydrated" if is_hydrated else "dehydrated"

        if is_hydrated:
            st.markdown("""
            <div class="hydrated-banner">
                <h2>💧 Great News! Your Skin is Hydrated!</h2>
                <p>Your skin shows healthy hydration levels. Would you like personalized tips to maintain it?</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dehy-banner">
                <h2>⚠️ Your Skin Appears Dehydrated</h2>
                <p>Don't worry — our AI specialist will help you get it back to its best!</p>
            </div>
            """, unsafe_allow_html=True)

        # Start buttons
        if not st.session_state.post_chat_started:
            col_yes, col_no = st.columns(2)
            with col_yes:
                btn_label = "✅ Yes, give me recommendations!" if is_hydrated else "✅ Yes, help me fix it!"
                if st.button(btn_label, key="post_yes"):
                    bot  = HydraBot(api_key=st.session_state.groq_key, mode=post_mode)
                    st.session_state.post_bot  = bot
                    st.session_state.post_msgs = [
                        {"role": "assistant", "content": bot.get_opening_message()}
                    ]
                    st.session_state.post_chat_started = True
                    st.rerun()
            with col_no:
                if st.button("❌ Not now", key="post_no"):
                    if is_hydrated:
                        st.info("All good! Keep drinking water and maintaining your skincare routine! 💧")
                    else:
                        st.info("No problem! Drink water, use a good moisturizer, and consider seeing a dermatologist. 💧")

        # Chat interface
        if st.session_state.post_chat_started and st.session_state.post_msgs:
            # Update key in case it changed
            if isinstance(st.session_state.post_bot, HydraBot):
                st.session_state.post_bot.set_api_key(st.session_state.groq_key)

            st.markdown("### 💬 Personalized Skin Consultation")
            render_chat(st.session_state.post_msgs)

            # Symptom multiselect for dehydrated — first user turn only
            user_turns = [m for m in st.session_state.post_msgs if m["role"] == "user"]
            if not is_hydrated and len(user_turns) == 0:
                st.markdown("**Select all symptoms that apply to the red-highlighted areas:**")
                selected = st.multiselect("", DEHYDRATION_SYMPTOMS, key="symptom_multiselect")
                if st.button("Get My Personalized Plan →", key="submit_symptoms"):
                    if selected:
                        with st.spinner("Preparing your personalized recommendations..."):
                            resp = st.session_state.post_bot.chat_with_symptoms(selected)
                        st.session_state.post_msgs.append({"role": "user",      "content": "My skin feels: " + ", ".join(selected)})
                        st.session_state.post_msgs.append({"role": "assistant", "content": resp})
                        st.rerun()
                    else:
                        st.warning("Please select at least one symptom.")
            else:
                col_pi, col_pb = st.columns([5, 1])
                with col_pi:
                    post_input = st.text_input("Ask a follow-up...", key="post_input",
                                               label_visibility="collapsed",
                                               placeholder="e.g. What moisturizer should I use?",
                                               value=st.session_state.get("post_input_val", ""))
                with col_pb:
                    post_send = st.button("Send ›", key="post_send")

                if post_send and post_input.strip():
                    msg = post_input.strip()
                    st.session_state["post_input_val"] = ""
                    with st.spinner(""):
                        resp = st.session_state.post_bot.chat(msg)
                    st.session_state.post_msgs.append({"role": "user",      "content": msg})
                    st.session_state.post_msgs.append({"role": "assistant", "content": resp})
                    st.rerun()

            st.markdown("---")
            if st.button("🔄 Start New Analysis", key="reset_all"):
                st.session_state.post_chat_started = False
                st.session_state.post_msgs = []
                st.session_state.post_bot  = None
                st.session_state.analysis_result = None
                st.rerun()

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem; color: #546e8a;">
            <div style="font-size: 4rem;">📸</div>
            <div style="font-family: 'DM Mono', monospace; margin-top: 1rem; font-size: 1rem;">
                Upload an image or use your camera above to begin skin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)