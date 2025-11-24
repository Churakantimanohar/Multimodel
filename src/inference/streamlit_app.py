import streamlit as st
import torch, os, sys, time, json, numpy as np, pandas as pd, altair as alt
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
from PIL import Image
from tempfile import NamedTemporaryFile

# Ensure root path added before importing internal src modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.expression_utils import detect_expression, overlay_expression
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

# Page config MUST be first Streamlit call
st.set_page_config(page_title="Multimodal Mental Health Detection", layout="wide", page_icon="🧠")

# (Path setup moved above to precede internal imports)

from src.utils.config import get_config
from src.models.text_model import TextEncoder
from src.models.audio_model import AudioEncoder
from src.models.video_model import VideoEncoder
from src.models.fusion.attention_fusion import MultiModalAttentionFusion
from src.models.classifier import FusionClassifier
from src.inference.checkpoint_utils import load_latest
from src.preprocessing.text_preprocess import preprocess_text_batch
from src.preprocessing.audio_preprocess import extract_audio_features
from src.preprocessing.video_preprocess import process_video

cfg = get_config()
LABELS = ["Normal","Anxiety","Stress","Depression"]

@st.cache_resource
def load_models():
    text_enc = TextEncoder().to(cfg.device)
    audio_enc = AudioEncoder().to(cfg.device)
    video_enc = VideoEncoder().to(cfg.device)
    fusion = MultiModalAttentionFusion(dim_text=256, dim_audio=256, dim_video=256, fusion_dim=cfg.fusion_hidden_dim).to(cfg.device)
    classifier = FusionClassifier(seq_dim=cfg.fusion_hidden_dim, num_classes=cfg.num_classes).to(cfg.device)
    ckpt_path, _ = load_latest({
        'text_enc': text_enc,
        'audio_enc': audio_enc,
        'video_enc': video_enc,
        'fusion': fusion,
        'classifier': classifier
    })
    st.session_state['ckpt_loaded'] = ckpt_path or ''
    return text_enc, audio_enc, video_enc, fusion, classifier

text_enc, audio_enc, video_enc, fusion, classifier = load_models()

CUSTOM_CSS = """
<style>
.block-container {padding-top:1rem;}
h1, h2, h3 {font-family: 'Inter', system-ui, sans-serif; font-weight:600;}
.prob-card {background:#111827;padding:1rem 1.25rem;border-radius:10px;border:1px solid #1f2937;margin-top:0.5rem;}
.prob-label {font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;color:#9ca3af;}
.prob-value {font-size:1.6rem;font-weight:600;color:#f9fafb;}
.small {font-size:0.7rem;color:#64748b;}
.expr-badge {position:absolute;top:8px;left:8px;background:#7c3aed;color:#fff;padding:4px 10px;border-radius:6px;font-size:0.7rem;font-weight:600;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
# 🧠 Multimodal Mental Health Detection
Real-time fusion of Text, Audio, and Video with facial expression overlay.
""")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    live_mode = st.toggle("Live webcam", value=False)
    auto_update = st.checkbox("Auto-update", value=True) if live_mode else False
    user_text = st.text_area("User Text", height=120, placeholder="Describe how you feel...")
    audio_file = st.file_uploader("Audio (wav/mp3 optional)", type=['wav','mp3'])
    video_file = None if live_mode else st.file_uploader("Video (mp4)", type=['mp4'])
    st.caption("Checkpoint: " + ("✅" if st.session_state.get('ckpt_loaded') else "❌"))
    st.caption("Device: " + str(cfg.device))

if 'live_history' not in st.session_state:
    st.session_state['live_history'] = []
if 'last_probs' not in st.session_state:
    st.session_state['last_probs'] = None

def recommend(probs):
    if probs is None: return "No prediction yet"
    label = LABELS[int(np.argmax(probs))]
    return {
        'Normal': "Maintain healthy routines; continue monitoring.",
        'Anxiety': "Try breathing exercises; seek support if persistent.",
        'Stress': "Take breaks, hydrate, consider task prioritization.",
        'Depression': "Reach out to support; consider professional help.",
    }.get(label, "Monitoring recommended.")

def predict_batch(text: str, audio_path: str, video_path: str):
    tok = preprocess_text_batch([text])
    tvec = text_enc(tok['input_ids'].to(cfg.device), tok['attention_mask'].to(cfg.device))
    feats = extract_audio_features(audio_path)
    mfcc_batch = torch.tensor(feats['mfcc'], dtype=torch.float32).unsqueeze(0)
    avec = audio_enc(mfcc_batch.to(cfg.device), feats['pitch'], feats['energy'], feats['jitter'], feats['shimmer'])
    vfeats = process_video(video_path)
    frame_tensor = torch.zeros(1,3,cfg.video_frame_size,cfg.video_frame_size).to(cfg.device)
    landmarks = torch.tensor(vfeats['landmarks'], dtype=torch.float32).to(cfg.device)
    vvec = video_enc(frame_tensor, landmarks)
    _, fused_seq = fusion(tvec, avec, vvec)
    logits = classifier(fused_seq)
    return torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

def compute_live(text, frame):
    if frame is None or not text: return None
    tok = preprocess_text_batch([text])
    tvec = text_enc(tok['input_ids'].to(cfg.device), tok['attention_mask'].to(cfg.device))
    resized = cv2.resize(frame, (cfg.video_frame_size, cfg.video_frame_size))
    ften = torch.tensor(resized).permute(2,0,1).unsqueeze(0).float()/255.0
    landmarks = torch.zeros(1,468,3)
    vvec = video_enc(ften.to(cfg.device), landmarks.to(cfg.device))
    if audio_file:
        if 'cached_audio_bytes' not in st.session_state:
            st.session_state['cached_audio_bytes'] = audio_file.getvalue()
        with NamedTemporaryFile(suffix='.wav') as af:
            af.write(st.session_state['cached_audio_bytes']); af.flush()
            feats = extract_audio_features(af.name)
            mfcc_batch = torch.tensor(feats['mfcc'], dtype=torch.float32).unsqueeze(0)
            avec = audio_enc(mfcc_batch.to(cfg.device), feats['pitch'], feats['energy'], feats['jitter'], feats['shimmer'])
    else:
        dummy_mfcc = torch.zeros(1,120,cfg.n_mfcc)
        avec = audio_enc(dummy_mfcc.to(cfg.device), torch.zeros(10), torch.zeros(10), torch.zeros(1), torch.zeros(1))
    _, fused_seq = fusion(tvec, avec, vvec)
    logits = classifier(fused_seq)
    return torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

# Live video handling
live_frame = None
live_expr = 'Unknown'
if live_mode:
    st.markdown("### Live Webcam")
    if _HAS_WEBRTC:
        class VP(VideoProcessorBase):
            def __init__(self):
                self.count = 0
                self.latest = None
                self.expr = 'Unknown'
            def recv(self, frame):
                try:
                    img = frame.to_ndarray(format="bgr24")
                except Exception:
                    return frame
                self.count += 1
                # Sample every 5th frame for light processing
                if self.count % 5 == 0:
                    e = detect_expression(img)
                    self.expr = e
                    self.latest = overlay_expression(img, e)
                return frame
        try:
            ctx = webrtc_streamer(
                key="live",
                video_processor_factory=VP,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
        except Exception as e:
            st.warning(f"WebRTC initialization failed: {e}. Falling back to manual capture.")
            ctx = None
            _HAS_WEBRTC = False
        if ctx and getattr(ctx, 'video_processor', None) and ctx.video_processor.latest is not None:
            live_frame = ctx.video_processor.latest
            live_expr = ctx.video_processor.expr
            if _HAS_CV2 and live_frame is not None:
                st.image(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB), caption=f"Expression: {live_expr}")
            else:
                st.image(live_frame, caption=f"Expression: {live_expr}")
        else:
            st.info("Waiting for frames... (Allow camera permission or use manual snapshot below)")
    else:
        snap = st.camera_input("Capture frame")
        if snap:
            img = Image.open(snap)
            live_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            live_expr = detect_expression(live_frame)
            st.image(cv2.cvtColor(overlay_expression(live_frame, live_expr), cv2.COLOR_BGR2RGB), caption=f"Expression: {live_expr}")

    col_a, col_b, col_c = st.columns(3)
    manual = col_b.button("Manual Predict")
    trigger = (auto_update and live_frame is not None and len(user_text) > 5) or manual
    if trigger and live_frame is not None:
        try:
            probs = compute_live(user_text, live_frame)
            if probs is not None:
                st.session_state['last_probs'] = probs
                st.session_state['live_history'].append({'t': time.time(), 'expr': live_expr, 'text_len': len(user_text), 'probs': probs.tolist()})
        except Exception as e:
            st.error(f"Live prediction error: {e}")
    if st.session_state['last_probs'] is not None:
        p = st.session_state['last_probs']
        df_live = pd.DataFrame({'label': LABELS, 'prob': p})
        chart = alt.Chart(df_live).mark_bar(cornerRadius=4).encode(
            x=alt.X('prob:Q', scale=alt.Scale(domain=[0,1])),
            y=alt.Y('label:N', sort='-x'),
            color=alt.Color('label:N', legend=None, scale=alt.Scale(domain=LABELS, range=['#0ea5e9','#6366f1','#f59e0b','#ef4444'])),
            tooltip=['label','prob']
        ).properties(height=160)
        st.altair_chart(chart, use_container_width=True)
        top_label = LABELS[int(np.argmax(p))]
        st.markdown(f"<div class='prob-card'><div class='prob-label'>Top Prediction</div><div class='prob-value'>{top_label}</div></div>", unsafe_allow_html=True)
        st.caption(recommend(p))
    else:
        st.caption("No live prediction yet.")
    if st.button("Download Live Report") and st.session_state['live_history']:
        report = {
            'events': st.session_state['live_history'],
            'final_probs': st.session_state['last_probs'].tolist() if st.session_state['last_probs'] is not None else None,
            'recommendation': recommend(st.session_state['last_probs']) if st.session_state['last_probs'] is not None else None,
            'count': len(st.session_state['live_history'])
        }
        st.download_button("Save JSON", data=json.dumps(report, indent=2), file_name="live_session_report.json")

# Tabs for upload prediction & history
tabs = st.tabs(["Upload Predict", "Session History", "About"])

with tabs[0]:
    st.markdown("### Batch Upload Prediction")
    if st.button("Run Upload Prediction"):
        if not (audio_file and video_file and user_text):
            st.warning("Provide text, audio, and video.")
        else:
            with NamedTemporaryFile(suffix='.wav') as af, NamedTemporaryFile(suffix='.mp4') as vf:
                af.write(audio_file.read()); af.flush(); vf.write(video_file.read()); vf.flush()
                probs = predict_batch(user_text, af.name, vf.name)
                df = pd.DataFrame({'label': LABELS, 'prob': probs})
                chart = alt.Chart(df).mark_bar(cornerRadius=4).encode(
                    x=alt.X('prob:Q', scale=alt.Scale(domain=[0,1])),
                    y=alt.Y('label:N', sort='-x'),
                    color=alt.Color('label:N', legend=None, scale=alt.Scale(domain=LABELS, range=['#0ea5e9','#6366f1','#f59e0b','#ef4444']))
                ).properties(height=180)
                st.altair_chart(chart, use_container_width=True)
                top_label = LABELS[int(np.argmax(probs))]
                st.markdown(f"<div class='prob-card'><div class='prob-label'>Top Prediction</div><div class='prob-value'>{top_label}</div></div>", unsafe_allow_html=True)
                st.caption(recommend(probs))

with tabs[1]:
    st.markdown("### Live Session History")
    hist = st.session_state.get('live_history', [])
    if hist:
        dfh = pd.DataFrame(hist)
        st.dataframe(dfh.tail(50), use_container_width=True)
    else:
        st.info("No live events yet.")

with tabs[2]:
    st.markdown("""### About
Demonstration interface – not a medical tool. Predictions are illustrative.
Planned improvements: microphone streaming, landmark emotion model, fine-tuning with real dataset.
""")

# Sidebar diagnostics
with st.sidebar:
    st.subheader("Diagnostics")
    lp = st.session_state.get('last_probs')
    if lp is not None:
        st.write({LABELS[i]: round(float(lp[i]),3) for i in range(len(LABELS))})
    st.write("Live mode:", live_mode, "Auto:", auto_update)
    st.write("Audio uploaded:", audio_file is not None)
    st.write("Frame available:", live_frame is not None)
    if live_mode and _HAS_WEBRTC and 'ctx' in locals() and ctx and ctx.video_processor:
        st.write("Expression:", getattr(ctx.video_processor, 'expr', 'N/A'))
    st.caption("UI theme + Altair visualization active")

live_frame = None
if live_mode:
    st.markdown("Webcam Stream (WebRTC) & Snapshot")
    if _HAS_WEBRTC:
        class LiveVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.counter = 0
                self.latest_frame = None
                self.latest_expr = 'Unknown'
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self.counter += 1
                if self.counter % 5 == 0:
                    expr = detect_expression(img)
                    img_overlay = overlay_expression(img, expr)
                    self.latest_expr = expr
                    self.latest_frame = img_overlay
                return frame
        webrtc_ctx = webrtc_streamer(key="live_video", video_processor_factory=LiveVideoProcessor, media_stream_constraints={"video": True, "audio": False})
        if webrtc_ctx and webrtc_ctx.video_processor and webrtc_ctx.video_processor.latest_frame is not None:
            live_frame = webrtc_ctx.video_processor.latest_frame
            st.image(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB), caption=f"Latest stream frame (Expr: {webrtc_ctx.video_processor.latest_expr})", use_column_width=True)
            st.caption(f"Frame shape: {live_frame.shape}")
        else:
            st.caption("Waiting for stream frames...")
    else:
        st.caption("streamlit-webrtc not available; fallback to manual camera snapshot below.")
        cam_image = st.camera_input("Capture frame")
        if cam_image is not None:
            image = Image.open(cam_image)
            live_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if 'live_history' not in st.session_state:
    st.session_state['live_history'] = []
if 'last_probs' not in st.session_state:
    st.session_state['last_probs'] = None

def recommend(probs):
    if probs is None:
        return "No data yet"
    idx = int(np.argmax(probs))
    label = LABELS[idx]
    if label == 'Normal':
        return "Normal range: Maintain healthy routines and monitor mood."
    if label == 'Anxiety':
        return "Indicators of anxiety: Consider breathing exercises, short breaks, and if persistent, consult a professional."
    if label == 'Stress':
        return "Stress detected: Prioritize rest, hydration, task delegation, and brief mindfulness practice."
    if label == 'Depression':
        return "Depressive signals: Reach out to support network; if enduring, seek professional evaluation."
    return "General monitoring recommended."

def compute_live_prediction(text, frame):
    if frame is None or not text:
        return None
    tokenized = preprocess_text_batch([text])
    text_vec = text_enc(tokenized['input_ids'].to(cfg.device), tokenized['attention_mask'].to(cfg.device))
    resized = cv2.resize(frame, (cfg.video_frame_size, cfg.video_frame_size))
    frame_tensor = torch.tensor(resized).permute(2,0,1).unsqueeze(0).float()/255.0
    landmarks = torch.zeros(1,468,3)
    video_vec = video_enc(frame_tensor.to(cfg.device), landmarks.to(cfg.device))
    # Audio optional; if uploaded treat it once
    if audio_file:
        if 'cached_audio_bytes' not in st.session_state:
            st.session_state['cached_audio_bytes'] = audio_file.getvalue()
        with NamedTemporaryFile(suffix='.wav') as af:
            af.write(st.session_state['cached_audio_bytes']); af.flush()
            feats = extract_audio_features(af.name)
            mfcc_batch = torch.tensor(feats['mfcc'], dtype=torch.float32).unsqueeze(0)  # (1,T,F)
            audio_vec = audio_enc(mfcc_batch.to(cfg.device), feats['pitch'], feats['energy'], feats['jitter'], feats['shimmer'])
    else:
        dummy_mfcc = torch.zeros(1,120,cfg.n_mfcc)
        audio_vec = audio_enc(dummy_mfcc.to(cfg.device), torch.zeros(10), torch.zeros(10), torch.zeros(1), torch.zeros(1))
    fused_pooled, fused_seq = fusion(text_vec, audio_vec, video_vec)
    logits = classifier(fused_seq)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return probs

if live_mode:
    st.subheader("Live Prediction")
    col_live_btn, col_manual_btn, col_report_btn = st.columns(3)
    auto_trigger = False
    if auto_update:
        auto_trigger = True
    # Manual predict button always available in live mode
    manual_clicked = col_manual_btn.button("Manual Live Predict")
    generate_report_clicked = col_report_btn.button("Generate Live Report")
    # Attempt to extract frame from WebRTC processor state if available
    if _HAS_WEBRTC and 'latest_frame' in st.session_state:
        live_frame = st.session_state['latest_frame']
    # Trigger conditions
    trigger = (auto_trigger and live_frame is not None and len(user_text) > 5) or manual_clicked
    if trigger and live_frame is not None:
        try:
            probs = compute_live_prediction(user_text, live_frame)
            if probs is not None:
                st.session_state['last_probs'] = probs
                st.session_state['live_history'].append({'t': time.time(), 'text_len': len(user_text), 'probs': probs.tolist()})
        except Exception as e:
            st.error(f"Live prediction error: {e}")
    if st.session_state['last_probs'] is not None:
        df_live = pd.DataFrame({'label': LABELS, 'prob': st.session_state['last_probs']})
        st.bar_chart(df_live.set_index('label'))
        st.info(recommend(st.session_state['last_probs']))
    else:
        st.caption("No live prediction yet. Type text and capture frames.")
    if (generate_report_clicked or st.button("Download Session Report")) and st.session_state['live_history']:
        report = {
            'samples': st.session_state['live_history'],
            'final_probs': st.session_state['last_probs'].tolist() if st.session_state['last_probs'] is not None else None,
            'recommendation': recommend(st.session_state['last_probs']) if st.session_state['last_probs'] is not None else None,
            'total_events': len(st.session_state['live_history'])
        }
        st.download_button("Download JSON Report", data=json.dumps(report, indent=2), file_name="live_session_report.json")

if st.button("Predict") and not (live_mode and auto_update):
    if live_mode:
        if live_frame is None or not user_text:
            st.warning("Capture a webcam frame and enter text (audio optional).")
        else:
            # Handle audio optional: if provided treat as standard path; else fallback zeros
            if audio_file:
                with NamedTemporaryFile(suffix='.wav') as af:
                    af.write(audio_file.read()); af.flush()
                    audio_feats = extract_audio_features(af.name)
                    audio_vec = audio_enc(
                        torch.tensor(audio_feats['mfcc'], dtype=torch.float32).to(cfg.device),
                        audio_feats['pitch'], audio_feats['energy'], audio_feats['jitter'], audio_feats['shimmer']
                    )
            else:
                # Fallback: zero vector shaped like text encoder output (will be projected in fusion)
                dummy_mfcc = torch.zeros(1,120,cfg.n_mfcc)
                audio_vec = audio_enc(dummy_mfcc.squeeze(0).to(cfg.device), torch.zeros(10), torch.zeros(10), torch.zeros(1), torch.zeros(1))
            tokenized = preprocess_text_batch([user_text])
            text_vec = text_enc(tokenized['input_ids'].to(cfg.device), tokenized['attention_mask'].to(cfg.device))
            # Prepare single frame tensor
            resized = cv2.resize(live_frame, (cfg.video_frame_size, cfg.video_frame_size))
            frame_tensor = torch.tensor(resized).permute(2,0,1).unsqueeze(0).float()/255.0
            landmarks = torch.zeros(1,468,3)  # mediapipe fallback
            video_vec = video_enc(frame_tensor.to(cfg.device), landmarks.to(cfg.device))
            fused_pooled, fused_seq = fusion(text_vec, audio_vec, video_vec)
            logits = classifier(fused_seq)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            # Update session state for report generation & history
            st.session_state['last_probs'] = probs
            st.session_state['live_history'].append({'t': time.time(), 'text_len': len(user_text), 'probs': probs.tolist()})
            # Display chart in consistent format
            df_live_manual = pd.DataFrame({'label': LABELS, 'prob': probs})
            st.bar_chart(df_live_manual.set_index('label'))
            top_label = LABELS[int(probs.argmax())]
            st.markdown(
                f"<div class='prob-card'><div class='prob-label'>Top Prediction</div><div class='prob-value'>{top_label}</div></div>",
                unsafe_allow_html=True
            )
            st.info(recommend(probs))
    tabs = st.tabs(["Batch Upload Predict", "Session Summary", "About"])

    with tabs[0]:
        if st.button("Run Upload Prediction") and not (live_mode and auto_update):
            # Validate required inputs before running prediction
            if not (audio_file and video_file and user_text):
                st.warning("Provide text, audio, and video.")
            else:
                with NamedTemporaryFile(suffix='.wav') as af, NamedTemporaryFile(suffix='.mp4') as vf:
                    af.write(audio_file.read()); af.flush()
                    vf.write(video_file.read()); vf.flush()
                    tokenized = preprocess_text_batch([user_text])
                    text_vec = text_enc(tokenized['input_ids'].to(cfg.device), tokenized['attention_mask'].to(cfg.device))
                    feats = extract_audio_features(af.name)
                    mfcc_batch = torch.tensor(feats['mfcc'], dtype=torch.float32).unsqueeze(0)
                    audio_vec = audio_enc(mfcc_batch.to(cfg.device), feats['pitch'], feats['energy'], feats['jitter'], feats['shimmer'])
                    video_feats = process_video(vf.name)
                    frame_tensor = torch.zeros(1,3,cfg.video_frame_size,cfg.video_frame_size).to(cfg.device)
                    landmarks = torch.tensor(video_feats['landmarks'], dtype=torch.float32).to(cfg.device)
                    video_vec = video_enc(frame_tensor, landmarks)
                    fused_pooled, fused_seq = fusion(text_vec, audio_vec, video_vec)
                    logits = classifier(fused_seq)
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                    df_pred = pd.DataFrame({'label': LABELS, 'prob': probs})
                    st.bar_chart(df_pred.set_index('label'))
                    st.success(f"Prediction: {LABELS[int(probs.argmax())]}")
                    st.write("Debug shapes:", {
                        'mfcc_batch': list(mfcc_batch.shape),
                        'audio_vec': list(audio_vec.shape),
                        'video_vec': list(video_vec.shape),
                        'text_vec': list(text_vec.shape),
                        'fused_seq': list(fused_seq.shape)
                    })

# Error diagnostic helper
st.sidebar.subheader("Diagnostics")
st.sidebar.write("Checkpoint loaded:" if os.path.exists('outputs/ckpt_final.pt') else "No ckpt_final.pt")
if 'last_probs' in st.session_state and st.session_state['last_probs'] is not None:
    st.sidebar.write("Last probs:", {LABELS[i]: float(st.session_state['last_probs'][i]) for i in range(len(LABELS))})
st.sidebar.write("Live mode:", live_mode, "Auto update:", auto_update)
st.sidebar.write("Audio uploaded:", audio_file is not None)
st.sidebar.write("Frame available:", live_frame is not None)
if _HAS_WEBRTC and 'live_video' in locals():
    pass
if live_mode and _HAS_WEBRTC and 'webrtc_ctx' in locals() and webrtc_ctx and webrtc_ctx.video_processor:
    st.sidebar.write("Expression:", getattr(webrtc_ctx.video_processor, 'latest_expr', 'N/A'))
if live_frame is None and live_mode:
    st.sidebar.warning("No live frame yet - wait for stream or capture snapshot.")

# Dummy sample prediction button for troubleshooting
st.sidebar.markdown("---")
if st.sidebar.button("Run Dummy Sample Predict"):
    try:
        dummy_text = "I feel okay but sometimes stressed about work deadlines"
        tok = preprocess_text_batch([dummy_text])
        tvec = text_enc(tok['input_ids'].to(cfg.device), tok['attention_mask'].to(cfg.device))
        mfcc_dummy = torch.randn(1, 100, cfg.n_mfcc)
        avec = audio_enc(mfcc_dummy.to(cfg.device), torch.randn(30), torch.randn(30), torch.randn(1), torch.randn(1))
        frame_tensor = torch.zeros(1,3,cfg.video_frame_size,cfg.video_frame_size)
        landmarks = torch.zeros(1,468,3)
        vvec = video_enc(frame_tensor.to(cfg.device), landmarks.to(cfg.device))
        fused_pooled, fused_seq = fusion(tvec, avec, vvec)
        logits = classifier(fused_seq)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        st.sidebar.success("Dummy prediction OK")
        # Optional sidebar debug info
        st.sidebar.write({LABELS[i]: float(probs[i]) for i in range(len(LABELS))})
        st.sidebar.write({
            'text_vec': list(tvec.shape),
            'audio_vec': list(avec.shape),
            'video_vec': list(vvec.shape),
            'fused_seq': list(fused_seq.shape)
        })
    except Exception as e:
        st.sidebar.error(f"Dummy prediction failed: {e}")
