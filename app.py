import streamlit as st
from PIL import Image
import os
import json
import tempfile
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from dotenv import load_dotenv
from model import get_densenet121_model, COMPETITION_LABELS
from gradcam_utils import get_gradcam_heatmap
from llm_explainer import get_diagnosis_verdicts, format_system_prompt, get_gemini_chain

# ── Load .env silently (API key hidden from UI) ──────────────────────────
load_dotenv()
_GEMINI_KEY = os.getenv('gemini_api_key', '')

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Radiology AI Explainer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for a polished, modern look ───────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ────────────────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 8px 32px rgba(0,0,0,.25);
}
.hero-container h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: .5rem;
    letter-spacing: -0.5px;
}
.hero-container p {
    font-size: 1.05rem;
    opacity: .85;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Stat card (metric style) ───────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,.18);
    border: 1px solid rgba(255,255,255,.06);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,.28);
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4fc3f7;
    margin-bottom: .15rem;
}
.stat-card .stat-label {
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(255,255,255,.55);
}

/* ── Pathology risk badges ──────────────────────────────── */
.risk-high   { background:#ff1744; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; font-size:.82rem; }
.risk-mod    { background:#ff9100; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; font-size:.82rem; }
.risk-low    { background:#00e676; color:#111; padding:2px 10px; border-radius:12px; font-weight:600; font-size:.82rem; }

/* ── Tab styling ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    font-size: .95rem;
    padding: 0 1.4rem;
}

/* ── Nicer file-uploader ────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(79,195,247,.4);
    border-radius: 14px;
    padding: 1rem;
}

/* ── Section divider ────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,.08);
    margin: 2rem 0;
}

/* Hide default Streamlit hamburger & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_cached_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_densenet121_model(pretrained=False)
    weight_file = None
    for w in ['best_densenet121.pth',
              'best_densenet121_phase2.pth',
              'best_densenet121_phase1.pth',
              'best_resnet50.pth']:
        if os.path.exists(w):
            try:
                model.load_state_dict(
                    torch.load(w, map_location=device, weights_only=True)
                )
                weight_file = w
                break
            except Exception:
                continue
    model.to(device)
    model.eval()
    return model, device, weight_file

model, device, _weight_file = load_cached_model()


# ══════════════════════════════════════════════════════════════════════════
#  LOAD METRICS DATA (used by multiple tabs)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_metrics():
    """Return (eval_metrics_dict | None, checkpoint_dict | None)."""
    _m, _c = None, None
    mp = os.path.join('results', 'metrics.json')
    cp = 'checkpoint_phase1.pth'
    if os.path.exists(mp):
        with open(mp) as f:
            _m = json.load(f)
    if os.path.exists(cp):
        _c = torch.load(cp, map_location='cpu', weights_only=False)
    return _m, _c

@st.cache_data
def load_training_history():
    """Return training history dict or None."""
    hp = os.path.join('results', 'training_history.json')
    if os.path.exists(hp):
        with open(hp) as f:
            return json.load(f)
    return None

_eval, _ckpt = load_metrics()
_history = load_training_history()

# Derived helpers
_stanford = {
    "Atelectasis": 0.858, "Cardiomegaly": 0.832,
    "Consolidation": 0.899, "Edema": 0.924, "Pleural Effusion": 0.968,
}


def _metrics_source():
    """Return (auc_per_class, mean_auc, overall_dict, n_samples, source_label, epoch)."""
    epoch = 0
    if _ckpt:
        epoch = _ckpt.get('epoch', -1) + 1
    if _eval:
        ec = _eval.get('per_class', {})
        auc_cls = {l: m.get('auc', 0.0) or 0.0 for l, m in ec.items()}
        mean_auc = _eval.get('overall', {}).get('mean_auc', 0.0)
        overall = _eval.get('overall', {})
        n = _eval.get('num_samples', 234)
        return auc_cls, mean_auc, overall, n, "Best-weights evaluation", epoch
    elif _ckpt:
        auc_cls = _ckpt.get('auc_per_class', {})
        mean_auc = _ckpt.get('best_mean_auc', 0.0)
        return auc_cls, mean_auc, {}, 234, "Checkpoint (last epoch)", epoch
    return {}, 0.0, {}, 0, "No data", 0


# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════
for key, default in [
    ("conversation", None),
    ("chat_history", []),
    ("system_prompt", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <h1>🫁 Radiology AI Explainer</h1>
    <p>
        Powered by <strong>DenseNet121</strong> trained on CheXpert &mdash;
        detects <strong>5 chest pathologies</strong> with Grad-CAM++ visual explanations
        and <strong>Google Gemini</strong> AI discussion.
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════
tab_scan, tab_perf, tab_about = st.tabs([
    "🔬  Scan & Diagnose",
    "📊  Model Performance",
    "ℹ️  About",
])


# ──────────────────────────────────────────────────────────────────────────
#  TAB 1 — SCAN & DIAGNOSE
# ──────────────────────────────────────────────────────────────────────────
with tab_scan:
    st.markdown("#### Upload a Chest X-Ray to get started")
    st.caption("Supported formats: JPG, JPEG, PNG — the model works best with PA (posterior-anterior) views.")

    uploaded_file = st.file_uploader(
        "Drop your X-Ray here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_img_path = tmp_file.name
        image = Image.open(uploaded_file).convert('RGB')

        # ── Two-column layout: image | results ─────────────────────
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

        with col_res:
            analyse_btn = st.button("🔍  Analyze Image", use_container_width=True, type="primary")

        if analyse_btn:
            with st.spinner("Running DenseNet121 inference…"):
                transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output_logits = model(input_tensor)
                    probabilities = torch.sigmoid(output_logits)[0].cpu().numpy()

                preds = {
                    label: float(prob)
                    for label, prob in zip(COMPETITION_LABELS, probabilities)
                }

                st.session_state.preds = preds
                st.session_state.input_tensor = input_tensor
                st.session_state.tmp_img_path = tmp_img_path
                st.session_state.conversation = None
                st.session_state.chat_history = []

        # ── Show results if available ──────────────────────────────
        if "preds" in st.session_state:
            preds = st.session_state.preds

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### 🩺 Diagnosis Results")

            # Risk cards row
            risk_cols = st.columns(len(COMPETITION_LABELS))
            for i, (disease, prob) in enumerate(preds.items()):
                with risk_cols[i]:
                    if prob > 0.5:
                        badge = "risk-high"
                        label_text = "HIGH"
                    elif prob > 0.2:
                        badge = "risk-mod"
                        label_text = "MODERATE"
                    else:
                        badge = "risk-low"
                        label_text = "LOW"
                    st.markdown(f"""
                    <div style="text-align:center; padding:.6rem .3rem;">
                        <div style="font-size:.82rem; font-weight:600; margin-bottom:.3rem;">{disease}</div>
                        <div style="font-size:1.4rem; font-weight:700;">{prob:.0%}</div>
                        <span class="{badge}">{label_text}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Probability bars
            with st.expander("📋 Detailed probability breakdown", expanded=False):
                for disease, prob in preds.items():
                    st.markdown(f"**{disease}** — {prob:.1%}")
                    st.progress(prob)

            # ── Grad-CAM ──────────────────────────────────────────
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### 🔥 Grad-CAM++ Heatmap")
            st.caption("Highlights the regions the model focused on for each pathology.")

            gcam_col1, gcam_col2 = st.columns([1, 2])
            with gcam_col1:
                target_disease = st.selectbox("Pathology:", COMPETITION_LABELS)
            target_idx = COMPETITION_LABELS.index(target_disease)

            with st.spinner("Generating heatmap…"):
                heatmap_vis, active_regions = get_gradcam_heatmap(
                    model,
                    st.session_state.input_tensor.cpu(),
                    model.features.denseblock4,
                    st.session_state.tmp_img_path,
                    target_category_idx=target_idx,
                )

            gcam_left, gcam_right = st.columns(2)
            with gcam_left:
                st.image(image, caption="Original", use_container_width=True)
            with gcam_right:
                st.image(heatmap_vis, caption=f"{target_disease} — {active_regions}",
                         use_container_width=True)

            # ── AI Radiologist Chat ───────────────────────────────
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("### 💬 AI Radiologist")
            st.caption("Ask follow-up questions about the findings — powered by Google Gemini.")

            api_key = _GEMINI_KEY
            if api_key:
                # Initialise conversation if needed
                if st.session_state.conversation is None:
                    with st.spinner("Connecting to Gemini…"):
                        try:
                            chain, model_used = get_gemini_chain(api_key)
                            st.session_state.conversation = chain
                        except RuntimeError as e:
                            st.error(f"Could not connect to Gemini: {e}")
                            st.stop()

                    prob_text_format = get_diagnosis_verdicts(preds)
                    st.session_state.system_prompt = format_system_prompt(
                        prob_text_format, active_regions
                    )

                    try:
                        response = st.session_state.conversation.invoke({
                            "system_prompt": st.session_state.system_prompt,
                            "history": [],
                            "input": "Translate these findings into a clear and empathetic medical impression."
                        })
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response.content}
                        )
                    except Exception as e:
                        st.error(f"Gemini API error: {e}")

                # Chat history
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                if prompt := st.chat_input("Ask a follow-up question…"):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking…"):
                            from langchain_core.messages import HumanMessage, AIMessage
                            history_msgs = []
                            for m in st.session_state.chat_history[:-1]:
                                if m["role"] == "user":
                                    history_msgs.append(HumanMessage(content=m["content"]))
                                else:
                                    history_msgs.append(AIMessage(content=m["content"]))
                            try:
                                resp = st.session_state.conversation.invoke({
                                    "system_prompt": st.session_state.system_prompt,
                                    "history": history_msgs,
                                    "input": prompt,
                                })
                                st.markdown(resp.content)
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": resp.content}
                                )
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.info(
                    "AI chat is unavailable — no Gemini API key found. "
                    "Add `gemini_api_key=YOUR_KEY` to a `.env` file in the project root."
                )


# ──────────────────────────────────────────────────────────────────────────
#  TAB 2 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────
with tab_perf:
    auc_cls, mean_auc, overall, n_samples, source, epoch = _metrics_source()

    if not auc_cls:
        st.info("No metrics available yet. Train the model with `python train.py` "
                "then run `python evaluate.py`.")
    else:
        st.caption(f"DenseNet121 · {epoch} epochs on CheXpert (Phase 1) · "
                   f"{n_samples} validation images · Source: {source}")

        # ── Summary stat cards ────────────────────────────────────
        card_data = [
            ("Mean AUC", f"{mean_auc:.4f}"),
            ("Accuracy", f"{overall.get('accuracy', 0):.4f}" if overall else "—"),
            ("F1 Score", f"{overall.get('f1_score', 0):.4f}" if overall else "—"),
            ("Precision", f"{overall.get('precision', 0):.4f}" if overall else "—"),
            ("Recall", f"{overall.get('recall', 0):.4f}" if overall else "—"),
            ("Epochs", str(epoch)),
        ]

        cols = st.columns(len(card_data))
        for col, (label, value) in zip(cols, card_data):
            col.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")  # spacing

        # ── Per-class comparison subtabs ──────────────────────────
        perf_tab1, perf_tab2, perf_tab3, perf_tab4, perf_tab5 = st.tabs([
            "📋 AUC Table",
            "📊 Bar Chart",
            "🕸️ Radar Chart",
            "📈 Evaluation Curves",
            "📉 Learning Curve",
        ])

        # ···· AUC Table ···········································
        with perf_tab1:
            rows = []
            for lbl in COMPETITION_LABELS:
                our = auc_cls.get(lbl, float('nan'))
                ref = _stanford.get(lbl, float('nan'))
                gap = our - ref if not (np.isnan(our) or np.isnan(ref)) else float('nan')
                status = '🟢' if our >= 0.85 else ('🟡' if our >= 0.75 else '🔴')
                rows.append({
                    'Pathology': f"{status} {lbl}",
                    'Our AUC': f"{our:.4f}" if not np.isnan(our) else "n/a",
                    'Stanford AUC': f"{ref:.4f}" if not np.isnan(ref) else "n/a",
                    'Gap': f"{gap:+.4f}" if not np.isnan(gap) else "n/a",
                })
            rows.append({
                'Pathology': '**Mean (5 labels)**',
                'Our AUC': f"**{mean_auc:.4f}**",
                'Stanford AUC': f"**{np.mean(list(_stanford.values())):.4f}**",
                'Gap': f"**{mean_auc - np.mean(list(_stanford.values())):+.4f}**",
            })
            st.table(pd.DataFrame(rows))
            st.caption("Stanford baseline: Irvin et al. 2019 — DenseNet121 ensemble on CheXpert.")

        # ···· Bar Chart ···········································
        with perf_tab2:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(COMPETITION_LABELS))
            w = 0.35
            ours = [auc_cls.get(l, 0) for l in COMPETITION_LABELS]
            refs = [_stanford.get(l, 0) for l in COMPETITION_LABELS]

            b1 = ax.bar(x - w/2, ours, w, label='Ours', color='#4fc3f7', edgecolor='white')
            b2 = ax.bar(x + w/2, refs, w, label='Stanford', color='#7e57c2',
                        edgecolor='white', alpha=.75)

            ax.set_ylabel('AUC Score')
            ax.set_title('Per-Class AUC — Ours vs Stanford Baseline')
            ax.set_xticks(x)
            ax.set_xticklabels(COMPETITION_LABELS, fontsize=10)
            ax.set_ylim(0, 1.08)
            ax.legend()
            ax.grid(axis='y', alpha=.25)

            for bar in b1:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9)
            for bar in b2:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ···· Radar Chart ·········································
        with perf_tab3:
            labels_r = COMPETITION_LABELS + [COMPETITION_LABELS[0]]
            our_r = [auc_cls.get(l, 0) for l in COMPETITION_LABELS] + [auc_cls.get(COMPETITION_LABELS[0], 0)]
            ref_r = [_stanford.get(l, 0) for l in COMPETITION_LABELS] + [_stanford.get(COMPETITION_LABELS[0], 0)]
            angles = np.linspace(0, 2 * np.pi, len(labels_r), endpoint=True)

            fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax2.plot(angles, our_r, 'o-', color='#4fc3f7', linewidth=2, label='Ours')
            ax2.fill(angles, our_r, alpha=.15, color='#4fc3f7')
            ax2.plot(angles, ref_r, 's--', color='#7e57c2', linewidth=2, label='Stanford')
            ax2.fill(angles, ref_r, alpha=.10, color='#7e57c2')
            ax2.set_thetagrids(angles[:-1] * 180 / np.pi, COMPETITION_LABELS, fontsize=10)
            ax2.set_ylim(0, 1.05)
            ax2.set_rlabel_position(30)
            ax2.legend(loc='lower right', fontsize=10)
            ax2.set_title("AUC Profile", fontsize=13, pad=20)
            ax2.grid(True, alpha=.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ···· Evaluation Curves ···································
        with perf_tab4:
            _roc_path = os.path.join('results', 'roc_curves.png')
            _pr_path  = os.path.join('results', 'precision_recall.png')
            _cm_path  = os.path.join('results', 'confusion_matrices.png')

            curve_tabs = []
            curve_imgs = []
            if os.path.exists(_roc_path):
                curve_tabs.append("ROC Curves")
                curve_imgs.append((_roc_path, "ROC Curves — CheXpert Validation Set"))
            if os.path.exists(_pr_path):
                curve_tabs.append("Precision-Recall")
                curve_imgs.append((_pr_path, "Precision-Recall Curves"))
            if os.path.exists(_cm_path):
                curve_tabs.append("Confusion Matrices")
                curve_imgs.append((_cm_path, "Confusion Matrices @ threshold 0.5"))

            if curve_tabs:
                ctabs = st.tabs(curve_tabs)
                for ct, (img_path, caption) in zip(ctabs, curve_imgs):
                    with ct:
                        st.image(img_path, caption=caption, use_container_width=True)
            else:
                st.info("Run `python evaluate.py` to generate ROC, PR, and confusion matrix plots.")

        # ···· Learning Curve ·····································
        with perf_tab5:
            if _history and len(_history.get('epochs', [])) > 0:
                epochs_list = _history['epochs']
                train_losses = _history['train_loss']
                val_losses = _history['val_loss']
                mean_aucs = _history['mean_auc']
                lr_list = _history.get('lr', [])

                # ── Loss curve ────────────────────────────────────
                st.markdown("##### Training & Validation Loss")
                fig_loss, ax_loss = plt.subplots(figsize=(10, 4.5))
                ax_loss.plot(epochs_list, train_losses, 'o-', color='#4fc3f7',
                             linewidth=2.2, markersize=6, label='Train Loss')
                ax_loss.plot(epochs_list, val_losses, 's-', color='#ff7043',
                             linewidth=2.2, markersize=6, label='Val Loss')
                ax_loss.set_xlabel('Epoch', fontsize=11)
                ax_loss.set_ylabel('Loss', fontsize=11)
                ax_loss.set_title('Loss Curve', fontsize=13, fontweight='bold')
                ax_loss.legend(fontsize=10)
                ax_loss.grid(True, alpha=.25)
                ax_loss.set_xticks(epochs_list)
                plt.tight_layout()
                st.pyplot(fig_loss)
                plt.close()

                # ── Mean AUC curve ────────────────────────────────
                st.markdown("##### Mean AUC Over Epochs")
                fig_auc, ax_auc = plt.subplots(figsize=(10, 4.5))
                ax_auc.plot(epochs_list, mean_aucs, 'D-', color='#66bb6a',
                            linewidth=2.2, markersize=7, label='Mean AUC')
                ax_auc.axhline(y=0.85, color='#7e57c2', linestyle='--',
                               linewidth=1.5, alpha=0.7, label='Target (0.85)')
                best_idx = int(np.argmax(mean_aucs))
                ax_auc.annotate(
                    f'Best: {mean_aucs[best_idx]:.4f}',
                    xy=(epochs_list[best_idx], mean_aucs[best_idx]),
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#66bb6a',
                    arrowprops=dict(arrowstyle='->', color='#66bb6a', lw=1.5),
                )
                ax_auc.set_xlabel('Epoch', fontsize=11)
                ax_auc.set_ylabel('Mean AUC', fontsize=11)
                ax_auc.set_title('Mean AUC Over Training', fontsize=13, fontweight='bold')
                ax_auc.legend(fontsize=10)
                ax_auc.grid(True, alpha=.25)
                ax_auc.set_xticks(epochs_list)
                ax_auc.set_ylim(0, 1.05)
                plt.tight_layout()
                st.pyplot(fig_auc)
                plt.close()

                # ── Per-class AUC curves ──────────────────────────
                per_class_auc_history = _history.get('per_class_auc', [])
                if per_class_auc_history:
                    st.markdown("##### Per-Class AUC Over Epochs")
                    fig_pc, ax_pc = plt.subplots(figsize=(10, 5))
                    colors_pc = ['#4fc3f7', '#ff7043', '#66bb6a', '#ab47bc', '#ffa726']
                    for idx, label in enumerate(COMPETITION_LABELS):
                        per_label = [
                            ep_data.get(label, None) for ep_data in per_class_auc_history
                        ]
                        # Replace None with NaN for plotting
                        per_label = [v if v is not None else float('nan') for v in per_label]
                        ax_pc.plot(epochs_list, per_label, 'o-',
                                   color=colors_pc[idx % len(colors_pc)],
                                   linewidth=1.8, markersize=5, label=label)
                    ax_pc.set_xlabel('Epoch', fontsize=11)
                    ax_pc.set_ylabel('AUC', fontsize=11)
                    ax_pc.set_title('Per-Class AUC Over Training', fontsize=13, fontweight='bold')
                    ax_pc.legend(fontsize=9, loc='lower right')
                    ax_pc.grid(True, alpha=.25)
                    ax_pc.set_xticks(epochs_list)
                    ax_pc.set_ylim(0, 1.05)
                    plt.tight_layout()
                    st.pyplot(fig_pc)
                    plt.close()

                # ── Learning rate schedule ────────────────────────
                if lr_list:
                    st.markdown("##### Learning Rate Schedule")
                    fig_lr, ax_lr = plt.subplots(figsize=(10, 3.5))
                    ax_lr.plot(epochs_list, lr_list, '^-', color='#ef5350',
                               linewidth=2, markersize=6)
                    ax_lr.set_xlabel('Epoch', fontsize=11)
                    ax_lr.set_ylabel('Learning Rate', fontsize=11)
                    ax_lr.set_title('Learning Rate Schedule (CosineAnnealing)',
                                    fontsize=13, fontweight='bold')
                    ax_lr.grid(True, alpha=.25)
                    ax_lr.set_xticks(epochs_list)
                    ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
                    plt.tight_layout()
                    st.pyplot(fig_lr)
                    plt.close()

            else:
                st.info(
                    "No training history available yet. "
                    "Run `python train.py` to generate learning curves. "
                    "The training history is saved automatically after each epoch."
                )

        # ── Detailed per-class metrics expander ───────────────────
        if _eval:
            _eval_cls = _eval.get('per_class', {})
            if _eval_cls:
                with st.expander("🔍 Detailed Per-Class Metrics"):
                    detail_data = []
                    for lbl in COMPETITION_LABELS:
                        m = _eval_cls.get(lbl, {})
                        detail_data.append({
                            'Label': lbl,
                            'AUC': f"{m.get('auc', 0):.4f}",
                            'Accuracy': f"{m.get('accuracy', 0):.4f}",
                            'F1': f"{m.get('f1_score', 0):.4f}",
                            'Precision': f"{m.get('precision', 0):.4f}",
                            'Recall': f"{m.get('recall', 0):.4f}",
                        })
                    st.table(pd.DataFrame(detail_data))

        # ── Training configuration ────────────────────────────────
        with st.expander("⚙️ Training Configuration"):
            cfg1, cfg2 = st.columns(2)
            with cfg1:
                st.markdown("""
                **Model Architecture**
                - Backbone: DenseNet121 (ImageNet pre-trained)
                - Head: Linear(1024 → 5)
                - Input: 320×320 RGB
                - Output: 5 pathology probabilities (sigmoid)
                - Loss: Masked BCE with class-balanced pos_weight
                """)
            with cfg2:
                st.markdown("""
                **Training Setup**
                - Phase 1: CheXpert 5-label pre-training (10 epochs)
                - Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
                - Scheduler: CosineAnnealingLR
                - Batch size: 8 × 4 grad-accum = 32 effective
                - Validation: 234 radiologist-labelled CheXpert images
                - Primary Metric: Mean AUC over 5 competition labels
                """)


# ──────────────────────────────────────────────────────────────────────────
#  TAB 3 — ABOUT
# ──────────────────────────────────────────────────────────────────────────
with tab_about:
    a_left, a_right = st.columns([2, 1])

    with a_left:
        st.markdown("### What does this app do?")
        st.markdown("""
        This tool helps **medical professionals and students** quickly screen
        chest X-rays for five common thoracic pathologies:

        | # | Pathology | What it is |
        |---|-----------|------------|
        | 1 | **Atelectasis** | Partial or complete lung collapse |
        | 2 | **Cardiomegaly** | Enlarged heart silhouette |
        | 3 | **Consolidation** | Lung tissue filled with fluid / pus |
        | 4 | **Edema** | Excess fluid in the lungs |
        | 5 | **Pleural Effusion** | Fluid between lung and chest wall |
        """)

        st.markdown("### How it works")
        st.markdown("""
        1. **Upload** a PA chest X-ray image.
        2. **DenseNet121** (trained on the CheXpert dataset) classifies
           each pathology with a confidence score.
        3. **Grad-CAM++** generates a heatmap showing *where* the model
           looked to make its decision.
        4. **Google Gemini** provides an AI radiologist explanation you
           can discuss interactively.
        """)

        st.markdown("### Important disclaimer")
        st.warning(
            "This tool is for **educational and research purposes only**. "
            "It is NOT a substitute for professional medical diagnosis. "
            "Always consult a qualified radiologist for clinical decisions."
        )

    with a_right:
        st.markdown("### Tech Stack")
        st.markdown("""
        - **Model:** DenseNet121
        - **Dataset:** CheXpert (Stanford)
        - **XAI:** Grad-CAM++
        - **LLM:** Google Gemini
        - **Framework:** PyTorch
        - **Frontend:** Streamlit
        """)

        if _weight_file:
            st.success(f"Loaded weights: `{_weight_file}`")

        st.markdown("### Quick links")
        st.markdown("""
        - [CheXpert Paper](https://arxiv.org/abs/1901.07031)
        - [Grad-CAM++ Paper](https://arxiv.org/abs/1710.11063)
        - [Streamlit Docs](https://docs.streamlit.io)
        """)
