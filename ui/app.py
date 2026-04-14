"""
Forest Fire Prediction & Simulation — Streamlit UI
A modern, presentation-ready dashboard with prediction maps,
fire spread animation, and model evaluation.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import io
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation.ca_model import simulate_fire, save_animation

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Fire AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ─── Global ─── */
    * { font-family: 'Inter', sans-serif; }
    .block-container {
        padding-top: 1rem; padding-bottom: 1rem;
        max-width: 1100px;
    }

    /* ─── Hero ─── */
    .hero {
        text-align: center; padding: 2rem 0 0.5rem;
    }
    .hero h1 {
        font-size: 2.6rem; font-weight: 900; margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #FF6B35 0%, #FF2D2D 40%, #FF6B35 70%, #FFD700 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero .tagline {
        color: #8892a4; font-size: 0.95rem; margin-top: 0.2rem;
        font-weight: 400;
    }

    /* ─── Metric strip ─── */
    .metric-strip {
        display: flex; gap: 0.6rem; justify-content: center;
        flex-wrap: wrap; margin: 1rem 0 1.5rem;
    }
    .m-card {
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        border: 1px solid rgba(255,107,53,0.15);
        border-radius: 14px;
        padding: 0.65rem 1.5rem;
        text-align: center;
        min-width: 120px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .m-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255,107,53,0.45);
    }
    .m-card .lbl {
        font-size: 0.65rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.1em; font-weight: 600;
    }
    .m-card .val {
        font-size: 1.45rem; font-weight: 800;
        background: linear-gradient(135deg, #FF6B35, #FFD700);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* ─── Section headers ─── */
    .sec-header {
        display: flex; align-items: center; gap: 0.6rem;
        margin: 1.8rem 0 0.8rem; padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,107,53,0.2);
    }
    .sec-header .icon {
        font-size: 1.4rem;
    }
    .sec-header .title {
        font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
        letter-spacing: -0.2px;
    }
    .sec-header .badge {
        font-size: 0.6rem; font-weight: 600; color: #FF6B35;
        background: rgba(255,107,53,0.1);
        padding: 0.15rem 0.55rem; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* ─── Glass cards ─── */
    .glass-card {
        background: linear-gradient(135deg, rgba(17,24,39,0.8), rgba(30,41,59,0.6));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1rem;
        backdrop-filter: blur(12px);
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255,107,53,0.25);
    }

    /* ─── Plot captions ─── */
    .caption {
        text-align: center; font-size: 0.75rem; color: #6b7280;
        margin-top: 0.3rem; padding-bottom: 0.3rem; font-weight: 500;
    }

    /* ─── Sim result stats ─── */
    .sim-stats {
        display: flex; gap: 0.5rem; justify-content: center;
        flex-wrap: wrap; margin-top: 0.6rem;
    }
    .sim-stat {
        background: rgba(17,24,39,0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 0.4rem 1rem;
        text-align: center; min-width: 90px;
    }
    .sim-stat .s-lbl { font-size: 0.6rem; color: #6b7280; text-transform: uppercase; font-weight: 600; letter-spacing: 0.08em; }
    .sim-stat .s-val { font-size: 1.15rem; font-weight: 700; }
    .sim-stat .s-val.burning { color: #ff4444; }
    .sim-stat .s-val.burnedout { color: #6b7280; }
    .sim-stat .s-val.unburned { color: #22c55e; }

    /* ─── Controls panel ─── */
    .ctrl-panel {
        background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(30,41,59,0.7));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 1.2rem;
    }
    .ctrl-title {
        font-size: 0.85rem; font-weight: 700; color: #e2e8f0;
        margin-bottom: 0.6rem; letter-spacing: -0.2px;
    }
    .legend-box {
        font-size: 0.75rem; color: #8892a4; line-height: 1.7;
        margin-top: 0.8rem; padding: 0.8rem;
        background: rgba(0,0,0,0.2); border-radius: 10px;
    }

    /* ─── Dividers ─── */
    .divider {
        border: none; border-top: 1px solid rgba(255,255,255,0.05);
        margin: 1.5rem 0;
    }

    /* ─── Frame slider ─── */
    .frame-label {
        text-align: center; font-size: 0.8rem; font-weight: 600;
        color: #FF6B35; margin-bottom: 0.3rem;
    }

    /* ─── Buttons ─── */
    .stButton > button {
        border-radius: 10px; font-weight: 600;
        border: 1px solid rgba(255,107,53,0.35);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #FF6B35;
        background: rgba(255,107,53,0.1);
        box-shadow: 0 0 20px rgba(255,107,53,0.15);
    }

    /* ─── Hide streamlit chrome ─── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ─── Tabs styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; font-weight: 600; font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ────────────────────────────────────
@st.cache_data
def load_data():
    """Load all prediction outputs."""
    data = {}
    data["probs"] = np.load("outputs/predictions/fire_map.npy")
    data["mask"] = np.load("outputs/predictions/fire_mask.npy")

    conf_path = "outputs/predictions/confidence.npy"
    data["confidence"] = np.load(conf_path) if os.path.exists(conf_path) else None

    t_path = "outputs/evaluation/best_threshold.txt"
    data["threshold"] = float(open(t_path).read().strip()) if os.path.exists(t_path) else 0.5

    return data


data = load_data()
probs = data["probs"]
mask = data["mask"]
confidence = data["confidence"]
threshold = data["threshold"]


# ── Helper: render a compact matplotlib figure ───────────────
def render_map(arr, cmap, title, vmin=None, vmax=None, colorbar=True, figsize=(4.5, 4.5)):
    """Create a compact, styled matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=8,
                 fontfamily="sans-serif")
    ax.axis("off")

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
        cbar.ax.tick_params(colors="white", labelsize=7)
        cbar.outline.set_edgecolor("#ffffff20")

    fig.tight_layout(pad=0.4)
    return fig


def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0e1117", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔥 Forest Fire AI</h1>
    <p class="tagline">CNN-based fire risk assessment with Cellular Automata spread simulation</p>
</div>
""", unsafe_allow_html=True)

# ── Metric strip ─────────────────────────────────────────────
fire_px = int(mask.sum())
total_px = int(mask.size)
fire_pct = 100 * fire_px / total_px
avg_prob = float(probs.mean()) * 100
max_prob = float(probs.max()) * 100

st.markdown(f"""
<div class="metric-strip">
    <div class="m-card">
        <div class="lbl">Fire Pixels</div>
        <div class="val">{fire_px:,}</div>
    </div>
    <div class="m-card">
        <div class="lbl">Coverage</div>
        <div class="val">{fire_pct:.1f}%</div>
    </div>
    <div class="m-card">
        <div class="lbl">Threshold</div>
        <div class="val">{threshold:.3f}</div>
    </div>
    <div class="m-card">
        <div class="lbl">Avg Prob</div>
        <div class="val">{avg_prob:.1f}%</div>
    </div>
    <div class="m-card">
        <div class="lbl">Resolution</div>
        <div class="val">{probs.shape[0]}×{probs.shape[1]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — Prediction Maps
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-header">
    <span class="icon">🔥</span>
    <span class="title">Prediction Maps</span>
    <span class="badge">CNN Output</span>
</div>
""", unsafe_allow_html=True)

num_maps = 3 if confidence is not None else 2
cols = st.columns(num_maps, gap="medium")

# ── Fire probability ──
with cols[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fire_cmap = mcolors.LinearSegmentedColormap.from_list(
        "fire", ["#0a0a1a", "#1a1a3e", "#ff6b35", "#ff2d2d", "#ffdd00"])
    fig = render_map(probs, fire_cmap, "Fire Probability", vmin=0, vmax=1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('<p class="caption">Predicted probability of fire at each pixel</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Binary mask ──
with cols[1]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    mask_cmap = mcolors.LinearSegmentedColormap.from_list("mask", ["#0a0a1a", "#ff2d2d"])
    fig = render_map(mask, mask_cmap, f"Fire Mask (t={threshold:.2f})",
                     vmin=0, vmax=1, colorbar=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('<p class="caption">Binary classification — fire vs no-fire</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Confidence map ──
if confidence is not None:
    with cols[2]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = render_map(confidence, "inferno", "Model Confidence", vmin=0, vmax=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('<p class="caption">Distance from decision boundary (higher → more certain)</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — Fire Spread Simulation
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-header">
    <span class="icon">🎥</span>
    <span class="title">Fire Spread Simulation</span>
    <span class="badge">Cellular Automata</span>
</div>
""", unsafe_allow_html=True)

ctrl_col, vis_col = st.columns([1, 2.5], gap="large")

with ctrl_col:
    st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ctrl-title">⚙️ Simulation Controls</div>', unsafe_allow_html=True)

    hours = st.selectbox("Simulation duration", [1, 2, 3, 6, 12],
                         index=2, help="Number of time steps to simulate")

    mode = st.radio("Visualization mode", ["Final Frame", "Step-by-Step Slider", "Generate GIF"],
                    help="Choose how to view the simulation results")

    run_sim = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    st.markdown("""
    <div class="legend-box">
        <strong>Cell states:</strong><br>
        🟤 Unburned &nbsp; 🔴 Burning &nbsp; ⚫ Burned out<br><br>
        Fire spreads to 8-connected neighbors weighted by predicted probability.
        Cells burn out after 3 time steps.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with vis_col:
    sim_cmap = mcolors.ListedColormap(["#0a0a1a", "#ff2d2d", "#2a2a2a"])

    if run_sim:
        with st.spinner("🔥 Simulating fire spread..."):
            frames = simulate_fire(probs, steps=hours)

        if mode == "Final Frame":
            frame = frames[-1]
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig = render_map(frame, sim_cmap, f"Fire Spread — {hours}h",
                             vmin=0, vmax=2, colorbar=False, figsize=(5.5, 5.5))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        elif mode == "Step-by-Step Slider":
            st.markdown(f'<div class="frame-label">Scrub through {len(frames)} frames</div>',
                        unsafe_allow_html=True)
            step_idx = st.slider("Time step", 0, len(frames) - 1, len(frames) - 1,
                                 label_visibility="collapsed")
            frame = frames[step_idx]
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig = render_map(frame, sim_cmap, f"Step {step_idx} / {len(frames) - 1}",
                             vmin=0, vmax=2, colorbar=False, figsize=(5.5, 5.5))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        elif mode == "Generate GIF":
            with st.spinner("🎬 Rendering animation..."):
                gif_path = save_animation(probs, steps=hours)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(gif_path, caption=f"Fire spread over {hours} hours",
                     use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Simulation stats ──
        frame = frames[-1]
        burning = int((frame == 1).sum())
        burned = int((frame == 2).sum())
        unburned = int((frame == 0).sum())

        st.markdown(f"""
        <div class="sim-stats">
            <div class="sim-stat">
                <div class="s-lbl">Burning</div>
                <div class="s-val burning">{burning:,}</div>
            </div>
            <div class="sim-stat">
                <div class="s-lbl">Burned Out</div>
                <div class="s-val burnedout">{burned:,}</div>
            </div>
            <div class="sim-stat">
                <div class="s-lbl">Unburned</div>
                <div class="s-val unburned">{unburned:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Placeholder state ──
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem 1.5rem;">
            <div style="font-size:3rem; margin-bottom:0.6rem;">🔥</div>
            <div style="font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:0.3rem;">
                Ready to Simulate
            </div>
            <div style="font-size:0.8rem; color:#6b7280; line-height:1.6;">
                Select duration & mode, then click <strong>Run Simulation</strong><br>
                to visualize fire spread using cellular automata.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pre-existing GIF ──
    gif_static = "outputs/animations/fire.gif"
    if os.path.exists(gif_static) and not run_sim:
        with st.expander("📎 View previously generated animation", expanded=False):
            st.image(gif_static, caption="Cached fire spread animation",
                     use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — Model Evaluation
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-header">
    <span class="icon">📊</span>
    <span class="title">Model Evaluation</span>
    <span class="badge">Performance</span>
</div>
""", unsafe_allow_html=True)

eval_files = {
    "pr_curve": "outputs/evaluation/pr_curve.png",
    "confusion": "outputs/evaluation/confusion_matrix.png",
    "comparison": "outputs/evaluation/prediction_comparison.png",
}
has_eval = any(os.path.exists(p) for p in eval_files.values())

if has_eval:
    tab_labels = []
    tab_keys = []
    if os.path.exists(eval_files["pr_curve"]):
        tab_labels.append("📈 Precision–Recall")
        tab_keys.append("pr_curve")
    if os.path.exists(eval_files["confusion"]):
        tab_labels.append("🔢 Confusion Matrix")
        tab_keys.append("confusion")
    if os.path.exists(eval_files["comparison"]):
        tab_labels.append("🖼️ Prediction Comparison")
        tab_keys.append("comparison")

    if tab_labels:
        tabs = st.tabs(tab_labels)
        for tab, key in zip(tabs, tab_keys):
            with tab:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                captions = {
                    "pr_curve": "Precision–Recall curve with optimal threshold",
                    "confusion": "Confusion matrix on test set",
                    "comparison": "Ground Truth vs Prediction vs Thresholded output",
                }
                st.image(eval_files[key], caption=captions[key],
                         use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Run `python training/evaluate.py` first to generate evaluation plots.", icon="📊")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="divider"></div>
<div style="text-align:center; padding:1.5rem 0 1rem;">
    <span style="font-size:0.7rem; color:#4b5563; font-weight:500;">
        Forest Fire AI — CNN + Cellular Automata &nbsp;·&nbsp; PyTorch + Streamlit
    </span>
</div>
""", unsafe_allow_html=True)