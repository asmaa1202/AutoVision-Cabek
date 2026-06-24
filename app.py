"""
AutoVision Cabek — Détection des pièces automobile par IA (YOLOv11)
PFE Licence Professionnelle en Ingénierie des Données
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import time
import os
import base64

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"
LOGO_PATH  = "logo.png"

def get_logo_base64(path: str) -> str:
    """Encode le logo en base64 pour l'injecter directement dans le HTML."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        ext = "jpeg" if ext == "jpg" else ext
        return f"data:image/{ext};base64,{data}"
    except Exception:
        return ""
    

st.set_page_config(
    page_title="AutoVision Cabek",
    page_icon=get_logo_base64(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best.pt"
LOGO_PATH  = "logo.png"

COUTS_MOYENS = {
    "p_Pare-brise avant":     3200, "p_Pare-brise arriere":  2800,
    "p_Capot":                4500, "p_Coffre":              3800,
    "p_Calandre":             1800, "p_Pare-chocs avant":    2200,
    "p_Pare-chocs arriere":   2000, "p_Aile avant droite":   2600,
    "p_Aile avant gauche":    2600, "p_Aile arriere droite": 2400,
    "p_Aile arriere gauche":  2400, "p_Porte avant droite":  5500,
    "p_Porte avant gauche":   5500, "p_Porte arriere droite":4800,
    "p_Porte arriere gauche": 4800, "p_Phare avant droit":   1900,
    "p_Phare avant gauche":   1900, "p_Feu arriere droit":   1200,
    "p_Feu arriere gauche":   1200, "p_Retroviseur droit":    950,
    "p_Retroviseur gauche":    950, "p_Roue":                1600,
    "p_Pneu":                  800, "p_vitre":               1500,
    "p_bas de caisse":        1100,
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Logo en base64 (évite les problèmes de chemin relatif)
# ─────────────────────────────────────────────────────────────────────────────
# def get_logo_base64(path: str) -> str:
#     """Encode le logo en base64 pour l'injecter directement dans le HTML."""
#     try:
#         with open(path, "rb") as f:
#             data = base64.b64encode(f.read()).decode("utf-8")
#         ext = os.path.splitext(path)[1].lstrip(".").lower()
#         ext = "jpeg" if ext == "jpg" else ext
#         return f"data:image/{ext};base64,{data}"
#     except Exception:
#         return ""

LOGO_B64 = get_logo_base64(LOGO_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# CSS + TOGGLE SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
logo_html = (
    f'<img src="{LOGO_B64}" alt="Logo" '
    f'style="max-width:160px;max-height:60px;object-fit:contain;'
    f'display:block;margin:0 auto 4px;" />'
    if LOGO_B64 else
    '<div style="font-size:1.8rem;margin-bottom:4px;">🔬</div>'
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background-color: #F0F4F8;
    font-family: 'Inter', sans-serif;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    border-right: 1px solid #334155;
}}
[data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: #F1F5F9 !important; }}
[data-testid="stSidebar"] .stSlider label {{
    color: #94A3B8 !important; font-size: 0.8rem !important;
}}

/* ── Cacher boutons natifs Streamlit ── */
[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
[data-testid="collapsedControl"]        {{ display: none !important; }}
#MainMenu, footer, header               {{ visibility: hidden; }}

/* ── Bouton hamburger custom ── */
#av-toggle {{
    position: fixed; top: 13px; left: 13px; z-index: 999999;
    width: 38px; height: 38px;
    background: rgba(30,58,138,0.9);
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 9px; cursor: pointer;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 5px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    transition: all .2s ease;
    backdrop-filter: blur(6px);
}}
#av-toggle:hover {{
    background: rgba(37,99,235,0.95);
    border-color: rgba(96,165,250,0.6);
    transform: scale(1.06);
}}
#av-toggle span {{
    display: block; width: 17px; height: 2px;
    background: #93C5FD; border-radius: 2px;
    transition: all .2s ease;
}}

/* ── Brand dans sidebar ── */
.sb-brand {{
    padding: 20px 16px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 16px;
    text-align: center;
}}
.sb-brand-name {{
    font-size: 1.15rem; font-weight: 700;
    color: #F1F5F9 !important;
    letter-spacing: -0.02em; margin-top: 8px;
}}
.sb-brand-sub {{
    font-size: 0.68rem; color: #475569 !important;
    text-transform: uppercase; letter-spacing: .1em; margin-top: 2px;
}}

/* ── Status pill ── */
.status-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: #F0FDF4; border: 1px solid #86EFAC;
    color: #16A34A; padding: 5px 13px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 600;
    width: 100%; justify-content: center;
}}
.status-dot {{
    width: 7px; height: 7px; background: #22C55E;
    border-radius: 50%; animation: pulse 2s infinite;
}}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.4}} }}

/* ── Stats bloc sidebar ── */
.sb-stats {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 12px 14px;
    margin: 0 0 16px 0;
}}
.sb-stat-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0;
}}
.sb-stat-lbl {{ font-size: 0.72rem; color: #475569 !important; }}
.sb-stat-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #93C5FD !important; font-weight: 600;
}}

/* ── Divider ── */
.av-div {{
    height: 1px;
    background: linear-gradient(90deg, #334155 0%, transparent 100%);
    margin: 10px 0 14px;
}}

/* ── Page header ── */
.page-title {{
    font-size: 1.75rem; font-weight: 700; color: #0F172A;
    letter-spacing: -0.025em; line-height: 1.2;
}}
.page-sub {{
    font-size: 0.88rem; color: #64748B;
    margin-top: 4px; margin-bottom: 22px;
}}

/* ── Metric cards ── */
.metric-row {{ display: flex; gap: 12px; margin-bottom: 22px; flex-wrap: wrap; }}
.metric-card {{
    flex: 1; min-width: 130px;
    background: white; border-radius: 13px;
    padding: 18px 18px 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
    position: relative; overflow: hidden;
}}
.metric-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #2563EB, #06B6D4);
}}
.mc-label {{
    font-size: 0.68rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px;
}}
.mc-value {{
    font-size: 1.85rem; font-weight: 700; color: #0F172A; line-height: 1;
}}
.mc-unit {{ font-size: 0.82rem; color: #94A3B8; font-weight: 400; margin-left: 2px; }}

/* ── Confidence ── */
.conf-badge {{
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600;
}}
.conf-high {{ background: #DCFCE7; color: #16A34A; }}
.conf-mid  {{ background: #FEF9C3; color: #B45309; }}
.conf-low  {{ background: #FEE2E2; color: #DC2626; }}
.conf-bar-wrap {{
    background: #F1F5F9; border-radius: 999px;
    height: 6px; width: 88px; overflow: hidden;
}}
.conf-bar-fill {{ height: 100%; border-radius: 999px; transition: width .5s ease; }}

/* ── Part chip ── */
.part-chip {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 11px; border-radius: 7px;
    background: #EFF6FF; color: #1E40AF;
    border: 1px solid #BFDBFE;
    font-size: 0.76rem; font-weight: 500; margin: 2px;
}}
.chip-dot {{
    width: 6px; height: 6px; border-radius: 50%; background: #3B82F6;
}}

/* ── Section label ── */
.section-lbl {{
    font-size: 0.68rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: .1em;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 12px; margin-top: 4px;
}}
.section-lbl::after {{
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #E2E8F0, transparent);
}}

/* ── Detection table ── */
.det-table {{
    width: 100%; border-collapse: collapse; font-size: 0.84rem;
}}
.det-table thead tr {{
    background: #F8FAFC; border-bottom: 2px solid #E2E8F0;
}}
.det-table thead th {{
    text-align: left; padding: 10px 16px;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; color: #64748B;
}}
.det-table tbody tr {{ border-bottom: 1px solid #F1F5F9; }}
.det-table tbody tr:hover {{ background: #FAFBFC; }}
.det-table td {{ padding: 10px 16px; color: #1E293B; vertical-align: middle; }}
.det-table td.mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem; color: #64748B;
}}

/* ── Image frame (mac style) ── */
.img-frame {{
    background: #0D1117; border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}}
.img-frame-bar {{
    background: rgba(255,255,255,0.04);
    padding: 9px 14px; display: flex; align-items: center; gap: 7px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.img-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.img-frame-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: rgba(255,255,255,0.35); margin-left: 6px;
}}

/* ── Cost box ── */
.cost-box {{
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
    border: 1px solid #A7F3D0; border-radius: 14px;
    padding: 18px 22px; margin-top: 14px;
}}
.cost-label {{ font-weight: 600; color: #065F46; font-size: 0.92rem; }}
.cost-sub   {{ font-size: 0.73rem; color: #6EE7B7; margin-top: 2px; }}
.cost-value {{
    font-size: 2rem; font-weight: 700; color: #059669;
    font-family: 'Inter', sans-serif; letter-spacing: -0.02em;
}}
.cost-currency {{ font-size: 0.9rem; color: #6EE7B7; margin-left: 4px; }}

/* ── Card container ── */
.av-card {{
    background: white; border-radius: 13px; padding: 22px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 4px 12px rgba(0,0,0,.03);
    margin-bottom: 14px;
}}

/* ── Empty state ── */
.empty-wrap {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 70px 40px; text-align: center;
    background: white; border-radius: 18px;
    border: 2px dashed #CBD5E1;
}}
.empty-icon {{
    width: 68px; height: 68px; border-radius: 16px;
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 1px solid #BFDBFE;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; margin-bottom: 18px;
}}
.empty-title {{
    font-size: 1.15rem; font-weight: 600; color: #334155; margin-bottom: 8px;
}}
.empty-sub {{ font-size: 0.84rem; color: #94A3B8; line-height: 1.65; max-width: 360px; }}

/* ── How-it-works ── */
.how-steps {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 22px; }}
.how-step {{
    background: white; border-radius: 13px; padding: 20px 18px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}}
.how-num {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 7px;
    background: #EFF6FF; color: #1D4ED8;
    font-size: 0.8rem; font-weight: 700; margin-bottom: 11px;
}}
.how-title {{ font-weight: 600; color: #1E293B; margin-bottom: 5px; font-size: 0.88rem; }}
.how-desc  {{ font-size: 0.78rem; color: #94A3B8; line-height: 1.6; }}

/* ── Tabs ── */
[data-baseweb="tab-list"] {{
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 11px !important;
    padding: 4px !important; gap: 2px !important;
    width: fit-content !important; margin-bottom: 18px !important;
}}
[data-baseweb="tab"] {{
    border-radius: 8px !important; font-weight: 500 !important;
    font-size: 0.82rem !important; padding: 7px 16px !important;
    color: #64748B !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: white !important; color: #1D4ED8 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.09) !important;
}}

/* ── Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    border: none !important; border-radius: 9px !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}}
[data-testid="stButton"] > button {{
    border-radius: 9px !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ border-radius: 11px !important; overflow: hidden !important; }}

/* ── Progress ── */
[data-testid="stProgressBar"] > div > div {{
    background: linear-gradient(90deg, #2563EB, #06B6D4) !important;
    border-radius: 999px !important;
}}
</style>

<!-- Bouton hamburger -->
<button id="av-toggle" title="Ouvrir / Fermer le menu" onclick="avToggle()">
  <span></span><span></span><span></span>
</button>
<script>
function avToggle() {{
    const selectors = [
        '[data-testid="stSidebarCollapseButton"] button',
        '[data-testid="collapsedControl"] button',
        'button[aria-label="Close sidebar"]',
        'button[aria-label="Open sidebar"]'
    ];
    for (const s of selectors) {{
        const b = window.parent.document.querySelector(s);
        if (b) {{ b.click(); return; }}
    }}
}}
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def conf_class(c):
    return "conf-high" if c >= 0.75 else ("conf-mid" if c >= 0.45 else "conf-low")

def conf_color(c):
    return "#22C55E" if c >= 0.75 else ("#F59E0B" if c >= 0.45 else "#EF4444")

def annotate_image(result, line_width=3, font_scale=0.65, target_width=1024):
    """Annotation OpenCV avec palette fixe par classe."""
    img = np.array(result.orig_img)  # BGR
    h, w = img.shape[:2]
    if w > target_width:
        s = target_width / w
        img = cv2.resize(img, (target_width, int(h * s)))
        sx, sy = s, s
    else:
        sx, sy = 1.0, 1.0

    np.random.seed(42)
    palette = {i: tuple(int(c) for c in np.random.randint(60, 220, 3).tolist())
               for i in range(80)}

    if result.boxes is None or len(result.boxes) == 0:
        return img[:, :, ::-1]

    for box in result.boxes:
        cid   = int(box.cls[0])
        conf  = float(box.conf[0])
        label = result.names[cid].replace("p_", "")
        color = palette[cid % 80]

        x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy))
                           for i, v in enumerate(box.xyxy[0].tolist())]
        lw = line_width
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

        # Coins décoratifs
        cs = min(18, (x2 - x1) // 5, (y2 - y1) // 5)
        cl = lw + 1
        for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img, (px, py), (px + dx*cs, py), color, cl)
            cv2.line(img, (px, py), (px, py + dy*cs), color, cl)

        # Label
        text = f"{label}  {conf*100:.0f}%"
        tk = max(1, lw - 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, tk)
        ty = y1 - 9 if y1 - th - 9 > 0 else y1 + th + 9
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, ty - th - 7), (x1 + tw + 10, ty + 4), color, -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.putText(img, text, (x1 + 5, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), tk, cv2.LINE_AA)

    return img[:, :, ::-1]  # BGR → RGB


def detection_table(boxes, names):
    rows = ""
    for box in sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True):
        cls  = names[int(box.cls[0])]
        conf = float(box.conf[0])
        cout = COUTS_MOYENS.get(cls, None)
        bar  = int(conf * 100)
        rows += f"""
        <tr>
          <td style="padding:10px 16px;font-weight:500;">{cls.replace('p_','')}</td>
          <td style="padding:10px 16px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{bar}%;background:{conf_color(conf)};"></div>
              </div>
              <span class="conf-badge {conf_class(conf)}">{conf*100:.1f}%</span>
            </div>
          </td>
          <td class="mono" style="padding:10px 16px;">{f"{cout:,} MAD" if cout else "—"}</td>
        </tr>"""
    return f"""<div style="overflow-x:auto;">
    <table class="det-table">
      <thead><tr>
        <th>Pièce</th><th>Confiance</th><th>Coût moy. estimé</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table></div>"""


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str) -> YOLO:
    return YOLO(path)

try:
    model    = load_model(MODEL_PATH)
    model_ok = True
except Exception as e:
    model_ok  = False
    model_err = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "analyses" not in st.session_state:
    st.session_state.analyses = []


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Branding + Logo ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sb-brand">
        {logo_html}
        <div class="sb-brand-name">AutoVision Cabek</div>
        <div class="sb-brand-sub">YOLOv11 · Détection IA</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Statut modèle ────────────────────────────────────────────────────────
    if model_ok:
        st.markdown("""
        <div class="status-pill">
          <div class="status-dot"></div>Modèle opérationnel
        </div>""", unsafe_allow_html=True)
    else:
        st.error("❌ Modèle introuvable")

    st.markdown('<div class="av-div"></div>', unsafe_allow_html=True)

    # ── Paramètres de détection ──────────────────────────────────────────────
    st.markdown("**Détection**")
    conf_threshold = st.slider(
        "Seuil de confiance", 0.0, 1.0, 0.35, 0.01,
        help="F1 optimal ~ 0.35–0.45 mesuré lors de l'évaluation."
    )
    iou_threshold = st.slider(
        "Seuil IoU (NMS)", 0.0, 1.0, 0.50, 0.05,
        help="Supprime les détections redondantes sur une même pièce."
    )

    st.markdown('<div class="av-div"></div>', unsafe_allow_html=True)

    # ── Paramètres d'affichage ───────────────────────────────────────────────
    st.markdown("**Affichage**")
    line_width   = st.slider("Épaisseur des boîtes", 1, 8, 3)
    font_scale   = st.slider("Taille du texte", 0.4, 1.4, 0.65, 0.05)
    target_width = st.select_slider(
        "Résolution (px)", options=[640, 800, 1024, 1280, 1600], value=1024
    )

    # ── Performances du modèle ───────────────────────────────────────────────
    

    st.markdown("""
    <div style="text-align:center;padding:8px 0 6px;">
      <div style="font-size:0.68rem;color:#334155;line-height:1.9;">
        <span style="color:#1E293B;">© 2026 — Tous droits réservés</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if not model_ok:
    st.error(f"❌ Impossible de charger le modèle `{MODEL_PATH}`. Vérifiez le chemin.")
    st.stop()

st.markdown("""
<div class="page-title">Inspection visuelle par IA</div>
<div class="page-sub">
  Déposez une ou plusieurs photos · le modèle localise et identifie chaque pièce de carrosserie
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📂  Glissez-déposez vos photos ici — ou cliquez pour parcourir",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAITEMENT
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_files:
    existing  = {a["name"] for a in st.session_state.analyses}
    new_files = [f for f in uploaded_files if f.name not in existing]

    if new_files:
        prog = st.progress(0.0, text="")
        for i, f in enumerate(new_files):
            prog.progress(
                i / len(new_files),
                text=f"⚙️ Traitement de **{f.name}** ({i+1}/{len(new_files)})…"
            )
            img  = Image.open(f).convert("RGB")
            t0   = time.time()
            res  = model.predict(img, conf=conf_threshold,
                                 iou=iou_threshold, verbose=False)[0]
            dt   = round(time.time() - t0, 3)
            cout = sum(COUTS_MOYENS.get(model.names[int(b.cls[0])], 0)
                       for b in res.boxes)
            st.session_state.analyses.append({
                "name":       f.name,
                "image":      img,
                "result":     res,
                "time":       dt,
                "n_objects":  len(res.boxes),
                "cout_total": cout,
            })
        prog.progress(1.0, text=f"✅ {len(new_files)} image(s) analysée(s) avec succès")
        time.sleep(0.8)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
analyses = st.session_state.analyses

if analyses:
    total_img    = len(analyses)
    total_pieces = sum(a["n_objects"] for a in analyses)
    total_time   = sum(a["time"]      for a in analyses)
    total_cout   = sum(a["cout_total"]for a in analyses)
    all_confs    = [float(b.conf[0]) for a in analyses for b in a["result"].boxes]
    avg_conf     = round(sum(all_confs) / len(all_confs) * 100, 1) if all_confs else 0

    # ── KPI ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="mc-label">Images analysées</div>
        <div class="mc-value">{total_img}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Pièces détectées</div>
        <div class="mc-value">{total_pieces}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Confiance moy.</div>
        <div class="mc-value">{avg_conf}<span class="mc-unit">%</span></div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Temps total</div>
        <div class="mc-value">{total_time:.2f}<span class="mc-unit"> s</span></div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Coût total estimé</div>
        <div class="mc-value" style="font-size:1.4rem;">
          {total_cout:,}<span class="mc-unit"> MAD</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔬 Analyse détaillée",
        "📊 Récapitulatif batch",
        "📥 Export rapport"
    ])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        c_sel, c_del = st.columns([5, 1])
        with c_sel:
            idx = st.selectbox(
                "Image à inspecter",
                range(len(analyses)),
                format_func=lambda i: (
                    f"{analyses[i]['name']}  ·  "
                    f"{analyses[i]['n_objects']} pièces  ·  "
                    f"{analyses[i]['time']} s"
                )
            )
        with c_del:
            st.write("")
            if st.button("🗑 Retirer", use_container_width=True):
                st.session_state.analyses.pop(idx)
                st.rerun()

        sel = analyses[idx]
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-lbl">Image source</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="img-frame">'
                f'<div class="img-frame-bar">'
                f'<span class="img-dot" style="background:#FF5F56;"></span>'
                f'<span class="img-dot" style="background:#FEBC2E;"></span>'
                f'<span class="img-dot" style="background:#28C840;"></span>'
                f'<span class="img-frame-title">{sel["name"]}</span>'
                f'</div>',
                unsafe_allow_html=True)
            st.image(sel["image"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-lbl">Détections YOLOv11</div>',
                        unsafe_allow_html=True)
            if sel["n_objects"] > 0:
                st.markdown(
                    f'<div class="img-frame">'
                    f'<div class="img-frame-bar">'
                    f'<span class="img-dot" style="background:#FF5F56;"></span>'
                    f'<span class="img-dot" style="background:#FEBC2E;"></span>'
                    f'<span class="img-dot" style="background:#28C840;"></span>'
                    f'<span class="img-frame-title">'
                    f'{sel["n_objects"]} objets · conf ≥ {conf_threshold:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True)
                annotated = annotate_image(
                    sel["result"], line_width, font_scale, target_width
                )
                st.image(annotated, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning(
                    "Aucune pièce détectée — "
                    "essayez de baisser le seuil de confiance dans la sidebar."
                )

        # Détails
        if sel["n_objects"] > 0:
            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-lbl">Composants identifiés</div>',
                        unsafe_allow_html=True)

            noms  = sorted({model.names[int(b.cls[0])] for b in sel["result"].boxes})
            chips = "".join(
                f'<span class="part-chip"><div class="chip-dot"></div>'
                f'{n.replace("p_","")}</span>'
                for n in noms
            )
            st.markdown(f'<div style="margin-bottom:18px;">{chips}</div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="section-lbl">Détail des détections</div>',
                        unsafe_allow_html=True)
            tbl = detection_table(sel["result"].boxes, model.names)
            st.markdown(
                f'<div class="av-card" style="padding:0;overflow:hidden;">{tbl}</div>',
                unsafe_allow_html=True
            )

            if sel["cout_total"] > 0:
                st.markdown(f"""
                <div class="cost-box">
                  <div>
                    <div class="cost-label">💡 Estimation du coût de réparation</div>
                    <div class="cost-sub">
                      Coûts moyens historiques par pièce détectée · valeur indicative
                    </div>
                  </div>
                  <div>
                    <span class="cost-value">{sel['cout_total']:,}</span>
                    <span class="cost-currency">MAD</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-lbl">Toutes les images de la session</div>',
                    unsafe_allow_html=True)
        rows = []
        for i, a in enumerate(analyses):
            cf = [float(b.conf[0]) for b in a["result"].boxes]
            rows.append({
                "#": i+1,
                "Fichier":              a["name"],
                "Pièces":              a["n_objects"],
                "Confiance moy. (%)":  round(sum(cf)/len(cf)*100, 1) if cf else 0,
                "Temps (s)":           a["time"],
                "Coût estimé (MAD)":   f"{a['cout_total']:,}",
                "Résolution":          f"{a['image'].width}×{a['image'].height}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="section-lbl" style="margin-top:22px;">'
            'Distribution des pièces détectées</div>',
            unsafe_allow_html=True
        )
        all_parts = [
            model.names[int(b.cls[0])].replace("p_", "")
            for a in analyses for b in a["result"].boxes
        ]
        if all_parts:
            df_p = (pd.Series(all_parts).value_counts()
                    .rename_axis("Pièce").reset_index(name="Détections"))
            st.bar_chart(df_p.set_index("Pièce"), height=300, color="#2563EB")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Réinitialiser toutes les analyses", type="secondary"):
            st.session_state.analyses = []
            st.rerun()

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        all_rows = []
        for a in analyses:
            for b in a["result"].boxes:
                cls    = model.names[int(b.cls[0])]
                conf   = float(b.conf[0])
                cout   = COUTS_MOYENS.get(cls, None)
                coords = [int(c) for c in b.xyxy[0].tolist()]
                all_rows.append({
                    "Image":              a["name"],
                    "Pièce":              cls,
                    "Confiance (%)":      round(conf * 100, 2),
                    "Coût estimé (MAD)":  cout if cout else "",
                    "X1": coords[0], "Y1": coords[1],
                    "X2": coords[2], "Y2": coords[3],
                })

        if all_rows:
            df_exp = pd.DataFrame(all_rows)
            st.markdown('<div class="section-lbl">Rapport complet de la session</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c_dl, c_inf = st.columns([2, 3])
            with c_dl:
                st.download_button(
                    label="📥 Télécharger le rapport CSV",
                    data=df_exp.to_csv(index=False).encode("utf-8"),
                    file_name=f"autovision_rapport_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )
            with c_inf:
                st.info(
                    f"**{len(all_rows)}** détections · "
                    f"**{total_img}** image(s) · "
                    f"Coût global estimé : **{total_cout:,} MAD**"
                )
        else:
            st.warning("Aucune détection à exporter pour cette session.")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT VIDE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="empty-wrap">
      <div class="empty-icon">🔬</div>
      <div class="empty-title">Aucune image importée</div>
      <div class="empty-sub">
        Utilisez le sélecteur ci-dessus pour importer une ou plusieurs photos de véhicule.<br>
        Le modèle localise automatiquement les pièces et génère une estimation de coût.
      </div>
    </div>

    <div class="how-steps">
      <div class="how-step">
        <div class="how-num">01</div>
        <div class="how-title">Importer les photos</div>
        <div class="how-desc">JPG ou PNG · sélection multiple supportée · traitées en lot automatiquement</div>
      </div>
      <div class="how-step">
        <div class="how-num">02</div>
        <div class="how-title">Analyse par le modèle</div>
        <div class="how-desc">YOLOv11 localise et identifie chaque pièce avec un score de confiance</div>
      </div>
      <div class="how-step">
        <div class="how-num">03</div>
        <div class="how-title">Résultats & export</div>
        <div class="how-desc">Consultez les détections, l'estimation de coût et exportez le rapport CSV</div>
      </div>
    </div>
    """, unsafe_allow_html=True)