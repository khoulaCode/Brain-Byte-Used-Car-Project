# app_autovaluer_bentley_robust_darkboxes_v2_no_range_no_why.py
# UI Redesign (olive theme, square cards, centered labels)
# This version removes: (1) Range line under price, (2) "Why this price?" expander (and its CSS)
import os, re, hashlib, warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="AutoValuer — Car Price (OMR)", page_icon="🚗", layout="wide")

# ===== Palette =====
INK_DARK = "#2B2F2C"; INK_SOFT = "#5B6360"; INK_MUTE = "#7D8681"
BG_LIGHT = "#F4F7F5"; PANEL = "#FFFFFF"; PANEL_LINE = "#D7E0DB"
ACCENT = "#6B8F71"; ACCENT_DK = "#4F7258"
INPUT_BG_DK = "#243d34"; INPUT_BORDER = "#365247"; INPUT_TEXT = "#F3F6F4"
OLIVE_DARK = "#132922"; OLIVE_FAINT = "#E9EFE9"

# Google fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ===== Base CSS =====
CSS = f"""
<style>
:root{{
  --ink:{INK_DARK}; --ink-soft:{INK_SOFT}; --ink-mute:{INK_MUTE};
  --bg:{BG_LIGHT}; --panel:{PANEL}; --panel-line:{PANEL_LINE};
  --accent:{ACCENT}; --accent-dk:{ACCENT_DK};
  --input-bg-dk:{INPUT_BG_DK}; --input-border:{INPUT_BORDER}; --input-text:{INPUT_TEXT};
  --olive-dark:{OLIVE_DARK}; --olive-faint:{OLIVE_FAINT};
}}
html, body, .stApp * {{ font-family:'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; letter-spacing:.2px; }}
h1,h2,h3,.hero-title{{ font-family:'Manrope','Inter',sans-serif; }}

/* App background + container */
.stApp{{ background:var(--bg); color:var(--ink); }}
.block-container{{ background:linear-gradient(180deg, rgba(19,41,34,0.08) 0%, var(--olive-faint) 100%); border-radius:0; padding-top:0; }}

/* Cards */
.section{{ margin:24px 0; }}
.card{{ background:var(--panel); border:1px solid var(--panel-line); border-radius:16px; padding:20px; box-shadow:0 8px 24px rgba(0,0,0,.08); color:var(--ink); }}
hr.sep{{ border:none; border-top:1px solid var(--panel-line); margin:12px 0; }}

/* ---------- HERO ---------- */
.hero-wrap{{ position:relative; left:50%; right:50%; margin-left:-50vw; margin-right:-50vw; width:100vw; }}
.hero-aspect{{ position:relative; width:100%; padding-top:56.25%; background:#000; }}
.hero-aspect iframe{{ position:absolute; inset:0; width:100%; height:100%; border:none; }}
.hero-overlay{{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center; pointer-events:none; background:linear-gradient(180deg, rgba(0,0,0,.15), rgba(0,0,0,.55)); }}
.hero-title{{ color:#fff; font-weight:600; line-height:1.06; font-size:clamp(36px,4.2vw,52px); letter-spacing:.25px; margin:0 16px 16px; text-transform:uppercase; }}
.hero-sub{{ color:#F5F7FA; font-weight:400; font-size:clamp(16px,1.6vw,20px); margin:0 16px; }}
.hero-sub .strong{{ color:#fff; font-weight:700; }}
.hero-stack{{ display:flex; flex-direction:column; gap:8px; align-items:center; }}

/* hero buttons */
a.btn:link, a.btn:visited, a.btn:active{{ color:inherit!important; text-decoration:none!important; }}
.btn{{ display:inline-flex; align-items:center; gap:10px; padding:14px 22px; border-radius:14px; font-weight:600; letter-spacing:.6px; text-transform:uppercase; font-size:15px; transition:all .18s ease; }}
.btn svg{{ width:18px; height:18px; }}
.btn-filled{{ background:var(--accent); color:#0b0d11!important; border:none; }}
.btn-filled:hover{{ background:var(--accent-dk); transform:translateY(-1px); }}
.btn-outline{{ background:transparent; color:#FFFFFF!important; border:1px solid rgba(255,255,255,.6); }}
.btn-outline:hover{{ border-color:#fff; transform:translateY(-1px); }}
.actions{{ display:flex; gap:14px; margin-top:22px; justify-content:center; }}

/* ---------- ABOUT ---------- */
.about-wrap{{ position:relative; left:50%; right:50%; margin-left:-50vw; margin-right:-50vw; width:100vw; }}
.about-aspect{{ position:relative; width:100%; padding-top:56.25%; background:#000; overflow:hidden; }}
.about-aspect iframe{{ position:absolute; inset:-8% 0 0 0; width:100%; height:116%; border:none; pointer-events:none; }}
.about-overlay{{ position:absolute; inset:0; background:linear-gradient(180deg, rgba(0,0,0,.10), rgba(0,0,0,.35)); }}
.about-deco{{ position:absolute; left:max(40px,5%); top:calc(50% - 280px); width:520px; height:520px; border-radius:0;
  background:linear-gradient(135deg, rgba(165,186,172,.65), rgba(119,150,127,.35)); }}
.about-card{{ position:absolute; left:max(72px,7%); top:calc(50% - 260px); width:560px; height:560px; border-radius:0; padding:34px;
  background:linear-gradient(165deg, rgba(45,64,52,.92), rgba(79,114,88,.80)); border:1px solid rgba(79,114,88,.45);
  box-shadow:0 30px 80px rgba(0,0,0,.35); backdrop-filter:blur(3px);
  display:flex; flex-direction:column; align-items:center; justify-content:flex-start; text-align:center; gap:16px; }}
.about-card h2{{ color:#fff; margin:6px 0 4px 0; font-size:30px; font-weight:500; letter-spacing:.3px; text-transform:uppercase; }}
.about-card p{{ color:#F1F5F3; margin:0; max-width:520px; font-size:17px; line-height:1.85; font-weight:400; }}
.about-card .actions{{ width:100%; display:flex; justify-content:center; margin-top:10px; }}
.btn-invert{{ display:inline-flex; align-items:center; gap:10px; background:#fff; color:var(--accent-dk)!important; border:1.5px solid #fff; border-radius:12px; padding:14px 24px; font-weight:600; letter-spacing:.6px; text-transform:uppercase; font-size:15px; transition:all .18s ease; }}
.btn-invert svg{{ width:18px; height:18px; }}
.btn-invert:hover{{ background:transparent; color:#fff!important; border-color:#fff; transform:translateY(-1px); }}

@media (max-width:1100px){{ .about-deco{{ left:4%; width:440px; height:440px; top:calc(50% - 230px); }} .about-card{{ left:6%; width:490px; height:490px; top:calc(50% - 220px); }} }}
@media (max-width:860px){{ .about-deco{{ display:none; }} .about-card{{ position:static; width:auto; height:auto; margin:14px; }} }}

/* ---------- Input section ---------- */
.light-form{{ position:relative; border-radius:16px; padding:22px; background:var(--panel); color:var(--ink); border:1px solid var(--panel-line); box-shadow:0 8px 24px rgba(0,0,0,.08); }}
.light-form [data-testid="stWidgetLabel"] > label,
.light-form .stSelectbox label,.light-form .stNumberInput label,.light-form .stTextInput label,.light-form .stRadio label{{ color:var(--ink); opacity:1!important; font-weight:600; text-align:center; width:100%; margin-bottom:6px; letter-spacing:.2px; }}
.light-form .stSelectbox > div, .light-form .stNumberInput > div, .light-form .stTextInput > div{{ max-width:520px; margin:0 auto 14px; }}

/* Predict button */
.light-form .stButton > button{{ background:var(--accent); color:#0b0d11; font-weight:800; border:0; border-radius:12px; padding:12px 18px; font-size:16px; display:inline-flex; align-items:center; gap:10px; position:relative; }}
.light-form .stButton > button::before{{ content:""; width:16px; height:16px; background:#0b0d11; -webkit-mask-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M10 5l7 7-7 7-1.5-1.5L13 12 8.5 6.5 10 5Z'/></svg>"); mask-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M10 5l7 7-7 7-1.5-1.5L13 12 8.5 6.5 10 5Z'/></svg>"); mask-repeat:no-repeat; mask-size:contain; }}
.light-form .stButton > button > div{{ display:none; }}

.price-text{{ font-size:36px; font-weight:900; color:var(--ink); }}
.badge{{ display:inline-block; padding:2px 10px; border-radius:999px; background:var(--accent); color:#0b0d11; font-size:12px; font-weight:800; }}
.small{{ color:var(--ink-mute); font-size:13px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ===== Strong OVERRIDE for input boxes (olive a bit lighter) =====
DARK_OLIVE_BG = "#1b352b"
DARK_OLIVE_BD = "#365247"
INPUT_TXT     = "#F3F6F4"

OVERRIDE = f"""
<style>
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {{
  background:{DARK_OLIVE_BG} !important;
  color:{INPUT_TXT} !important;
  border:1px solid {DARK_OLIVE_BD} !important;
  border-radius:12px !important;
}}
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stSelectbox"] > div,
div[data-baseweb="select"] > div {{
  background:{DARK_OLIVE_BG} !important;
  color:{INPUT_TXT} !important;
  border:1px solid {DARK_OLIVE_BD} !important;
  border-radius:12px !important;
}}
div[data-testid="stSelectbox"] [role="combobox"] *,
div[data-baseweb="select"] * {{
  color:{INPUT_TXT} !important;
  fill:{INPUT_TXT} !important;
}}
ul[role="listbox"],
div[data-baseweb="menu"] {{
  background:{DARK_OLIVE_BG} !important;
  color:{INPUT_TXT} !important;
  border:1px solid {DARK_OLIVE_BD} !important;
  border-radius:12px !important;
}}
div[data-testid="stTextInput"] > div > div,
div[data-testid="stNumberInput"] > div > div,
div[data-testid="stSelectbox"] > div > div {{
  background:transparent !important;
}}
</style>
"""
st.markdown(OVERRIDE, unsafe_allow_html=True)

# ---------- HERO ----------
video_hero = "dzqYxMuG7W0"
embed_hero = f"https://www.youtube.com/embed/{video_hero}?autoplay=1&mute=1&controls=0&rel=0&showinfo=0&modestbranding=1&playsinline=1&loop=1&playlist={video_hero}"
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-aspect">
    <iframe src="{embed_hero}" allow="autoplay; encrypted-media"></iframe>
    <div class="hero-overlay">
      <div class="hero-stack">
        <div class="hero-title">AUTOVALUER</div>
        <div class="hero-sub">Get instant, realistic car price predictions</div>
        <div class="hero-sub"><span class="strong">Powered by real market data</span></div>
        <div class="actions" style="pointer-events:auto;">
          <a class="btn btn-filled" href="#predict" title="Explore">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 12l18-9-9 18-2-7-7-2z"/></svg>
            Explore
          </a>
          <a class="btn btn-outline" href="#learn" title="Learn more">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a10 10 0 100 20 10 10 0 000-20zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
            Learn
          </a>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- ABOUT ----------
video_about = "r_iLK8dO7eE"
embed_about = f"https://www.youtube.com/embed/{video_about}?autoplay=1&mute=1&controls=0&rel=0&showinfo=0&modestbranding=1&playsinline=1&loop=1&playlist={video_about}"
st.markdown(f"""
<div class="about-wrap">
  <div class="about-aspect">
    <iframe src="{embed_about}" allow="autoplay; encrypted-media"></iframe>
    <div class="about-overlay"></div>
    <div class="about-deco" aria-hidden="true"></div>
    <div class="about-card" role="region" aria-label="About AutoValuer">
      <h2>ABOUT AUTOVALUER</h2>
      <p>
        AutoValuer delivers realistic price estimates backed by market data.
        We blend machine learning with mileage-aware comparables and brand-tier heuristics
        to avoid outliers, so the number you see aligns with how buyers actually shop.
      </p>
      <div class="actions" style="pointer-events:auto;">
        <a class="btn-invert" href="#predict" title="Explore Predictions">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 12l18-9-9 18-2-7-7-2z"/></svg>
          Explore Predictions
        </a>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Section Title (Predictions) ----------
st.markdown("""
<div style="text-align:center; margin:50px 0 20px 0;">
  <h2 style="color:#4F7258; font-size:34px; font-weight:600; margin-bottom:6px;">Car Price Predictions</h2>
  <p style="color:#6B8F71; font-size:17px; margin:0;">Enter Your Car Details Below To Predict The Price.</p>
</div>
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
DEFAULT_PATH = r"C:\Users\bbuser\Downloads\data_new_one.csv"
st.markdown('<div class="section card" id="learn">', unsafe_allow_html=True)
st.subheader("Data Source")
choice = st.radio("Load CSV:", ["Use default path", "Enter path", "Upload"], horizontal=True)
df_raw, data_label = None, None
if choice == "Use default path":
    data_label = DEFAULT_PATH
    if os.path.exists(DEFAULT_PATH):
        df_raw = pd.read_csv(DEFAULT_PATH)
        st.success(f"Loaded: {DEFAULT_PATH}")
    else:
        st.error(f"Not found: {DEFAULT_PATH}")
elif choice == "Enter path":
    p = st.text_input("Full CSV path", value=DEFAULT_PATH)
    if p:
        if os.path.exists(p):
            try:
                df_raw = pd.read_csv(p); data_label = p; st.success(f"Loaded: {p}")
            except Exception as e:
                st.error(f"Failed reading CSV: {e}")
        else:
            st.warning("Path does not exist.")
else:
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_raw = pd.read_csv(up); data_label = getattr(up, "name", "uploaded.csv")
            st.success(f"Loaded uploaded file: {data_label}")
        except Exception as e:
            st.error(f"Failed reading CSV: {e}")
st.markdown('</div>', unsafe_allow_html=True)
if df_raw is None: st.stop()

# ---------- PREP ----------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in out.columns]
    return out

df = normalize_cols(df_raw)
TARGET_CANDS = ["price","target_price","sale_price","listed_price"]
target_col = next((c for c in TARGET_CANDS if c in df.columns), None)
if target_col is None:
    st.error("No price column found. Expected one of: price/target_price/sale_price/listed_price"); st.stop()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df[df[target_col].notna()].copy()

WANTS = {
    "brand": ["brand","make","brand_from_features"],
    "model": ["model","car_model"],
    "body_type": ["body_type","body","car_body"],
    "mileage": ["mileage","odo","odometer"],
    "year": ["year","model_year"],
    "transmission": ["transmission","gearbox"],
    "color": ["color","colour","exterior_color"],
    "seats": ["seats","num_seats"],
    "interior": ["interior","interior_color","trim"],
    "warranty": ["warranty"],
    "owners": ["owners","num_owners","owner_count"],
}
def resolve_col(keys):
    for k in keys:
        if k in df.columns: return k
    return None
RES = {k: resolve_col(v) for k,v in WANTS.items()}

TX_MAP = {"automatic":"Automatic","manual":"Manual","cvt":"CVT","tiptronic":"Tiptronic","semi-automatic":"Semi-automatic","other":"Other",
          0:"Automatic",1:"Manual",2:"CVT",3:"Tiptronic",4:"Semi-automatic",9:"Other"}
INV_TX_MAP = {v:k for k,v in TX_MAP.items()}
LUXURY = {"Bentley","Rolls Royce","Rolls-Royce","Ferrari","Lamborghini","Porsche","Aston Martin","Maserati","McLaren","Maybach","Bugatti","Lotus"}
PREMIUM = {"BMW","Mercedes","Mercedes-Benz","Audi","Lexus","Infiniti","Genesis","Jaguar","Land Rover","Range Rover","Cadillac","Lincoln","Volvo","Acura","Alfa Romeo","Tesla"}

bcol = RES.get("brand")
if bcol:
    _tmp = df[[bcol, target_col]].copy()
    _tmp[target_col] = pd.to_numeric(_tmp[target_col], errors="coerce")
    _tmp = _tmp[_tmp[target_col].notna()]
    BRAND_STATS = (_tmp.groupby(bcol)[target_col]
        .agg(count="count", median="median",
             p25=lambda s: s.quantile(0.25),
             p75=lambda s: s.quantile(0.75),
             p90=lambda s: s.quantile(0.90), max="max").to_dict("index"))
else:
    BRAND_STATS = {}

X_cols = [c for c in df.columns if c != target_col]
TOO_LONG = [c for c in X_cols if df[c].dtype == object and df[c].astype(str).str.len().mean() > 60]
X_cols = [c for c in X_cols if c not in TOO_LONG]

if bcol:
    def brand_tier(x: str) -> str:
        s = str(x).strip()
        if s in LUXURY: return "luxury"
        if s in PREMIUM: return "premium"
        return "economy"
    df["brand_tier"] = df[bcol].apply(brand_tier)
    if "brand_tier" not in X_cols: X_cols.append("brand_tier")

X = df[X_cols].copy()
y = df[target_col].astype(float).copy()
for c in X.columns:
    if pd.api.types.is_bool_dtype(X[c]): X[c] = X[c].astype(int)
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].astype(str).replace(["nan","None","NaN",""], "Unknown")

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]),
    ],
    remainder="drop", verbose_feature_names_out=False,
)

def base_model(name: str):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=700, max_depth=None, n_jobs=-1, random_state=42)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=900, learning_rate=0.05, max_depth=3, random_state=42)
    if name == "LightGBM" and _HAS_LGBM:
        return LGBMRegressor(n_estimators=1200, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    return RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)

st.markdown('<div class="section card">', unsafe_allow_html=True)
st.subheader("Data Source & Model Settings")
model_name = st.selectbox("Choose model", ["Random Forest", "Gradient Boosting"] + (["LightGBM"] if _HAS_LGBM else []), index=0)
fast_mode = st.checkbox("⚡ Fast mode (subset 60%)", value=False)
test_size = st.slider("Validation size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
st.markdown('</div>', unsafe_allow_html=True)

def data_hash(df_: pd.DataFrame) -> str:
    s = pd.util.hash_pandas_object(df_, index=True).values
    return hashlib.sha256(s).hexdigest()
cache_key = (data_hash(df[X_cols]), model_name, fast_mode, test_size)

@st.cache_resource(show_spinner=True)
def fit_cached(X_in, y_in, key):
    if fast_mode and len(X_in) > 2500:
        X_in = X_in.sample(frac=0.6, random_state=42)
        y_in = y_in.loc[X_in.index]
    X_tr, X_va, y_tr, y_va = train_test_split(X_in, y_in, test_size=test_size, random_state=42)
    reg = TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", base_model(model_name))]),
        func=np.log1p, inverse_func=np.expm1
    )
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_va)
    r2 = float(r2_score(y_va, preds))
    rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
    reg_all = TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", base_model(model_name))]),
        func=np.log1p, inverse_func=np.expm1
    )
    reg_all.fit(X_in, y_in)
    return reg_all, r2, rmse

with st.spinner("Training model (log-target)..."):
    reg_all, r2_val, rmse_val = fit_cached(X, y, cache_key)

# ---------- INPUTS ----------
def safe_year_input(df_scope, ycol_name, fallback=2015, label="Select Year"):
    if ycol_name and ycol_name in df_scope and pd.api.types.is_numeric_dtype(df_scope[ycol_name]):
        y_vals = pd.to_numeric(df_scope[ycol_name], errors="coerce").dropna().astype(int)
        if len(y_vals) > 0:
            y_min, y_max = int(y_vals.min()), int(y_vals.max())
            if y_min < y_max:
                return st.slider(label, min_value=y_min, max_value=y_max, value=int(np.median(y_vals)))
            else:
                return st.number_input(label, value=y_min, step=1)
    return st.number_input(label, value=fallback, step=1)

def safe_mileage_input(df_scope, mcol_name, fallback=60000.0, label="Enter Mileage"):
    if mcol_name and mcol_name in df_scope:
        m_vals = pd.to_numeric(df_scope[mcol_name], errors="coerce").dropna().astype(float)
        if len(m_vals) > 0:
            lo, hi = float(m_vals.min()), float(m_vals.max())
            if lo < hi:
                mid = float(np.median(m_vals)); mid = float(np.clip(mid, lo, hi))
                return st.slider(label, min_value=lo, max_value=hi, value=mid, step=100.0)
            else:
                return st.number_input(label, value=lo, step=500.0)
    return st.number_input(label, value=fallback, step=500.0)

def format_omr(x: float) -> str:
    try: return f"OMR {x:,.0f}"
    except Exception: return f"OMR {x}"

st.markdown('<div class="section light-form" id="predict">', unsafe_allow_html=True)
st.subheader("Please Enter Your Car Details")
st.markdown('<hr class="sep" />', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    bcol2 = RES.get("brand")
    if bcol2 and bcol2 in df:
        brands = sorted(df[bcol2].astype(str).fillna("Unknown").unique().tolist())
        brand_val = st.selectbox("Choose Your Car Brand", brands, index=0)
        df_b = df[df[bcol2].astype(str) == str(brand_val)].copy()
    else:
        brand_val, df_b = None, df.copy()

with col2:
    mcol_model = RES.get("model")
    if mcol_model and mcol_model in df_b:
        models = sorted(df_b[mcol_model].astype(str).fillna("Unknown").unique().tolist())
        model_val = st.selectbox("Choose Your Car Model", models, index=0)
        df_bm = df_b[df_b[mcol_model].astype(str) == str(model_val)].copy()
    else:
        model_val, df_bm = None, df_b.copy()

with col1:
    btcol = RES.get("body_type")
    if btcol and btcol in df_bm:
        body_types = sorted(df_bm[btcol].astype(str).fillna("Unknown").unique().tolist())
        body_type_val = st.selectbox("Choose Body Type", body_types, index=0)
    else:
        body_type_val = None

with col2:
    ycol = RES.get("year")

df_scope_year = df_bm if ('df_bm' in locals() and len(df_bm)) else (df_b if ('df_b' in locals() and len(df_b)) else df)
year_val = safe_year_input(df_scope_year, ycol, fallback=2015, label="Select Year")

with col1:
    mcol_mi = RES.get("mileage")

if 'df_b' in locals():
    df_scope_mileage = df_bm if ('df_bm' in locals() and len(df_bm)) else df_b
else:
    df_scope_mileage = df_bm if ('df_bm' in locals() and len(df_bm)) else df

mileage_val = safe_mileage_input(df_scope_mileage, mcol_mi, fallback=60000.0, label="Enter Mileage")

with col2:
    tcol = RES.get("transmission")
    if tcol and tcol in df:
        raw_tx = df[tcol].dropna().astype(str).str.lower().unique().tolist()
        opts = sorted({TX_MAP.get(v, v.title()) for v in raw_tx})
        if "Automatic" not in opts: opts.insert(0, "Automatic")
        if "Manual" not in opts:  opts.insert(1, "Manual")
        transmission_val_display = st.selectbox("Select Transmission", options=opts, index=0)
    else:
        transmission_val_display = st.selectbox("Select Transmission", options=["Automatic","Manual"], index=0)

with col1:
    color_val = st.text_input("Pick Exterior Color", value="Unknown") if not RES.get("color") else \
        st.selectbox("Pick Exterior Color", sorted(df[RES["color"]].astype(str).replace(["nan","None","NaN"],"Unknown").unique().tolist()))
    seats_val = st.number_input("Enter Seats", value=5, step=1)
    warranty_val = st.selectbox("Select Warranty", sorted(df[RES["warranty"]].astype(str).replace(["nan","None","NaN"],"Unknown").unique().tolist())) if RES.get("warranty") else None

with col2:
    interior_val = st.selectbox("Select Interior", sorted(df[RES["interior"]].astype(str).replace(["nan","None","NaN"],"Unknown").unique().tolist())) if RES.get("interior") else None
    owners_val   = st.number_input("Enter Owners", value=1, step=1)

predict_clicked = st.button("Predict Price", key="predict")
st.markdown('</div>', unsafe_allow_html=True)

if bcol and 'brand_val' in locals() and brand_val in BRAND_STATS:
    s = BRAND_STATS[brand_val]
    if s["count"] < 5:
        st.info(f"Only {int(s['count'])} listings for {brand_val} — using wider comps & stronger floors.")

# ---------- Helpers ----------
def sanitize_for_pipeline(tmp_df: pd.DataFrame) -> pd.DataFrame:
    for c in X_cols:
        if c not in tmp_df.columns: tmp_df[c] = np.nan
    NUMERIC_COLS_FIXED = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    CAT_COLS_FIXED     = [c for c in X.columns if c not in NUMERIC_COLS_FIXED]
    for c in NUMERIC_COLS_FIXED:
        tmp_df[c] = pd.to_numeric(tmp_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fill_val = float(pd.to_numeric(X[c], errors="coerce").median())
        tmp_df[c] = tmp_df[c].fillna(fill_val).astype(float)
    for c in CAT_COLS_FIXED:
        tmp_df[c] = tmp_df[c].astype(str)
        bad = tmp_df[c].isin(["", "nan", "None", "NaN"])
        if bad.any(): tmp_df.loc[bad, c] = "Unknown"
    if bcol and "brand_tier" in X_cols and "brand_tier" not in tmp_df.columns:
        def brand_tier_infer(s):
            s = str(s).strip()
            if s in LUXURY: return "luxury"
            if s in PREMIUM: return "premium"
            return "economy"
        tmp_df["brand_tier"] = tmp_df.get(bcol, pd.Series(["Unknown"])).apply(brand_tier_infer)
    return tmp_df[X_cols].copy()

def build_input_row():
    row = {c: np.nan for c in X_cols}
    def put(col_key, value):
        c = RES.get(col_key)
        if c is not None: row[c] = value
    put("brand", brand_val); put("model", model_val); put("body_type", body_type_val)
    put("year", year_val); put("mileage", mileage_val); put("color", color_val)
    put("seats", seats_val); put("interior", interior_val); put("warranty", warranty_val); put("owners", owners_val)
    tcol_local = RES.get("transmission")
    if tcol_local is not None: row[tcol_local] = INV_TX_MAP.get(transmission_val_display, transmission_val_display)
    if bcol:
        s = str(brand_val).strip()
        row["brand_tier"] = "luxury" if s in LUXURY else ("premium" if s in PREMIUM else "economy")
    return pd.DataFrame([row])

def comparable_subset(df_all: pd.DataFrame, specs: dict, RES: dict, year_window=2, target_n=12):
    sub = df_all.copy()
    for key in ["brand","model","body_type","transmission","color"]:
        c = RES.get(key)
        if c and specs.get(c) is not None and c in sub:
            sub = sub[sub[c].astype(str) == str(specs[c])]
    cy = RES.get("year")
    if cy and specs.get(cy) is not None and cy in sub:
        y0 = pd.to_numeric(specs[cy], errors="coerce")
        if pd.notna(y0):
            sub = sub[(pd.to_numeric(sub[cy], errors="coerce") >= y0 - year_window) & (pd.to_numeric(sub[cy], errors="coerce") <= y0 + year_window)]
    sub = sub[pd.to_numeric(sub[target_col], errors="coerce").notna()]
    if len(sub) < target_n and RES.get("model"):
        sub = df_all[(df_all[RES["brand"]].astype(str) == str(specs.get(RES["brand"], ""))) & (df_all[RES["model"]].astype(str) == str(specs.get(RES["model"], "")))]
    if len(sub) < target_n:
        sub = df_all[df_all[RES["brand"]].astype(str) == str(specs.get(RES["brand"], ""))]
    if len(sub) < target_n and cy:
        y0 = pd.to_numeric(specs.get(cy), errors="coerce")
        if pd.notna(y0):
            sub = df_all[(df_all[RES["brand"]].astype(str) == str(specs.get(RES["brand"], ""))) & (pd.to_numeric(df_all[cy], errors="coerce").between(y0-5, y0+5))]
    if len(sub) < max(5, target_n//2):
        sub = df_all[df_all[RES["brand"]].astype(str) == str(specs.get(RES["brand"], ""))]
    return sub

def anchor_price(raw_pred: float, specs_df_row: pd.DataFrame, df_train: pd.DataFrame, RES: dict):
    specs = specs_df_row.iloc[0].to_dict()
    brand = str(specs.get(RES.get("brand",""), "Unknown")).strip()
    is_lux = brand in LUXURY
    comps = comparable_subset(df_train, specs, RES, year_window=2, target_n=15 if is_lux else 12)
    n = len(comps)
    if n == 0: return raw_pred, None
    prices = pd.to_numeric(comps[target_col], errors="coerce").dropna()
    med = float(prices.median()); q05, q95 = float(prices.quantile(0.05)), float(prices.quantile(0.95))
    iqr = float(prices.quantile(0.75) - prices.quantile(0.25))
    if n >= 80: w = 0.70
    elif n >= 40: w = 0.60
    elif n >= 15: w = 0.50
    elif n >= 8:  w = 0.40
    else:        w = 0.30
    if is_lux: w = max(0.25, w - 0.20)
    blended = w * med + (1 - w) * raw_pred
    band_low = max(0.0, q05 - 0.10 * max(1.0, iqr)); band_high = q95 + 0.10 * max(1.0, iqr)
    final = float(np.clip(blended, band_low, band_high))
    info = {"n_comps": int(n), "median": med, "q05": q05, "q95": q95, "band_low": band_low, "band_high": band_high, "weight": w}
    return final, info

def apply_business_rules(brand: str, year: float, price: float):
    b = str(brand).strip(); y = float(year) if year is not None else 2015
    tier = "economy"
    if b in LUXURY: tier = "luxury"
    elif b in PREMIUM: tier = "premium"
    base_floor = 600.0
    if tier == "premium": base_floor = 3000.0 if y < 2010 else 4500.0
    elif tier == "luxury":
        if y >= 2020: base_floor = 35000.0
        elif y >= 2015: base_floor = 25000.0
        else: base_floor = 12000.0
    s = BRAND_STATS.get(b, None); extra_floor = 0.0
    if s:
        if pd.notna(s["median"]): extra_floor = max(extra_floor, 0.60 * float(s["median"]))
        if int(s["count"]) < 5 and pd.notna(s["median"]): extra_floor = max(extra_floor, 0.75 * float(s["median"]))
        if "p75" in s and pd.notna(s["p75"]): extra_floor = max(extra_floor, 0.40 * float(s["p75"]))
    floor_val = max(base_floor, extra_floor); max_p = None
    if tier == "economy" and y < 2008: max_p = 3000.0
    if price < floor_val: price = floor_val
    if max_p is not None and price > max_p: price = max_p
    return price, {"tier": tier, "min": floor_val, "max": max_p}

# ---------- Predict ----------
if predict_clicked:
    try:
        x_new_raw = build_input_row()
        x_new = sanitize_for_pipeline(x_new_raw.copy())
        raw_pred = float(reg_all.predict(x_new)[0])
        anchored_pred, ainfo = anchor_price(raw_pred, x_new_raw, df, RES)
        brand_for_rules = x_new_raw[RES["brand"]].iloc[0] if RES.get("brand") else "Unknown"
        year_for_rules = x_new_raw[RES["year"]].iloc[0] if RES.get("year") else 2015
        ruled_pred, rinfo = apply_business_rules(brand_for_rules, year_for_rules, anchored_pred)

        st.markdown(f"""
        <div class="card" style="text-align:center;">
          <div class="badge">{model_name} · log-target · anchored · rules</div>
          <div class="price-text" style="margin-top:6px;">{format_omr(ruled_pred)}</div>
          <hr class="sep" />
        </div>
        """, unsafe_allow_html=True)

        # NOTE: "Why this price?" expander removed
    except Exception as e:
        st.error(f"Prediction failed: {e}")
