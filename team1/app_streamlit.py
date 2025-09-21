import os, joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("Car Price Predictor 🚗")
st.caption("Upload your trained Pipeline model (joblib). Optional: upload catalog.csv to populate dropdowns. Only Model filters by Brand.")

st.sidebar.header("📦 Upload")
up_model = st.sidebar.file_uploader("Model file (.joblib / .pkl)", type=["joblib","pkl"])
up_catalog = st.sidebar.file_uploader("Optional: catalog.csv", type=["csv"])


default_model_path = "car_price_pipeline.joblib"
default_catalog_path = "catalog.csv"
if os.path.exists(default_model_path) and not up_model:
    up_model = open(default_model_path, "rb")
if os.path.exists(default_catalog_path) and not up_catalog:
    up_catalog = open(default_catalog_path, "rb")

def load_model(f):
    return joblib.load(f)

@st.cache_data
def load_catalog(f):
    df = pd.read_csv(f)

    rename_map = {
        "make":"brand","maker":"brand","manufacturer":"brand",
        "model_name":"model","colour":"color","transmission":"transmission_type",
        "doors":"number_of_doors"
    }
    df = df.rename(columns={c: c.lower() for c in df.columns}).rename(columns=rename_map)

    keep = ["brand","model","year","color","transmission_type",
            "number_of_doors","source","kilometers","price"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    if "year" in df: df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "kilometers" in df: df["kilometers"] = pd.to_numeric(df["kilometers"], errors="coerce")
    for c in ["brand","model","color","transmission_type","number_of_doors","source"]:
        if c in df: df[c] = df[c].astype(str).str.strip()
    return df.dropna(subset=[c for c in ["brand","model"] if c in df.columns])

catalog_df = load_catalog(up_catalog) if up_catalog else None
if up_catalog:
    st.sidebar.success(f"Catalog loaded: {len(catalog_df):,} rows")
else:
    st.sidebar.info("No catalog uploaded. Text inputs will be shown.")


if not up_model:
    st.info("Please upload your trained Pipeline model to continue.")
    st.stop()

try:
    model = load_model(up_model)
except ModuleNotFoundError as e:
    missing = str(e).split("'")[-2]
    st.error(f"Missing library **{missing}**. Install it and restart (e.g., `pip install {missing}`).")
    st.stop()

if not isinstance(model, Pipeline):
    st.warning("This model is not a scikit-learn Pipeline. Export as a Pipeline with preprocessing for best results.")


def expected_raw_columns(m):
    cols = []
    try:
        ct = dict(m.named_steps).get("pre", None)
        if ct is not None and hasattr(ct, "transformers_"):
            for name, trans, c in ct.transformers_:
                if name in ("num", "cat"):
                    if not isinstance(c, (list, tuple, np.ndarray)):
                        c = [c]
                    cols.extend([str(x) for x in c])
    except Exception:
        pass

    return list(dict.fromkeys(cols))

EXPECTED_COLS = expected_raw_columns(model)


def reset_downstream(changed_key, graph):
    for child in graph.get(changed_key, []):
        if child in st.session_state:
            st.session_state.pop(child)
        reset_downstream(child, graph)


DEPENDENCY_GRAPH = {
    "brand": ["model"],
}

def select_with_reset(label, key, options):
    sentinel = f"— Select {label} —"
    opts = [sentinel] + list(options)
    def _on_change():
        reset_downstream(key, DEPENDENCY_GRAPH)
    return st.selectbox(label, opts, index=0, key=key, on_change=_on_change)

def uniq(col):
    if catalog_df is None or col not in catalog_df.columns: return []
    return sorted(catalog_df[col].dropna().astype(str).unique())

def build_input_df(raw: dict) -> pd.DataFrame:
    row = {}
    for k, v in raw.items():
        if v in (None, "") or (isinstance(v, str) and v.startswith("— ")):
            row[k] = None
        elif k == "year":
            try: row[k] = int(v)
            except: row[k] = None
        elif k == "kilometers":
            try: row[k] = float(v)
            except: row[k] = None
        else:
            row[k] = v

    for col in EXPECTED_COLS:
        if col not in row:
            row[col] = None
    return pd.DataFrame([row])


st.subheader("Select car details")
inputs = {}

if catalog_df is not None and "brand" in catalog_df.columns:

    brand_opts = uniq("brand")
    select_with_reset("Brand", "brand", brand_opts)


    if st.session_state.get("brand") and not str(st.session_state["brand"]).startswith("—"):
        model_opts = sorted(
            catalog_df.loc[catalog_df["brand"] == st.session_state["brand"], "model"]
            .dropna().astype(str).unique()
        )
    else:
        model_opts = uniq("model")
    select_with_reset("Model", "model", model_opts)

    if "year" in (catalog_df.columns):
        year_vals = catalog_df["year"].dropna().unique()
        year_opts = [str(int(y)) for y in sorted(year_vals)]
    else:
        year_opts = []
    select_with_reset("Year", "year", year_opts)
    select_with_reset("Color", "color", uniq("color"))
    select_with_reset("Transmission", "transmission_type", uniq("transmission_type"))
    select_with_reset("Doors", "number_of_doors", uniq("number_of_doors"))
    select_with_reset("Source", "source", uniq("source"))

    default_km = int(catalog_df["kilometers"].median()) if "kilometers" in catalog_df.columns and catalog_df["kilometers"].notna().any() else 50000
    km = st.number_input("Kilometers", min_value=0, step=1000, value=default_km)

    inputs.update({
        "brand": st.session_state.get("brand"),
        "model": st.session_state.get("model"),
        "year": int(st.session_state["year"]) if st.session_state.get("year") and not str(st.session_state["year"]).startswith("—") else None,
        "color": st.session_state.get("color"),
        "transmission_type": st.session_state.get("transmission_type"),
        "number_of_doors": st.session_state.get("number_of_doors"),
        "source": st.session_state.get("source"),
        "kilometers": float(km),
    })

else:
    st.warning("No catalog.csv detected. Using independent text/number inputs.")
    for c in ["brand","model","color","transmission_type","number_of_doors","source"]:
        inputs[c] = st.text_input(c)
    inputs["year"] = st.number_input("year", step=1, value=2020)
    inputs["kilometers"] = st.number_input("kilometers", step=1000, value=50000)


if st.button("Predict price (OMR)"):
    X = build_input_df(inputs)
    try:
        reg = model.named_steps.get("reg", model) if isinstance(model, Pipeline) else model
        st.caption(f"Predicting with: **{reg.__class__.__name__}** (from car_price_pipeline.joblib)")
        y_pred = model.predict(X)[0]
        st.success(f"Estimated price: **{float(y_pred):,.0f} OMR**")
        with st.expander("Show inputs"):
            st.dataframe(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
