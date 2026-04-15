import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher
from io import BytesIO

st.set_page_config(page_title="BSDV_DUP_CHECK", layout="wide")

st.title("BSDV — Bug Duplicate Checker (POOL vs TARGET)")
st.caption("Sadece SUMMARY bazlı EXACT ve SEMANTIC duplicate kontrolü")

# ----------------------------
# Helpers
# ----------------------------
def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("Settings")
    delimiter = st.selectbox("CSV delimiter", [";", ","], index=0)
    threshold = st.slider("Semantic similarity threshold", 0.7, 1.0, 0.85)

st.subheader("Upload CSVs")

pool_file = st.file_uploader("POOL CSV (reference)", type=["csv"])
target_file = st.file_uploader("TARGET CSV (to check)", type=["csv"])

if not pool_file or not target_file:
    st.info("İki CSV de yüklenmeli.")
    st.stop()

# ----------------------------
# Load
# ----------------------------
def load_csv(file):
    df = pd.read_csv(file, sep=delimiter, engine="python")
    df.columns = [c.strip() for c in df.columns]

    required = ["Issue key", "Summary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df = df[required].copy()
    df["Summary_norm"] = df["Summary"].apply(normalize)

    return df

pool_df = load_csv(pool_file)
target_df = load_csv(target_file)

# ----------------------------
# Matching
# ----------------------------
results = []

for _, t in target_df.iterrows():
    for _, p in pool_df.iterrows():

        # EXACT
        if t["Summary"] == p["Summary"]:
            results.append({
                "Target Issue": t["Issue key"],
                "Pool Issue": p["Issue key"],
                "Type": "EXACT",
                "Similarity": 1.0
            })
            continue

        # SEMANTIC
        sim = similarity(t["Summary_norm"], p["Summary_norm"])
        if sim >= threshold:
            results.append({
                "Target Issue": t["Issue key"],
                "Pool Issue": p["Issue key"],
                "Type": "SEMANTIC",
                "Similarity": round(sim, 3)
            })

result_df = pd.DataFrame(results)

# ----------------------------
# UI Output
# ----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Target count", len(target_df))
col2.metric("Pool count", len(pool_df))
col3.metric("Matches found", len(result_df))

st.divider()

if result_df.empty:
    st.success("Duplicate bulunamadı 🎉")
else:
    st.subheader("Matches")
    st.dataframe(result_df, use_container_width=True, height=500)

    # download
    out = BytesIO()
    result_df.to_csv(out, sep=";", index=False)

    st.download_button(
        "Download Results",
        data=out.getvalue(),
        file_name="duplicate_results.csv",
        mime="text/csv"
    )
