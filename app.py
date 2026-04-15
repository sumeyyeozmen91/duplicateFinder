import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher
from io import BytesIO

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="BSDV_DUP_CHECK_AI", layout="wide")

st.title("BSDV — AI Duplicate Bug Checker")
st.caption("EXACT + FUZZY + SEMANTIC (TR + EN destekli)")

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

def fuzzy(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    delimiter = st.selectbox("CSV delimiter", [";", ","], index=0)
    fuzzy_threshold = st.slider("Fuzzy threshold", 0.7, 1.0, 0.85)
    semantic_threshold = st.slider("Semantic threshold", 0.7, 1.0, 0.80)
    use_ai = st.toggle("Enable Semantic AI", value=True)

pool_file = st.file_uploader("POOL CSV", type=["csv"])
target_file = st.file_uploader("TARGET CSV", type=["csv"])

run = st.button("🔍 Find Duplicates")

if not pool_file or not target_file:
    st.stop()

if not run:
    st.stop()

# ----------------------------
# Load data
# ----------------------------
def load(file):
    df = pd.read_csv(file, sep=delimiter)
    df.columns = [c.strip() for c in df.columns]

    df = df[["Issue key", "Summary"]].copy()
    df["norm"] = df["Summary"].apply(normalize)
    return df

pool = load(pool_file)
target = load(target_file)

# ----------------------------
# Embedding model
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_model() if use_ai else None

if use_ai:
    pool_emb = model.encode(pool["Summary"].tolist(), show_progress_bar=True)
    target_emb = model.encode(target["Summary"].tolist(), show_progress_bar=True)

# ----------------------------
# Matching
# ----------------------------
results = []

for i, t in target.iterrows():
    for j, p in pool.iterrows():

        # EXACT
        if t["Summary"] == p["Summary"]:
            results.append({
                "Target Key": t["Issue key"],
                "Target Summary": t["Summary"],
                "Pool Key": p["Issue key"],
                "Pool Summary": p["Summary"],
                "Type": "EXACT",
                "Score": 1.0
            })
            continue

        # FUZZY
        f = fuzzy(t["norm"], p["norm"])
        if f >= fuzzy_threshold:
            results.append({
                "Target Key": t["Issue key"],
                "Target Summary": t["Summary"],
                "Pool Key": p["Issue key"],
                "Pool Summary": p["Summary"],
                "Type": "FUZZY",
                "Score": round(f, 3)
            })
            continue

        # SEMANTIC
        if use_ai:
            s = cosine_similarity(
                [target_emb[i]],
                [pool_emb[j]]
            )[0][0]

            if s >= semantic_threshold:
                results.append({
                    "Target Key": t["Issue key"],
                    "Target Summary": t["Summary"],
                    "Pool Key": p["Issue key"],
                    "Pool Summary": p["Summary"],
                    "Type": "SEMANTIC",
                    "Score": round(float(s), 3)
                })

df = pd.DataFrame(results)

# ----------------------------
# UI output
# ----------------------------
st.metric("Matches", len(df))

if df.empty:
    st.success("Duplicate bulunamadı 🎉")
else:
    st.dataframe(df, use_container_width=True, height=500)

    out = BytesIO()
    df.to_csv(out, index=False, sep=";")

    st.download_button(
        "Download Results",
        data=out.getvalue(),
        file_name="duplicates.csv",
        mime="text/csv"
    )
