import streamlit as st
import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO
import numpy as np

from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="BSDV_DUP_CHECK_AI", layout="wide")

st.title("BSDV — AI Duplicate Bug Checker")
st.caption("EXACT + FUZZY + SEMANTIC (TR + EN destekli)")

st.markdown("""
EXACT → Summary birebir aynıysa eşleşme verir  
FUZZY → Yazım olarak çok benzerse eşleşme verir  
SEMANTIC → Yazım farklı olsa da anlam benzerse eşleşme verir
""")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Türkçe karakterleri korur
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_label(match_type: str) -> int:
    order = {"EXACT": 3, "FUZZY": 2, "SEMANTIC": 1}
    return order.get(match_type, 0)


def tokenize(text: str) -> set:
    return set(normalize(text).split())


def extract_feature(text: str) -> str:
    t = normalize(text)

    feature_map = {
        "Chats": [
            "chat", "message", "messages", "starred messages", "reply",
            "forward", "media", "link", "conversation", "mesaj", "sohbet",
            "bubble"
        ],
        "Calls": [
            "call", "voice call", "video call", "voip", "ringing",
            "audio", "microphone", "speaker", "arama", "sesli", "görüşme"
        ],
        "Status": [
            "status", "story", "stories", "durum", "hikaye"
        ],
        "Channels": [
            "channel", "channels", "discover", "discovery", "kanal"
        ],
        "Settings": [
            "settings", "privacy", "notification", "notifications",
            "ayar", "bildirim", "gizlilik"
        ],
        "Search": [
            "search", "arama", "search icon"
        ],
        "Reminders": [
            "reminder", "reminders", "hatırlatıcı", "hatirlatici"
        ],
        "Authentication": [
            "login", "sign in", "signin", "otp", "verification",
            "giriş", "giris", "oturum"
        ]
    }

    for feature, keywords in feature_map.items():
        for kw in keywords:
            if kw in t:
                return feature

    return "Other"


def extract_intent(text: str) -> str:
    t = normalize(text)

    intent_map = {
        "Reply": ["reply", "replied", "reply edilen", "yanit", "yanıt"],
        "Forward": ["forward", "forwarded", "ilet"],
        "Delete": ["delete", "deleted", "sil"],
        "Edit": ["edit", "edited", "duzenle", "düzenle"],
        "Search": ["search", "arama"],
        "Send": ["send", "sending", "gonder", "gönder"],
        "Receive": ["receive", "received", "al"],
        "Open": ["open", "opening", "acil", "açıl", "show", "display"],
        "Crash": ["crash", "crashes", "crashed", "cokme", "çökme"],
        "Layout": [
            "alignment", "aligned", "position", "size", "height", "width",
            "dik", "yatay", "buyukluk", "büyüklük", "gosterilmesi", "gösterilmesi",
            "bubble"
        ]
    }

    for intent, keywords in intent_map.items():
        for kw in keywords:
            if kw in t:
                return intent

    return "General"


def extract_screen_tokens(text: str) -> set:
    t = normalize(text)

    known_phrases = [
        "my reminders",
        "starred messages",
        "chat screen",
        "search screen",
        "settings screen",
        "notification screen",
        "channel details",
        "profile screen",
        "call screen",
        "message info",
        "archived chats",
        "chat ekranında",
        "reply edilen mesaj"
    ]

    found = set()
    for phrase in known_phrases:
        if phrase in t:
            found.add(phrase)

    important_tokens = {
        "reminder", "reminders",
        "starred", "messages", "message",
        "chat", "call", "status", "channel",
        "settings", "notification", "notifications",
        "profile", "search", "archive", "archived",
        "reply", "bubble", "ekran", "screen"
    }

    tokens = set(t.split())
    found.update(tokens & important_tokens)
    return found


def semantic_context_ok(target_summary: str, pool_summary: str) -> bool:
    target_feature = extract_feature(target_summary)
    pool_feature = extract_feature(pool_summary)

    target_screen = extract_screen_tokens(target_summary)
    pool_screen = extract_screen_tokens(pool_summary)

    if target_screen and pool_screen and target_screen.isdisjoint(pool_screen):
        return False

    if target_feature != pool_feature:
        if target_feature == "Other" and pool_feature == "Other":
            t_tokens = tokenize(target_summary)
            p_tokens = tokenize(pool_summary)
            common = t_tokens & p_tokens
            return len(common) >= 2
        return False

    return True


def semantic_intent_ok(target_summary: str, pool_summary: str) -> bool:
    t_intent = extract_intent(target_summary)
    p_intent = extract_intent(pool_summary)

    if t_intent != p_intent:
        return False

    return True


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    delimiter = st.selectbox("CSV delimiter", [";", ","], index=0)
    fuzzy_threshold = st.slider("Fuzzy threshold", 0.70, 1.00, 0.90, 0.01)
    semantic_threshold = st.slider("Semantic threshold", 0.70, 1.00, 0.82, 0.01)
    use_ai = st.toggle("Enable Semantic AI", value=True)
    best_match_only = st.toggle("Best match only", value=True)
    strict_semantic_context = st.toggle("Strict semantic context check", value=True)
    strict_semantic_intent = st.toggle("Strict semantic intent check", value=True)
    show_debug = st.toggle("Show debug info", value=False)

pool_file = st.file_uploader("POOL CSV", type=["csv"])
target_file = st.file_uploader("TARGET CSV", type=["csv"])

run = st.button("🔍 Find Duplicates")

if not pool_file or not target_file:
    st.stop()

if not run:
    st.stop()

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, delimiter: str) -> pd.DataFrame:
    from io import StringIO

    text = file_bytes.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(StringIO(text), sep=delimiter)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["Issue key", "Summary"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}")

    df = df[required_cols].copy()
    df["Issue key"] = df["Issue key"].fillna("").astype(str).str.strip()
    df["Summary"] = df["Summary"].fillna("").astype(str).str.strip()
    df["norm"] = df["Summary"].apply(normalize)
    df["Feature"] = df["Summary"].apply(extract_feature)
    df["Intent"] = df["Summary"].apply(extract_intent)

    df = df[df["norm"] != ""].reset_index(drop=True)
    return df


try:
    pool = load_csv(pool_file.getvalue(), delimiter)
    target = load_csv(target_file.getvalue(), delimiter)
except Exception as e:
    st.error(f"CSV okunamadı: {e}")
    st.stop()

if pool.empty or target.empty:
    st.warning("POOL veya TARGET boş görünüyor.")
    st.stop()

# -------------------------------------------------
# Exact lookup
# -------------------------------------------------
pool_exact_map = {}
for _, row in pool.iterrows():
    pool_exact_map.setdefault(row["Summary"], []).append(row)

# -------------------------------------------------
# Embedding model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner=True)
def build_embeddings(texts: list[str]) -> np.ndarray:
    model = load_model()
    emb = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return np.array(emb, dtype=np.float32)


pool_emb = None
target_emb = None
sim_matrix = None

if use_ai:
    with st.spinner("Semantic model yükleniyor ve embedding hesaplanıyor..."):
        pool_emb = build_embeddings(pool["Summary"].tolist())
        target_emb = build_embeddings(target["Summary"].tolist())
        sim_matrix = np.matmul(target_emb, pool_emb.T)

# -------------------------------------------------
# Matching
# -------------------------------------------------
results = []

progress = st.progress(0)
status = st.empty()

for i, t in target.iterrows():
    status.text(f"Processing {i + 1}/{len(target)}")
    progress.progress((i + 1) / len(target))

    candidate_rows = []

    # EXACT
    exact_hits = pool_exact_map.get(t["Summary"], [])
    for p in exact_hits:
        candidate_rows.append({
            "Target Key": t["Issue key"],
            "Target Summary": t["Summary"],
            "Target Feature": t["Feature"],
            "Target Intent": t["Intent"],
            "Pool Key": p["Issue key"],
            "Pool Summary": p["Summary"],
            "Pool Feature": p["Feature"],
            "Pool Intent": p["Intent"],
            "Type": "EXACT",
            "Score": 1.000
        })

    # FUZZY
    if not (best_match_only and candidate_rows):
        for _, p in pool.iterrows():
            if t["Summary"] == p["Summary"]:
                continue

            f = fuzzy(t["norm"], p["norm"])
            if f >= fuzzy_threshold:
                candidate_rows.append({
                    "Target Key": t["Issue key"],
                    "Target Summary": t["Summary"],
                    "Target Feature": t["Feature"],
                    "Target Intent": t["Intent"],
                    "Pool Key": p["Issue key"],
                    "Pool Summary": p["Summary"],
                    "Pool Feature": p["Feature"],
                    "Pool Intent": p["Intent"],
                    "Type": "FUZZY",
                    "Score": round(float(f), 3)
                })

    # SEMANTIC
    if use_ai and not (best_match_only and any(r["Type"] == "EXACT" for r in candidate_rows)):
        semantic_scores = sim_matrix[i]
        semantic_idx = np.where(semantic_scores >= semantic_threshold)[0]

        for j in semantic_idx:
            p = pool.iloc[j]

            if t["Summary"] == p["Summary"]:
                continue

            if strict_semantic_context and not semantic_context_ok(t["Summary"], p["Summary"]):
                continue

            if strict_semantic_intent and not semantic_intent_ok(t["Summary"], p["Summary"]):
                continue

            s = float(semantic_scores[j])

            candidate_rows.append({
                "Target Key": t["Issue key"],
                "Target Summary": t["Summary"],
                "Target Feature": t["Feature"],
                "Target Intent": t["Intent"],
                "Pool Key": p["Issue key"],
                "Pool Summary": p["Summary"],
                "Pool Feature": p["Feature"],
                "Pool Intent": p["Intent"],
                "Type": "SEMANTIC",
                "Score": round(s, 3)
            })

    if best_match_only and candidate_rows:
        candidate_rows = sorted(
            candidate_rows,
            key=lambda x: (score_label(x["Type"]), x["Score"]),
            reverse=True
        )
        results.append(candidate_rows[0])
    else:
        results.extend(candidate_rows)

progress.empty()
status.empty()

df = pd.DataFrame(results)

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
if not df.empty:
    df = df.drop_duplicates(
        subset=["Target Key", "Pool Key", "Type", "Score"]
    ).reset_index(drop=True)

    df["_type_rank"] = df["Type"].map({"EXACT": 3, "FUZZY": 2, "SEMANTIC": 1}).fillna(0)

    df = (
        df.sort_values(
            ["Target Key", "Pool Key", "_type_rank", "Score"],
            ascending=[True, True, False, False]
        )
        .drop_duplicates(subset=["Target Key", "Pool Key"], keep="first")
        .drop(columns=["_type_rank"])
        .reset_index(drop=True)
    )

# -------------------------------------------------
# Output
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("POOL", len(pool))
col2.metric("TARGET", len(target))
col3.metric("Matches", len(df))
col4.metric("Unique Target Matched", df["Target Key"].nunique() if not df.empty else 0)

if df.empty:
    st.success("Duplicate bulunamadı 🎉")
else:
    st.subheader("Duplicate Results")
    st.dataframe(df, use_container_width=True, height=550)

    summary = (
        df.groupby("Type", dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    st.subheader("Match Summary")
    st.dataframe(summary, use_container_width=True)

    out = BytesIO()
    df.to_csv(out, index=False, sep=";")

    st.download_button(
        "Download Results",
        data=out.getvalue(),
        file_name="duplicates.csv",
        mime="text/csv"
    )

if show_debug:
    st.subheader("Debug")
    st.write("POOL sample")
    st.dataframe(pool.head(), use_container_width=True)

    st.write("TARGET sample")
    st.dataframe(target.head(), use_container_width=True)
