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
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_type_rank(match_type: str) -> int:
    order = {"EXACT": 3, "FUZZY": 2, "SEMANTIC": 1}
    return order.get(match_type, 0)


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def count_matches(text: str, keywords: list[str]) -> int:
    return sum(1 for k in keywords if k in text)


# -------------------------------------------------
# Mini filter: Top Module
# -------------------------------------------------
def extract_top_module(summary: str) -> str:
    s = normalize(summary)

    if contains_any(s, ["mesajlar", "messages", "chat", "chats", "mesaj", "sohbet"]):
        return "Messages"

    if contains_any(s, ["aramalar", "calls", "call", "arama", "görüşme", "gorusme"]):
        return "Calls"

    if contains_any(s, ["kanallar", "channels", "channel", "kanal"]):
        return "Channels"

    if contains_any(s, ["status", "story", "stories", "durum", "hikaye"]):
        return "Status"

    if contains_any(s, ["more", "settings", "profile", "ayar", "profil", "privacy", "notifications"]):
        return "More"

    return "Other"


# -------------------------------------------------
# Mini filter: Object Label
# This does NOT decide duplicate.
# It only blocks very absurd matches.
# -------------------------------------------------
OBJECT_PATTERNS = {
    "Frequent_Contacts": [
        "sık konuşulanlar", "sik konusulanlar", "frequently contacted", "frequent contacts"
    ],
    "Voice_Message": [
        "ses kaydı", "ses kaydi", "voice message", "voice note", "audio message", "sesli mesaj"
    ],
    "Voice_Message_Playback": [
        "ileri sar", "rewind", "seek", "playback", "duration", "süre", "sure", "pause", "play"
    ],
    "Media": [
        "medya", "media", "resim", "image", "photo", "görsel", "gorsel", "gallery"
    ],
    "Bubble": [
        "bubble", "alignment", "position", "dik", "yatay", "büyüklük", "buyukluk", "size"
    ],
    "Reply": [
        "reply", "reply edilen", "yanıt", "yanit", "reply privately"
    ],
    "Forward": [
        "forward", "forwarded", "ilet", "share"
    ],
    "Search": [
        "search", "arama", "search icon"
    ],
    "Reminder": [
        "reminder", "reminders", "my reminders", "set reminder", "hatırlatıcı", "hatirlatici"
    ],
    "Starred": [
        "starred", "starred messages", "star"
    ],
    "Call_Voice": [
        "voice call", "incoming voice call", "outgoing voice call"
    ],
    "Call_Video": [
        "video call", "incoming video call", "outgoing video call"
    ],
    "Connection": [
        "connection", "bağlan", "baglan", "network", "internet"
    ],
    "Freeze": [
        "uygulama dondu", "dondu", "freeze", "frozen", "açılmadı", "acilmadi", "not opening"
    ],
    "Crash": [
        "crash", "crashes", "crashed", "çök", "cok"
    ],
    "Link_Share": [
        "channel link", "share link", "link paylaş", "link paylas", "share channel link"
    ],
    "Mention": [
        "mention", "@"
    ],
}


def extract_object_label(summary: str) -> str:
    s = normalize(summary)

    best_label = "General"
    best_score = 0

    for label, patterns in OBJECT_PATTERNS.items():
        score = count_matches(s, patterns)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def mini_filter_ok(target_row: dict, pool_row: dict) -> bool:
    # 1) Top module farklıysa ele
    if target_row["Top Module"] != pool_row["Top Module"]:
        return False

    # 2) İkisinde de net obje varsa ve farklıysa ele
    if target_row["Object Label"] != "General" and pool_row["Object Label"] != "General":
        if target_row["Object Label"] != pool_row["Object Label"]:
            return False

    # 3) Voice message ile voice/video call karışmasın
    voice_message_labels = {"Voice_Message", "Voice_Message_Playback"}
    call_labels = {"Call_Voice", "Call_Video"}

    if (
        target_row["Object Label"] in voice_message_labels and pool_row["Object Label"] in call_labels
    ) or (
        pool_row["Object Label"] in voice_message_labels and target_row["Object Label"] in call_labels
    ):
        return False

    return True


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    delimiter = st.selectbox("CSV delimiter", [";", ","], index=0)
    fuzzy_threshold = st.slider("Fuzzy threshold", 0.70, 1.00, 0.90, 0.01)
    semantic_threshold = st.slider("Semantic threshold", 0.70, 1.00, 0.85, 0.01)
    use_ai = st.toggle("Enable Semantic AI", value=True)
    best_match_only = st.toggle("Best match only", value=True)
    use_mini_filter = st.toggle("Use mini absurd-match filter", value=True)
    show_debug = st.toggle("Show debug info", value=False)

pool_file = st.file_uploader("POOL CSV", type=["csv"])
target_file = st.file_uploader("TARGET CSV", type=["csv"])
run = st.button("🔍 Find Similar Bugs")

if not pool_file or not target_file:
    st.stop()

if not run:
    st.stop()

# -------------------------------------------------
# Load CSV
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
    df["Top Module"] = df["Summary"].apply(extract_top_module)
    df["Object Label"] = df["Summary"].apply(extract_object_label)

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
# Exact map
# -------------------------------------------------
pool_exact_map = {}
for _, row in pool.iterrows():
    pool_exact_map.setdefault(row["Summary"], []).append(row.to_dict())

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
    t_row = t.to_dict()
    status.text(f"Processing {i + 1}/{len(target)}")
    progress.progress((i + 1) / len(target))

    candidate_rows = []

    # EXACT
    for p_row in pool_exact_map.get(t_row["Summary"], []):
        candidate_rows.append({
            "Target Key": t_row["Issue key"],
            "Target Summary": t_row["Summary"],
            "Target Top Module": t_row["Top Module"],
            "Target Object Label": t_row["Object Label"],
            "Pool Key": p_row["Issue key"],
            "Pool Summary": p_row["Summary"],
            "Pool Top Module": p_row["Top Module"],
            "Pool Object Label": p_row["Object Label"],
            "Type": "EXACT",
            "Score": 1.000
        })

    # FUZZY
    if not (best_match_only and candidate_rows):
        for _, p in pool.iterrows():
            p_row = p.to_dict()

            if t_row["Summary"] == p_row["Summary"]:
                continue

            if use_mini_filter and not mini_filter_ok(t_row, p_row):
                continue

            f = fuzzy(t_row["norm"], p_row["norm"])
            if f >= fuzzy_threshold:
                candidate_rows.append({
                    "Target Key": t_row["Issue key"],
                    "Target Summary": t_row["Summary"],
                    "Target Top Module": t_row["Top Module"],
                    "Target Object Label": t_row["Object Label"],
                    "Pool Key": p_row["Issue key"],
                    "Pool Summary": p_row["Summary"],
                    "Pool Top Module": p_row["Top Module"],
                    "Pool Object Label": p_row["Object Label"],
                    "Type": "FUZZY",
                    "Score": round(float(f), 3)
                })

    # SEMANTIC
    if use_ai and not (best_match_only and any(r["Type"] == "EXACT" for r in candidate_rows)):
        semantic_scores = sim_matrix[i]
        semantic_idx = np.where(semantic_scores >= semantic_threshold)[0]

        for j in semantic_idx:
            p = pool.iloc[j]
            p_row = p.to_dict()

            if t_row["Summary"] == p_row["Summary"]:
                continue

            if use_mini_filter and not mini_filter_ok(t_row, p_row):
                continue

            s = float(semantic_scores[j])

            candidate_rows.append({
                "Target Key": t_row["Issue key"],
                "Target Summary": t_row["Summary"],
                "Target Top Module": t_row["Top Module"],
                "Target Object Label": t_row["Object Label"],
                "Pool Key": p_row["Issue key"],
                "Pool Summary": p_row["Summary"],
                "Pool Top Module": p_row["Top Module"],
                "Pool Object Label": p_row["Object Label"],
                "Type": "SEMANTIC",
                "Score": round(s, 3)
            })

    if best_match_only and candidate_rows:
        candidate_rows = sorted(
            candidate_rows,
            key=lambda x: (match_type_rank(x["Type"]), x["Score"]),
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
            ["Target Key", "_type_rank", "Score"],
            ascending=[True, False, False]
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
    st.success("Benzer kayıt bulunamadı 🎉")
else:
    st.subheader("Similarity Results")
    st.dataframe(df, use_container_width=True, height=560)

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
        file_name="similarity_results.csv",
        mime="text/csv"
    )

if show_debug:
    st.subheader("Debug")
    st.write("TARGET sample")
    st.dataframe(target.head(20), use_container_width=True)

    st.write("POOL sample")
    st.dataframe(pool.head(20), use_container_width=True)
