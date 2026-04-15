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
# Basic helpers
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


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def count_matches(text: str, keywords: list[str]) -> int:
    return sum(1 for k in keywords if k in text)


def safe_split_tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def match_type_rank(match_type: str) -> int:
    order = {"EXACT": 3, "FUZZY": 2, "SEMANTIC": 1}
    return order.get(match_type, 0)


# -------------------------------------------------
# Domain extraction
# Domain-first approach
# -------------------------------------------------
def extract_module(text: str) -> str:
    t = normalize(text)

    if contains_any(t, [
        "voice call", "video call", "incoming call", "outgoing call",
        "group call", "call history", "recent calls", "missed call",
        "arama", "görüşme", "gorusme", "çağrı", "cagri", "voip"
    ]):
        return "Calls"

    if contains_any(t, [
        "status", "story", "my status", "text story", "durum", "hikaye"
    ]):
        return "Status"

    if contains_any(t, [
        "channel", "channels", "discover", "following", "service info",
        "channel info", "kanal"
    ]):
        return "Channels"

    if contains_any(t, [
        "profile", "settings", "privacy", "notifications", "storage management",
        "my account", "appearance", "apperance", "help", "blocked contacts",
        "bip web", "paycell", "profil", "ayar", "gizlilik", "bildirim"
    ]):
        return "More"

    if contains_any(t, [
        "chat", "message", "messages", "conversation", "reply", "forward",
        "bubble", "voice message", "voice note", "audio message",
        "ses kaydı", "ses kaydi", "sesli mesaj", "sticker", "emoji",
        "my reminders", "starred messages", "text message", "medya", "media",
        "resim", "image", "photo", "gallery", "mesaj", "sohbet"
    ]):
        return "Chats"

    return "Other"


def extract_object(text: str) -> str:
    t = normalize(text)

    object_map = {
        "Frequent_Contacts": [
            "sık konuşulanlar", "sik konusulanlar", "frequently contacted", "frequent"
        ],
        "Media_Display": [
            "medya gösterim", "medya", "media", "resim", "image", "photo",
            "gallery", "görsel", "gorsel"
        ],
        "Voice_Message": [
            "voice message", "voice note", "audio message", "ses kaydı",
            "ses kaydi", "sesli mesaj"
        ],
        "Voice_Message_Playback": [
            "playback", "play", "pause", "seek", "rewind", "ileri sar",
            "duration", "süre", "sure", "waveform"
        ],
        "Reply": [
            "reply", "reply edilen", "yanıt", "yanit", "reply privately"
        ],
        "Forward": [
            "forward", "forwarded", "ilet", "share"
        ],
        "Bubble": [
            "bubble", "alignment", "position", "size", "height", "width",
            "dik", "yatay", "büyüklük", "buyukluk"
        ],
        "Reminder": [
            "my reminders", "reminder", "reminders", "set reminder",
            "hatırlatıcı", "hatirlatici"
        ],
        "Starred": [
            "starred", "starred messages", "star"
        ],
        "Search": [
            "search", "arama", "search icon"
        ],
        "Sticker": [
            "sticker"
        ],
        "Emoji": [
            "emoji"
        ],
        "Reaction": [
            "reaction"
        ],
        "Document": [
            "document", "docs"
        ],
        "Location": [
            "location", "konum"
        ],
        "Poll": [
            "poll"
        ],
        "Call_Voice": [
            "voice call", "incoming voice call", "outgoing voice call"
        ],
        "Call_Video": [
            "video call", "incoming video call", "outgoing video call"
        ],
        "Call_History": [
            "recent calls", "call history", "missed call"
        ],
        "Dialer": [
            "dial a number", "new contact", "delete number"
        ],
        "Status": [
            "status", "my status"
        ],
        "Camera": [
            "camera"
        ],
        "Text_Story": [
            "text story"
        ],
    }

    best_name = "General"
    best_score = 0
    for name, keywords in object_map.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def extract_action(text: str) -> str:
    t = normalize(text)

    action_map = {
        "Crash": ["crash", "crashes", "crashed", "çök", "cok"],
        "Freeze_Open": [
            "uygulama dondu", "dondu", "freeze", "frozen", "açılmadı",
            "acilmadi", "not opening", "cannot open", "open", "opening"
        ],
        "Load_Display": [
            "load", "loading", "yüklen", "yuklen", "display", "show",
            "appear", "gelmiyor", "görünm", "goster", "göster", "listelen"
        ],
        "Send": ["send", "sending", "gönder", "gonder"],
        "Receive": ["receive", "received", "gel", "al"],
        "Reply": ["reply", "reply edilen", "yanıt", "yanit"],
        "Forward": ["forward", "forwarded", "ilet", "share"],
        "Playback": ["play", "pause", "seek", "rewind", "ileri sar", "duration", "süre", "sure"],
        "Search": ["search", "arama"],
        "Delete": ["delete", "deleted", "sil"],
        "Edit": ["edit", "edited", "düzenle", "duzenle"],
        "UI_Layout": [
            "alignment", "position", "size", "height", "width", "dik",
            "yatay", "büyüklük", "buyukluk", "spacing", "fade", "faded", "ui"
        ],
        "Call": ["call", "arama", "görüşme", "gorusme"],
    }

    best_name = "General"
    best_score = 0
    for name, keywords in action_map.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def extract_failure(text: str) -> str:
    t = normalize(text)

    failure_map = {
        "App_Freeze": [
            "uygulama dondu", "dondu", "freeze", "frozen", "açılmadı",
            "acilmadi", "not opening", "cannot open", "launch", "startup"
        ],
        "Crash": ["crash", "crashes", "crashed", "çök", "cok"],
        "Load_Failure": [
            "load problemi", "loading problem", "yüklenmiyor", "yuklenmiyor",
            "gelmiyor", "listelenmiyor", "görünmüyor", "gorunmuyor"
        ],
        "Playback_Failure": [
            "playback", "play", "pause", "rewind", "seek", "ileri sar",
            "duration", "süre", "sure"
        ],
        "UI_Layout": [
            "alignment", "position", "size", "height", "width", "bubble",
            "dik", "yatay", "büyüklük", "buyukluk", "spacing", "fade", "faded"
        ],
        "General_Display": [
            "display", "show", "appear", "göster", "goster"
        ],
    }

    best_name = "General"
    best_score = 0
    for name, keywords in failure_map.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


# -------------------------------------------------
# Domain-first candidate filter
# -------------------------------------------------
def domain_gate(a: dict, b: dict) -> bool:
    # 1) module farklıysa direkt red
    if a["Module"] != b["Module"]:
        return False

    # 2) object ikisi de spesifikse ve farklıysa red
    if a["Object"] != "General" and b["Object"] != "General":
        if a["Object"] != b["Object"]:
            return False

    # 3) failure ikisi de spesifikse ve farklıysa red
    if a["Failure"] != "General" and b["Failure"] != "General":
        if a["Failure"] != b["Failure"]:
            return False

    # 4) action ikisi de güçlü aksiyonsa ve farklıysa red
    strong_actions = {"Reply", "Forward", "Playback", "Call", "Search", "Freeze_Open"}
    if a["Action"] in strong_actions and b["Action"] in strong_actions:
        if a["Action"] != b["Action"]:
            return False

    # 5) kritik ayrım: call vs voice message
    call_objects = {"Call_Voice", "Call_Video", "Call_History", "Dialer"}
    voice_objects = {"Voice_Message", "Voice_Message_Playback"}

    if (a["Object"] in call_objects and b["Object"] in voice_objects) or \
       (b["Object"] in call_objects and a["Object"] in voice_objects):
        return False

    return True


def domain_distance(a: dict, b: dict) -> int:
    score = 0
    if a["Module"] != b["Module"]:
        score += 5
    if a["Object"] != b["Object"]:
        score += 4
    if a["Action"] != b["Action"]:
        score += 2
    if a["Failure"] != b["Failure"]:
        score += 3
    return score


def build_reason(match_type: str, a: dict, b: dict) -> str:
    if match_type == "EXACT":
        return "EXACT_TEXT"
    if match_type == "FUZZY":
        return "FUZZY_SIMILAR_TEXT"
    if a["Object"] == b["Object"] and a["Object"] != "General":
        return f'SEMANTIC_SAME_OBJECT:{a["Object"]}'
    if a["Failure"] == b["Failure"] and a["Failure"] != "General":
        return f'SEMANTIC_SAME_FAILURE:{a["Failure"]}'
    return "SEMANTIC_GENERAL"


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    delimiter = st.selectbox("CSV delimiter", [";", ","], index=0)
    fuzzy_threshold = st.slider("Fuzzy threshold", 0.70, 1.00, 0.90, 0.01)
    semantic_threshold = st.slider("Semantic threshold", 0.70, 1.00, 0.85, 0.01)
    use_ai = st.toggle("Enable Semantic AI", value=True)
    best_match_only = st.toggle("Best match only", value=True)
    use_domain_gate = st.toggle("Use domain-first filter", value=True)
    show_debug = st.toggle("Show debug info", value=False)

pool_file = st.file_uploader("POOL CSV", type=["csv"])
target_file = st.file_uploader("TARGET CSV", type=["csv"])
run = st.button("🔍 Find Duplicates")

if not pool_file or not target_file:
    st.stop()

if not run:
    st.stop()

# -------------------------------------------------
# Load
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

    df["Module"] = df["Summary"].apply(extract_module)
    df["Object"] = df["Summary"].apply(extract_object)
    df["Action"] = df["Summary"].apply(extract_action)
    df["Failure"] = df["Summary"].apply(extract_failure)

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
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
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
            "Target Module": t_row["Module"],
            "Target Object": t_row["Object"],
            "Target Action": t_row["Action"],
            "Target Failure": t_row["Failure"],
            "Pool Key": p_row["Issue key"],
            "Pool Summary": p_row["Summary"],
            "Pool Module": p_row["Module"],
            "Pool Object": p_row["Object"],
            "Pool Action": p_row["Action"],
            "Pool Failure": p_row["Failure"],
            "Type": "EXACT",
            "Score": 1.000,
            "Domain Distance": 0,
            "Match Reason": build_reason("EXACT", t_row, p_row),
        })

    # FUZZY
    if not (best_match_only and candidate_rows):
        for _, p in pool.iterrows():
            p_row = p.to_dict()

            if t_row["Summary"] == p_row["Summary"]:
                continue

            if use_domain_gate and not domain_gate(t_row, p_row):
                continue

            f = fuzzy(t_row["norm"], p_row["norm"])
            if f >= fuzzy_threshold:
                candidate_rows.append({
                    "Target Key": t_row["Issue key"],
                    "Target Summary": t_row["Summary"],
                    "Target Module": t_row["Module"],
                    "Target Object": t_row["Object"],
                    "Target Action": t_row["Action"],
                    "Target Failure": t_row["Failure"],
                    "Pool Key": p_row["Issue key"],
                    "Pool Summary": p_row["Summary"],
                    "Pool Module": p_row["Module"],
                    "Pool Object": p_row["Object"],
                    "Pool Action": p_row["Action"],
                    "Pool Failure": p_row["Failure"],
                    "Type": "FUZZY",
                    "Score": round(float(f), 3),
                    "Domain Distance": domain_distance(t_row, p_row),
                    "Match Reason": build_reason("FUZZY", t_row, p_row),
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

            if use_domain_gate and not domain_gate(t_row, p_row):
                continue

            s = float(semantic_scores[j])

            candidate_rows.append({
                "Target Key": t_row["Issue key"],
                "Target Summary": t_row["Summary"],
                "Target Module": t_row["Module"],
                "Target Object": t_row["Object"],
                "Target Action": t_row["Action"],
                "Target Failure": t_row["Failure"],
                "Pool Key": p_row["Issue key"],
                "Pool Summary": p_row["Summary"],
                "Pool Module": p_row["Module"],
                "Pool Object": p_row["Object"],
                "Pool Action": p_row["Action"],
                "Pool Failure": p_row["Failure"],
                "Type": "SEMANTIC",
                "Score": round(s, 3),
                "Domain Distance": domain_distance(t_row, p_row),
                "Match Reason": build_reason("SEMANTIC", t_row, p_row),
            })

    if best_match_only and candidate_rows:
        candidate_rows = sorted(
            candidate_rows,
            key=lambda x: (
                match_type_rank(x["Type"]),
                -x["Domain Distance"],
                x["Score"]
            ),
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
    df = df.drop_duplicates(subset=["Target Key", "Pool Key", "Type", "Score"]).reset_index(drop=True)

    df["_type_rank"] = df["Type"].map({"EXACT": 3, "FUZZY": 2, "SEMANTIC": 1}).fillna(0)

    df = (
        df.sort_values(
            ["Target Key", "_type_rank", "Domain Distance", "Score"],
            ascending=[True, False, True, False]
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
        file_name="duplicates.csv",
        mime="text/csv"
    )

if show_debug:
    st.subheader("Debug")
    st.write("TARGET sample")
    st.dataframe(target.head(20), use_container_width=True)
    st.write("POOL sample")
    st.dataframe(pool.head(20), use_container_width=True)
