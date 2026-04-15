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
# Text helpers
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


def match_type_rank(match_type: str) -> int:
    order = {"EXACT": 3, "FUZZY": 2, "SEMANTIC": 1}
    return order.get(match_type, 0)


# -------------------------------------------------
# Domain dictionaries
# -------------------------------------------------
MAIN_FEATURE_KEYWORDS = {
    "Calls": [
        "voice call", "video call", "incoming call", "outgoing call", "missed call",
        "group call", "call history", "recent calls", "dial a number", "arama",
        "görüşme", "gorusme", "çağrı", "cagri", "voip", "ringing"
    ],
    "Chats": [
        "chat", "chats", "message", "messages", "conversation", "mesaj", "sohbet",
        "reply", "forward", "bubble", "sticker", "emoji", "dictation",
        "starred messages", "my reminders", "secret message",
        "voice message", "voice note", "audio message", "ses kaydı", "ses kaydi",
        "sesli mesaj", "text message", "one to one chat", "group chat",
        "resim", "image", "photo", "gallery"
    ],
    "Status": [
        "status", "my status", "text story", "story", "stories", "durum", "hikaye"
    ],
    "Channels": [
        "channel", "channels", "discover", "following", "service info", "channel info",
        "kanal"
    ],
    "More": [
        "profile", "bip web", "storage management", "blocked contacts",
        "my account", "appearance", "apperance", "help", "emergency", "paycell",
        "settings", "chat settings", "privacy", "notifications", "ayar", "bildirim",
        "gizlilik", "profil"
    ],
}

SUB_FEATURE_KEYWORDS = {
    "Chats_List": [
        "chat list", "filter", "unread", "pin", "archive", "archived chats",
        "select chat", "select all", "read all", "scroll right", "scroll left",
        "delete chat", "clear chat", "export chat", "sık konuşulanlar",
        "sik konusulanlar", "frequently contacted", "frequent"
    ],
    "Chats_OneToOne": [
        "one to one chat", "contact header", "contact info", "profile photo",
        "chat search", "secret message", "translate settings",
        "groups in common", "view in adress book", "view in address book"
    ],
    "Chats_Group": [
        "group chat", "group subject", "group scheduler", "group settings",
        "add participants", "member", "make admin", "remove from group", "exit group",
        "invite with link", "reply privately"
    ],
    "Chats_Attachments": [
        "plus attach", "attach", "gallery", "document", "contact", "location",
        "poll", "camera", "instant video", "send money", "resim", "image", "photo"
    ],
    "Chats_Reminders": [
        "my reminders", "set reminder", "reminder", "reminders", "hatırlatıcı", "hatirlatici"
    ],
    "Chats_Starred": [
        "starred", "starred messages", "star"
    ],
    "Chats_Search": [
        "chat search", "search", "arama"
    ],
    "Chats_VoiceMessage": [
        "voice message", "voice note", "audio message", "ses kaydı", "ses kaydi",
        "sesli mesaj", "playback", "play", "pause", "seek", "rewind", "ileri sar",
        "duration", "süre", "sure", "waveform"
    ],
    "Calls_List": [
        "recent calls", "call history", "missed", "all calls", "favorites"
    ],
    "Calls_NewGroup": [
        "new group call", "group call"
    ],
    "Calls_Dialer": [
        "dial a number", "new contact", "delete number"
    ],
    "Calls_Voice": [
        "voice call", "incoming voice call", "outgoing voice call"
    ],
    "Calls_Video": [
        "video call", "incoming video call", "outgoing video call"
    ],
    "Status_Status": [
        "status", "my status"
    ],
    "Status_Camera": [
        "camera"
    ],
    "Status_TextStory": [
        "text story"
    ],
    "Channels_List": [
        "following", "discover", "filter", "search", "archive", "delete"
    ],
    "Channels_Info": [
        "service info", "channel info"
    ],
    "More_Profile": [
        "profile", "photo", "name", "customize", "games", "paycell", "emergency",
        "starred", "bip web", "invite"
    ],
    "More_Settings": [
        "settings", "chat settings", "backup", "chat wallpaper", "automatic download",
        "save to gallery", "read receipts", "storage management", "notifications",
        "blocked contacts", "my account", "privacy", "appearance", "apperance", "help"
    ],
}

COMPONENT_KEYWORDS = {
    "Frequent_Contacts": [
        "sık konuşulanlar", "sik konusulanlar", "frequently contacted", "frequent"
    ],
    "Reply": [
        "reply", "reply edilen", "yanıt", "yanit", "reply privately"
    ],
    "Forward": [
        "forward", "forwarded", "ilet", "share"
    ],
    "Voice_Message": [
        "voice message", "voice note", "audio message", "ses kaydı", "ses kaydi",
        "sesli mesaj"
    ],
    "Voice_Message_Playback": [
        "playback", "play", "pause", "seek", "rewind", "ileri sar", "duration",
        "süre", "sure", "waveform"
    ],
    "Bubble_UI": [
        "bubble", "alignment", "position", "size", "height", "width", "dik", "yatay",
        "büyüklük", "buyukluk"
    ],
    "Search": [
        "search", "arama", "search icon"
    ],
    "Starred": [
        "starred", "starred messages", "star"
    ],
    "Reminder": [
        "my reminders", "set reminder", "reminder", "reminders", "hatırlatıcı", "hatirlatici"
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
    "Text_Message": [
        "text message", "message"
    ],
    "Camera": [
        "camera", "instant video"
    ],
    "Gallery_Image": [
        "gallery", "resim", "image", "photo", "görsel", "gorsel"
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
    "Secret_Message": [
        "secret message"
    ],
    "Notifications": [
        "notification", "notifications", "bildirim"
    ],
    "Contact_Info": [
        "contact info", "contact header", "phone number", "contact name"
    ],
    "Call_Voice": [
        "voice call", "incoming voice call", "outgoing voice call"
    ],
    "Call_Video": [
        "video call", "incoming video call", "outgoing video call"
    ],
    "Call_History": [
        "recent calls", "call history", "missed call", "favorites"
    ],
    "Dialer": [
        "dial a number", "new contact", "delete number"
    ],
}

ACTION_KEYWORDS = {
    "Crash": [
        "crash", "crashes", "crashed", "çök", "cok"
    ],
    "Freeze_Open": [
        "uygulama dondu", "dondu", "freeze", "frozen", "açılmadı", "acilmadi",
        "not opening", "cannot open", "open", "opening"
    ],
    "Load_Display": [
        "load", "loading", "yüklen", "yuklen", "appear", "display", "gelmiyor",
        "görünm", "goster", "göster", "listelen"
    ],
    "Send": [
        "send", "sending", "gönder", "gonder"
    ],
    "Receive": [
        "receive", "received", "gel", "al"
    ],
    "Reply": [
        "reply", "reply edilen", "yanıt", "yanit"
    ],
    "Forward": [
        "forward", "forwarded", "ilet", "share"
    ],
    "Playback": [
        "play", "pause", "rewind", "seek", "ileri sar", "duration", "süre", "sure"
    ],
    "Search": [
        "search", "arama"
    ],
    "Delete": [
        "delete", "deleted", "sil"
    ],
    "Edit": [
        "edit", "edited", "düzenle", "duzenle"
    ],
    "Layout_UI": [
        "alignment", "position", "size", "height", "width", "dik", "yatay",
        "büyüklük", "buyukluk", "spacing", "fade", "faded", "ui"
    ],
    "Call": [
        "voice call", "video call", "call", "arama", "görüşme", "gorusme"
    ],
}

FAILURE_MODE_KEYWORDS = {
    "App_Freeze_Launch": [
        "uygulama dondu", "dondu", "freeze", "frozen", "açılmadı", "acilmadi",
        "app not opening", "not opening", "cannot open", "launch", "startup"
    ],
    "Crash": [
        "crash", "crashes", "crashed", "çök", "cok"
    ],
    "Media_Load": [
        "resim load", "image load", "photo load", "media load", "yüklenmiyor",
        "yuklenmiyor", "load problemi", "loading problem", "resim", "image", "photo"
    ],
    "Voice_Playback": [
        "ses kaydı", "ses kaydi", "voice message", "audio message", "playback",
        "play", "pause", "rewind", "seek", "ileri sar", "duration", "süre", "sure"
    ],
    "UI_Layout": [
        "alignment", "position", "size", "height", "width", "bubble", "dik", "yatay",
        "büyüklük", "buyukluk", "spacing", "fade", "faded"
    ],
    "Load_Display": [
        "load", "loading", "appear", "display", "gelmiyor", "görünm", "listelen"
    ],
}


# -------------------------------------------------
# Soft parsing
# -------------------------------------------------
def extract_main_feature(text: str) -> str:
    t = normalize(text)
    best_name = "Other"
    best_score = 0

    for name, keywords in MAIN_FEATURE_KEYWORDS.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    if contains_any(t, ["voice call", "video call", "incoming call", "outgoing call", "group call", "call history"]):
        return "Calls"

    if contains_any(t, ["voice message", "voice note", "audio message", "ses kaydı", "ses kaydi", "sesli mesaj"]):
        return "Chats"

    return best_name


def extract_sub_feature(text: str, main_feature: str | None = None) -> str:
    t = normalize(text)
    mf = main_feature or extract_main_feature(text)

    candidates = {name: kws for name, kws in SUB_FEATURE_KEYWORDS.items() if name.startswith(mf)}
    best_name = f"{mf}_General" if mf != "Other" else "Other"
    best_score = 0

    for name, keywords in candidates.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    if mf == "Chats":
        if contains_any(t, ["voice message", "voice note", "audio message", "ses kaydı", "ses kaydi", "sesli mesaj"]):
            return "Chats_VoiceMessage"
        if contains_any(t, ["my reminders", "set reminder", "reminder", "reminders", "hatırlatıcı", "hatirlatici"]):
            return "Chats_Reminders"
        if contains_any(t, ["starred", "starred messages", "star"]):
            return "Chats_Starred"
        if contains_any(t, ["group chat"]):
            return "Chats_Group"
        if contains_any(t, ["gallery", "document", "location", "poll", "camera", "instant video", "attach", "resim", "image", "photo"]):
            return "Chats_Attachments"
        if contains_any(t, ["sık konuşulanlar", "sik konusulanlar", "frequently contacted", "frequent", "chat list", "archive"]):
            return "Chats_List"

    if mf == "Calls":
        if contains_any(t, ["voice call", "incoming voice call", "outgoing voice call"]):
            return "Calls_Voice"
        if contains_any(t, ["video call", "incoming video call", "outgoing video call"]):
            return "Calls_Video"
        if contains_any(t, ["dial a number", "new contact", "delete number"]):
            return "Calls_Dialer"
        if contains_any(t, ["new group call", "group call"]):
            return "Calls_NewGroup"

    return best_name


def extract_component(text: str) -> str:
    t = normalize(text)

    best_name = "General"
    best_score = 0
    for name, keywords in COMPONENT_KEYWORDS.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    if contains_any(t, ["voice message", "voice note", "audio message", "ses kaydı", "ses kaydi", "sesli mesaj"]):
        if contains_any(t, ["playback", "play", "pause", "seek", "rewind", "ileri sar", "duration", "süre", "sure", "waveform"]):
            return "Voice_Message_Playback"
        return "Voice_Message"

    if contains_any(t, ["voice call", "incoming voice call", "outgoing voice call"]):
        return "Call_Voice"

    if contains_any(t, ["video call", "incoming video call", "outgoing video call"]):
        return "Call_Video"

    return best_name


def extract_action(text: str) -> str:
    t = normalize(text)

    best_name = "General"
    best_score = 0
    for name, keywords in ACTION_KEYWORDS.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def extract_failure_mode(text: str) -> str:
    t = normalize(text)

    best_name = "General"
    best_score = 0
    for name, keywords in FAILURE_MODE_KEYWORDS.items():
        score = count_matches(t, keywords)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def domain_distance(row_a: dict, row_b: dict) -> int:
    penalty = 0
    if row_a["Main Feature"] != row_b["Main Feature"]:
        penalty += 4
    if row_a["Sub Feature"] != row_b["Sub Feature"]:
        penalty += 3
    if row_a["Component"] != row_b["Component"]:
        penalty += 2
    if row_a["Action"] != row_b["Action"]:
        penalty += 1
    if row_a["Failure Mode"] != row_b["Failure Mode"]:
        penalty += 3
    return penalty


def semantic_guardrail_ok(row_a: dict, row_b: dict) -> bool:
    if row_a["Main Feature"] != row_b["Main Feature"]:
        return False

    if row_a["Failure Mode"] != "General" and row_b["Failure Mode"] != "General":
        if row_a["Failure Mode"] != row_b["Failure Mode"]:
            return False

    voice_message_components = {"Voice_Message", "Voice_Message_Playback"}
    call_components = {"Call_Voice", "Call_Video"}

    if (
        row_a["Component"] in voice_message_components and row_b["Component"] in call_components
    ) or (
        row_b["Component"] in voice_message_components and row_a["Component"] in call_components
    ):
        return False

    strong_components = {
        "Frequent_Contacts",
        "Voice_Message",
        "Voice_Message_Playback",
        "Reply",
        "Forward",
        "Reminder",
        "Starred",
        "Search",
        "Call_Voice",
        "Call_Video",
        "Gallery_Image",
        "Camera",
        "Document",
    }

    if row_a["Component"] in strong_components and row_b["Component"] in strong_components:
        if row_a["Component"] != row_b["Component"]:
            return False

    strong_actions = {"Reply", "Forward", "Playback", "Call", "Search", "Freeze_Open"}
    if row_a["Action"] in strong_actions and row_b["Action"] in strong_actions:
        if row_a["Action"] != row_b["Action"]:
            return False

    if (
        row_a["Sub Feature"] not in {"Other", f'{row_a["Main Feature"]}_General'} and
        row_b["Sub Feature"] not in {"Other", f'{row_b["Main Feature"]}_General'} and
        row_a["Sub Feature"] != row_b["Sub Feature"]
    ):
        return False

    return True


def build_match_reason(match_type: str, row_a: dict, row_b: dict) -> str:
    if match_type == "EXACT":
        return "EXACT_TEXT"
    if match_type == "FUZZY":
        return "FUZZY_SIMILAR_TEXT"
    if row_a["Failure Mode"] == row_b["Failure Mode"] and row_a["Failure Mode"] != "General":
        return f'SEMANTIC_SAME_FAILURE:{row_a["Failure Mode"]}'
    if row_a["Component"] == row_b["Component"] and row_a["Component"] != "General":
        return f'SEMANTIC_SAME_COMPONENT:{row_a["Component"]}'
    return "SEMANTIC_GENERAL"


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    delimiter = st.selectbox("CSV delimiter", [";", ","], index=0)
    fuzzy_threshold = st.slider("Fuzzy threshold", 0.70, 1.00, 0.90, 0.01)
    semantic_threshold = st.slider("Semantic threshold", 0.70, 1.00, 0.84, 0.01)
    use_ai = st.toggle("Enable Semantic AI", value=True)
    best_match_only = st.toggle("Best match only", value=True)
    use_semantic_guardrail = st.toggle("Use domain guardrail", value=True)
    show_debug = st.toggle("Show debug info", value=False)

pool_file = st.file_uploader("POOL CSV", type=["csv"])
target_file = st.file_uploader("TARGET CSV", type=["csv"])
run = st.button("🔍 Find Duplicates")

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

    df["Main Feature"] = df["Summary"].apply(extract_main_feature)
    df["Sub Feature"] = df.apply(lambda r: extract_sub_feature(r["Summary"], r["Main Feature"]), axis=1)
    df["Component"] = df["Summary"].apply(extract_component)
    df["Action"] = df["Summary"].apply(extract_action)
    df["Failure Mode"] = df["Summary"].apply(extract_failure_mode)

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
            "Target Main Feature": t_row["Main Feature"],
            "Target Sub Feature": t_row["Sub Feature"],
            "Target Component": t_row["Component"],
            "Target Action": t_row["Action"],
            "Target Failure Mode": t_row["Failure Mode"],
            "Pool Key": p_row["Issue key"],
            "Pool Summary": p_row["Summary"],
            "Pool Main Feature": p_row["Main Feature"],
            "Pool Sub Feature": p_row["Sub Feature"],
            "Pool Component": p_row["Component"],
            "Pool Action": p_row["Action"],
            "Pool Failure Mode": p_row["Failure Mode"],
            "Type": "EXACT",
            "Score": 1.000,
            "Domain Distance": 0,
            "Match Reason": build_match_reason("EXACT", t_row, p_row),
        })

    # FUZZY
    if not (best_match_only and candidate_rows):
        for _, p in pool.iterrows():
            p_row = p.to_dict()

            if t_row["Summary"] == p_row["Summary"]:
                continue

            f = fuzzy(t_row["norm"], p_row["norm"])
            if f >= fuzzy_threshold:
                dist = domain_distance(t_row, p_row)
                candidate_rows.append({
                    "Target Key": t_row["Issue key"],
                    "Target Summary": t_row["Summary"],
                    "Target Main Feature": t_row["Main Feature"],
                    "Target Sub Feature": t_row["Sub Feature"],
                    "Target Component": t_row["Component"],
                    "Target Action": t_row["Action"],
                    "Target Failure Mode": t_row["Failure Mode"],
                    "Pool Key": p_row["Issue key"],
                    "Pool Summary": p_row["Summary"],
                    "Pool Main Feature": p_row["Main Feature"],
                    "Pool Sub Feature": p_row["Sub Feature"],
                    "Pool Component": p_row["Component"],
                    "Pool Action": p_row["Action"],
                    "Pool Failure Mode": p_row["Failure Mode"],
                    "Type": "FUZZY",
                    "Score": round(float(f), 3),
                    "Domain Distance": dist,
                    "Match Reason": build_match_reason("FUZZY", t_row, p_row),
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

            if use_semantic_guardrail and not semantic_guardrail_ok(t_row, p_row):
                continue

            s = float(semantic_scores[j])
            dist = domain_distance(t_row, p_row)

            candidate_rows.append({
                "Target Key": t_row["Issue key"],
                "Target Summary": t_row["Summary"],
                "Target Main Feature": t_row["Main Feature"],
                "Target Sub Feature": t_row["Sub Feature"],
                "Target Component": t_row["Component"],
                "Target Action": t_row["Action"],
                "Target Failure Mode": t_row["Failure Mode"],
                "Pool Key": p_row["Issue key"],
                "Pool Summary": p_row["Summary"],
                "Pool Main Feature": p_row["Main Feature"],
                "Pool Sub Feature": p_row["Sub Feature"],
                "Pool Component": p_row["Component"],
                "Pool Action": p_row["Action"],
                "Pool Failure Mode": p_row["Failure Mode"],
                "Type": "SEMANTIC",
                "Score": round(s, 3),
                "Domain Distance": dist,
                "Match Reason": build_match_reason("SEMANTIC", t_row, p_row),
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
