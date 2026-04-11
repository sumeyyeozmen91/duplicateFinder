import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz # %90 benzerlik için
from sentence_transformers import SentenceTransformer, util # Semantik analiz için

# Sayfa Ayarları
st.set_page_config(page_title="Advanced Bug Deduplicator", page_icon="🧠", layout="wide")

# Modeli Önbelleğe Al (Her seferinde tekrar yüklenmesin)
@st.cache_resource
def load_model():
    # Çok dilli ve hafif bir model (Türkçe desteği için ideal)
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

def normalize_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🧠 Advanced Bug Deduplicator")
st.markdown("### Exact (%90 Similarity) + AI Semantic Analysis")

# Sidebar
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Jira CSV", type=["csv"])
similarity_threshold = st.sidebar.slider("Exact Match Similarity Threshold (%)", 0, 100, 90)

# Input
st.subheader("🔍 New Bug Entry")
u_sum = st.text_input("Summary")
u_desc = st.text_area("Description")

if uploaded_file and (u_sum or u_desc):
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("")
        df.columns = [c.strip() for c in df.columns]
        
        if st.button("Deep Search"):
            matches = []
            new_text = f"{u_sum} {u_desc}"
            norm_new = normalize_text(new_text)

            with st.spinner('Analiz ediliyor...'):
                # 1. HAVUZU HAZIRLA
                # Tüm havuzun summary+desc birleşimini alıyoruz
                pool_texts = (df['Summary'] + " " + df['Description']).tolist()
                
                # 2. SEMANTIC ANALİZ (AI Hikayeleştirme)
                new_embedding = model.encode(new_text, convert_to_tensor=True)
                pool_embeddings = model.encode(pool_texts, convert_to_tensor=True)
                cosine_scores = util.cos_sim(new_embedding, pool_embeddings)[0]

                # 3. DÖNGÜ VE KONTROL
                for i, row in df.iterrows():
                    p_key = str(row.get('Issue key', row.get('Key', 'N/A')))
                    p_text_raw = pool_texts[i]
                    p_text_norm = normalize_text(p_text_raw)
                    
                    # Exact Match Check (%90 Benzerlik)
                    ratio = fuzz.token_set_ratio(norm_new, p_text_norm)
                    
                    # Semantic Score
                    sem_score = float(cosine_scores[i])

                    if ratio >= similarity_threshold:
                        matches.append({
                            "Key": p_key,
                            "Method": f"Exact (%{ratio:.0f})",
                            "Match Score": ratio / 100,
                            "Summary": row['Summary']
                        })
                    elif sem_score > 0.75: # Anlamsal eşik
                        matches.append({
                            "Key": p_key,
                            "Method": "AI Semantic",
                            "Match Score": sem_score,
                            "Summary": row['Summary']
                        })

            # Sonuçları Göster
            if matches:
                # Skora göre sırala
                match_df = pd.DataFrame(matches).sort_values(by="Match Score", ascending=False)
                st.warning(f"Found {len(matches)} potential duplicates!")
                st.dataframe(match_df[["Key", "Method", "Summary"]], use_container_width=True)
            else:
                st.success("No duplicates found. You're good to go!")
                
    except Exception as e:
        st.error(f"Error: {e}")
