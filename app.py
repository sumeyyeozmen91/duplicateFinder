import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Sayfa Ayarları
st.set_page_config(page_title="Jira Bug Hunter Pro", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_ai_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🕵️ Jira Bug Deduplicator Pro")
st.markdown("Status ve Platform bilgileriyle zenginleştirilmiş akıllı arama motoru.")

# Sidebar
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])
exact_threshold = st.sidebar.slider("Exact Benzerlik Eşiği (%)", 30, 100, 80)
semantic_threshold = st.sidebar.slider("Semantic Benzerlik Eşiği (%)", 30, 100, 75)

# Giriş Alanları
st.subheader("🔍 2. Yeni Bulgu Kontrolü")
col_in1, col_in2 = st.columns(2)
with col_in1:
    u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: kişi numara")
with col_in2:
    u_desc = st.text_area("Bulgu Açıklaması (Description)", placeholder="Örn: rehber")

if uploaded_file and (u_sum or u_desc):
    try:
        # CSV Oku
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("Belirtilmemiş")
        df.columns = [c.strip() for c in df.columns]
        
        # Dinamik Kolon Eşleme (TR/EN)
        c_map = {c.lower(): c for c in df.columns}
        sum_col = c_map.get('summary', c_map.get('özet', None))
        desc_col = c_map.get('description', c_map.get('açıklama', None))
        key_col = c_map.get('issue key', c_map.get('key', c_map.get('anahtar', None)))
        stat_col = c_map.get('status', c_map.get('durum', None))
        plat_col = c_map.get('platform', c_map.get('işletim sistemi', None))

        if st.button("Derin Analizi Başlat"):
            raw_query = f"{u_sum} {u_desc}".strip()
            norm_query = clean_text(raw_query)
            search_keywords = [clean_text(w) for w in raw_query.split() if len(clean_text(w)) > 2]
            
            results = []
            
            with st.spinner('Kayıtlar taranıyor...'):
                pool_texts = []
                for _, row in df.iterrows():
                    pool_texts.append(str(row[sum_col]) + " " + (str(row[desc_col]) if desc_col else ""))
                
                # Semantic (AI) Analiz
                query_emb = model.encode(raw_query, convert_to_tensor=True)
                pool_embs = model.encode(pool_texts, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    p_text_raw = pool_texts[i]
                    p_text_clean = clean_text(p_text_raw)
                    
                    # 1. KELİME BAZLI (Contains)
                    matched_kws = [kw for kw in search_keywords if kw in p_text_clean]
                    
                    # 2. EXACT BENZERLİK
                    exact_score = fuzz.token_set_ratio(norm_query, p_text_clean)
                    
                    # 3. SEMANTIC
                    sem_score = float(cosine_scores[i]) * 100

                    match_type = None
                    final_score = 0
                    
                    if matched_kws:
                        match_type = "Kelime (Contains)"
                        final_score = 100
                    elif exact_score >= exact_threshold:
                        match_type = f"Exact (%{exact_score:.0f})"
                        final_score = exact_score
                    elif sem_score >= semantic_threshold:
                        match_type = "Semantic (AI)"
                        final_score = sem_score

                    if match_type:
                        results.append({
                            "ID": row[key_col] if key_col else f"S-{i}",
                            "Yöntem": match_type,
                            "Özet": row[sum_col],
                            "Status": row[stat_col] if stat_col else "N/A",
                            "Platform": row[plat_col] if plat_col else "N/A",
                            "Skor": final_score
                        })

            if results:
                st.warning(f"⚠️ {len(results)} benzer kayıt bulundu!")
                res_df = pd.DataFrame(results).drop_duplicates(subset=['ID']).sort_values(by="Skor", ascending=False)
                # Tabloyu daha şık gösterelim
                st.dataframe(
                    res_df[["ID", "Yöntem", "Status", "Platform", "Özet"]], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ Benzer bir bulgu bulunamadı.")
                
    except Exception as e:
        st.error(f"Sütun okuma hatası: {e}")
