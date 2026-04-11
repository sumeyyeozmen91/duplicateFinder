import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- Konfigürasyon ---
st.set_page_config(page_title="Jira Bug Hunter AI", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_ai_model():
    # Türkçe/İngilizce hibrit ve kısa metinlerde (Summary) başarılı model
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Sadece harf, rakam ve Türkçe karakterleri koru
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# --- Arayüz ---
st.title("🕵️ Jira Bug Deduplicator Pro")
st.caption("Arama Kapsamı: Sadece **Summary** | Kolon: **Component** Odaklı")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])

st.sidebar.header("⚙️ 2. Hassasiyet Ayarları")
exact_threshold = st.sidebar.slider("Exact Eşiği (%)", 50, 100, 90, 
                                     help="Tam öbek veya harf dizilimi benzerliği.")
semantic_threshold = st.sidebar.slider("Semantic Eşiği (AI)", 30, 100, 65, 
                                        help="Anlamsal benzerlik. Sonuç gelmiyorsa %50-60'a çekin.")

# --- Giriş ---
u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: voice call")

if uploaded_file and u_sum:
    try:
        # CSV Oku
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("N/A")
        df.columns = [c.strip() for c in df.columns]
        
        # Sütun Eşleşmeleri (Dinamik TR/EN)
        c_map = {c.lower(): c for c in df.columns}
        sum_col = c_map.get('summary', c_map.get('özet', None))
        key_col = c_map.get('issue key', c_map.get('key', 'Key'))
        stat_col = c_map.get('status', c_map.get('durum', 'Status'))
        comp_col = c_map.get('component', c_map.get('components', c_map.get('bileşen', None)))

        if not sum_col:
            st.error("Hata: CSV içinde 'Summary' sütunu bulunamadı!")
            st.stop()

        if st.button("Derin Analizi Başlat"):
            full_query = u_sum.strip()
            norm_query = clean_text(full_query)
            # Arama kelimeleri (Bonus puan için)
            search_keywords = [clean_text(w) for w in full_query.split() if len(clean_text(w)) > 2]
            
            exact_results = []
            semantic_results = []
            
            with st.spinner('Analiz ediliyor...'):
                summaries = df[sum_col].astype(str).tolist()
                
                # Semantic Hesaplama
                query_emb = model.encode(full_query, convert_to_tensor=True)
                pool_embs = model.encode(summaries, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    current_sum_raw = str(row[sum_col])
                    current_sum_clean = clean_text(current_sum_raw)
                    
                    # 1. EXACT KONTROLÜ
                    phrase_match = norm_query in current_sum_clean
                    e_score = fuzz.partial_ratio(norm_query, current_sum_clean)
                    
                    # 2. SEMANTIC KONTROLÜ
                    s_score = float(cosine_scores[i]) * 100
                    
                    # --- KELİME BONUSU ---
                    # Aranan kelimelerden biri geçiyorsa Semantic puana +15 ekle
                    keyword_bonus = 15 if any(kw in current_sum_clean for kw in search_keywords) else 0
                    final_semantic_score = min(s_score + keyword_bonus, 100.0)

                    res_obj = {
                        "ID": row.get(key_col, f"S-{i}"),
                        "Status": row.get(stat_col, "N/A"),
                        "Component": row.get(comp_col, "N/A"),
                        "Özet": current_sum_raw
                    }

                    # --- KARAR MEKANİZMASI ---
                    if phrase_match or e_score >= exact_threshold:
                        res_obj["Skor"] = f"%{max(e_score, 100 if phrase_match else 0):.0f}"
                        exact_results.append(res_obj)
                    
                    elif final_semantic_score >= semantic_threshold:
                        res_obj["Skor"] = f"%{final_semantic_score:.0f}"
                        semantic_results.append(res_obj)

            # --- SONUÇ TABLOLARI ---
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.subheader("🎯 Exact Summary Matches")
                if exact_results:
                    st.dataframe(pd.DataFrame(exact_results), use_container_width=True, hide_index=True)
                else:
                    st.info("Tam eşleşme bulunamadı.")

            with col_r:
                st.subheader("🧠 Semantic Matches (AI)")
                if semantic_results:
                    sem_df = pd.DataFrame(semantic_results).sort_values(by="Skor", ascending=False)
                    st.dataframe(sem_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Benzerlik bulunamadı (Eşiği düşürmeyi deneyin).")
                    
    except Exception as e:
        st.error(f"Hata: {e}")
else:
    if not uploaded_file:
        st.info("Lütfen sol menüden Jira CSV dosyasını yükleyin.")
