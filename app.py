import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Jira Bug Hunter AI", page_icon="🕵️", layout="wide")

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

# Sidebar
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])

st.sidebar.header("⚙️ 2. Hassasiyet Ayarları")
exact_threshold = st.sidebar.slider("Exact Eşiği (%90+ Önerilir)", 50, 100, 90)
semantic_threshold = st.sidebar.slider("Semantic Eşiği (AI)", 50, 100, 75)

# Girişler
st.subheader("🔍 3. Yeni Bulgu Kontrolü")
u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: voice recording")
u_desc = st.text_area("Bulgu Açıklaması (Description)")

if uploaded_file and (u_sum or u_desc):
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("N/A")
        df.columns = [c.strip() for c in df.columns]
        
        c_map = {c.lower(): c for c in df.columns}
        sum_col, desc_col = c_map.get('summary', 'Summary'), c_map.get('description', 'Description')
        key_col, stat_col, plat_col = c_map.get('issue key', 'Key'), c_map.get('status', 'Status'), c_map.get('platform', 'Platform')

        if st.button("Derin Analizi Başlat"):
            full_query = f"{u_sum} {u_desc}".strip()
            norm_query = clean_text(full_query)
            
            exact_results = []
            semantic_results = []
            
            with st.spinner('Hibrit analiz yapılıyor...'):
                pool_texts = [(str(row[sum_col]) + " " + (str(row[desc_col]) if desc_col else "")) for _, row in df.iterrows()]
                
                # AI Semantic Hesaplama
                query_emb = model.encode(full_query, convert_to_tensor=True)
                pool_embs = model.encode(pool_texts, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    p_text_raw = pool_texts[i]
                    p_text_clean = clean_text(p_text_raw)
                    
                    # 1. KADEME: EXACT KONTROLÜ (Sert Filtre)
                    # ratio: Karakter karakter benzerlik (voice recording != voice message)
                    e_score = fuzz.partial_ratio(norm_query, p_text_clean)
                    # Öbek kontrolü (Zorunlu İçerme)
                    phrase_match = norm_query in p_text_clean

                    # 2. KADEME: SEMANTIC KONTROLÜ
                    s_score = float(cosine_scores[i]) * 100

                    res_obj = {
                        "ID": row[key_col], "Status": row[stat_col], 
                        "Platform": row[plat_col], "Özet": row[sum_col]
                    }

                    # Karar: Önce Exact Match (Öbek geçiyorsa veya %90+ ise)
                    if phrase_match or e_score >= exact_threshold:
                        res_obj["Skor"] = f"%{max(e_score, 100 if phrase_match else 0):.0f}"
                        exact_results.append(res_obj)
                    
                    # Eğer Exact değilse ama Semantic eşiği geçiyorsa
                    elif s_score >= semantic_threshold:
                        res_obj["Skor"] = f"%{s_score:.0f}"
                        semantic_results.append(res_obj)

            # SONUÇLARI AYRI LİSTELERDE GÖSTER
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.subheader("🎯 Exact Matches")
                st.caption(f"İçinde '{full_query}' geçen veya %{exact_threshold}+ benzer olanlar")
                if exact_results:
                    st.dataframe(pd.DataFrame(exact_results), use_container_width=True, hide_index=True)
                else:
                    st.info("Birebir eşleşme bulunamadı.")

            with col_res2:
                st.subheader("🧠 Semantic Matches (AI)")
                st.caption(f"Anlamsal olarak benzeyenler (Eşik: %{semantic_threshold})")
                if semantic_results:
                    st.dataframe(pd.DataFrame(semantic_results), use_container_width=True, hide_index=True)
                else:
                    st.info("Anlamsal benzerlik bulunamadı.")
                    
    except Exception as e:
        st.error(f"Hata: {e}")
