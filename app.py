import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- Konfigürasyon ---
st.set_page_config(page_title="Jira Bug Hunter AI", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_ai_model():
    # Çok dilli ve Summary gibi kısa metinlerde başarılı model
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🕵️ Jira Bug Deduplicator Pro")
st.caption("Arama odağı: Sadece Summary (Özet) alanı")

# --- Sidebar ---
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])

st.sidebar.header("⚙️ 2. Hassasiyet Ayarları")
exact_threshold = st.sidebar.slider("Exact Eşiği (%)", 50, 100, 90, help="Karakter bazlı tam eşleşme hassasiyeti.")
semantic_threshold = st.sidebar.slider("Semantic Eşiği (AI)", 50, 100, 75, help="Anlamsal (hikaye) benzerlik hassasiyeti.")

# --- Giriş Alanı ---
st.subheader("🔍 3. Yeni Bulgu Kontrolü")
# Sadece Summary girişi alıyoruz
u_sum = st.text_input("Aranacak Özet (Summary)", placeholder="Örn: voice recording")

if uploaded_file and u_sum:
    try:
        # CSV Oku
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("N/A")
        df.columns = [c.strip() for c in df.columns]
        
        # Sütunları Eşle
        c_map = {c.lower(): c for c in df.columns}
        sum_col = c_map.get('summary', c_map.get('özet', None))
        key_col = c_map.get('issue key', c_map.get('key', 'Key'))
        stat_col = c_map.get('status', c_map.get('durum', 'Status'))
        plat_col = c_map.get('platform', 'Platform')

        if not sum_col:
            st.error("CSV'de 'Summary' sütunu bulunamadı!")
            st.stop()

        if st.button("Derin Analizi Başlat"):
            norm_query = clean_text(u_sum)
            
            exact_results = []
            semantic_results = []
            
            with st.spinner('Sadece Özetler analiz ediliyor...'):
                # Sadece Summary kolonundaki verileri alıyoruz
                summaries = df[sum_col].astype(str).tolist()
                
                # Semantic (AI) Hesaplama
                query_emb = model.encode(u_sum, convert_to_tensor=True)
                pool_embs = model.encode(summaries, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    current_sum_raw = str(row[sum_col])
                    current_sum_clean = clean_text(current_sum_raw)
                    
                    # 1. EXACT KONTROLÜ
                    # partial_ratio: "voice recording" öbeği Summary içinde geçiyor mu?
                    e_score = fuzz.partial_ratio(norm_query, current_sum_clean)
                    phrase_match = norm_query in current_sum_clean

                    # 2. SEMANTIC KONTROLÜ
                    s_score = float(cosine_scores[i]) * 100

                    res_obj = {
                        "ID": row.get(key_col, f"S-{i}"),
                        "Status": row.get(stat_col, "N/A"),
                        "Platform": row.get(plat_col, "N/A"),
                        "Özet": current_sum_raw
                    }

                    # Karar Mekanizması
                    if phrase_match or e_score >= exact_threshold:
                        res_obj["Skor"] = f"%{max(e_score, 100 if phrase_match else 0):.0f}"
                        exact_results.append(res_obj)
                    elif s_score >= semantic_threshold:
                        res_obj["Skor"] = f"%{s_score:.0f}"
                        semantic_results.append(res_obj)

            # --- İKİ KOLONLU GÖSTERİM ---
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.subheader("🎯 Exact Summary Matches")
                st.caption(f"Özetinde doğrudan '{u_sum}' geçenler")
                if exact_results:
                    st.dataframe(pd.DataFrame(exact_results), use_container_width=True, hide_index=True)
                else:
                    st.info("Birebir eşleşen bir özet bulunamadı.")

            with col_res2:
                st.subheader("🧠 Semantic Summary Matches")
                st.caption("Özeti anlamsal olarak benzeyenler")
                if semantic_results:
                    st.dataframe(pd.DataFrame(semantic_results), use_container_width=True, hide_index=True)
                else:
                    st.info("Benzer anlamda bir özet bulunamadı.")
                    
    except Exception as e:
        st.error(f"Beklenmedik bir hata oluştu: {e}")
else:
    if not uploaded_file:
        st.info("Lütfen sol menüden Jira CSV dosyasını yükleyin.")
