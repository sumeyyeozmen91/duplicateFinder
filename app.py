import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Jira Bug Hunter AI", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_ai_model():
    # Çok dilli ve kısa metinlerde (Summary) etkili model
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Sadece harf, rakam ve Türkçe karakterleri koru
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# --- Arayüz Başlangıcı ---
st.title("🕵️ Jira Bug Deduplicator Pro")
st.caption("Arama Kapsamı: Sadece **Summary** (Özet) alanı üzerinden hibrit analiz.")
st.markdown("---")

# --- Sidebar: Kontroller ---
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])

st.sidebar.header("⚙️ 2. Hassasiyet Ayarları")
exact_threshold = st.sidebar.slider("Exact Eşiği (%)", 50, 100, 90, 
                                     help="Karakter bazlı tam eşleşme. %90+ nokta atışıdır.")
semantic_threshold = st.sidebar.slider("Semantic Eşiği (AI)", 30, 100, 70, 
                                        help="Yapay zeka benzerliği. Sonuç gelmiyorsa %50-60'a çekin.")

# --- Ana Ekran: Girişler ---
st.subheader("🔍 3. Yeni Bulgu Kontrolü")
u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: voice recording")

if uploaded_file and u_sum:
    try:
        # CSV Oku
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("N/A")
        df.columns = [c.strip() for c in df.columns]
        
        # Sütun Eşleşmeleri
        c_map = {c.lower(): c for c in df.columns}
        sum_col = c_map.get('summary', c_map.get('özet', None))
        key_col = c_map.get('issue key', c_map.get('key', 'Key'))
        stat_col = c_map.get('status', c_map.get('durum', 'Status'))
        plat_col = c_map.get('platform', 'Platform')

        if not sum_col:
            st.error("Hata: CSV içinde 'Summary' veya 'Özet' sütunu bulunamadı!")
            st.stop()

        if st.button("Derin Analizi Başlat"):
            full_query = u_sum.strip()
            norm_query = clean_text(full_query)
            search_keywords = [clean_text(w) for w in full_query.split() if len(clean_text(w)) > 2]
            
            exact_results = []
            semantic_results = []
            
            with st.spinner('Summary havuzu taranıyor...'):
                # Sadece Summary kolonunu listeye al
                summaries = df[sum_col].astype(str).tolist()
                
                # Semantic (AI) Gömme Hesapla
                query_emb = model.encode(full_query, convert_to_tensor=True)
                pool_embs = model.encode(summaries, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    current_sum_raw = str(row[sum_col])
                    current_sum_clean = clean_text(current_sum_raw)
                    
                    # 1. KADEME: EXACT MATCH (Karakter Bazlı)
                    # ratio ve partial_ratio birleşimi ile en yüksek karakter puanını al
                    e_score = fuzz.partial_ratio(norm_query, current_sum_clean)
                    phrase_match = norm_query in current_sum_clean
                    
                    # 2. KADEME: SEMANTIC MATCH (AI Bazlı)
                    s_score = float(cosine_scores[i]) * 100

                    # 3. KADEME: KELİME VARLIĞI (Arama esnekliği için)
                    # Girdiğin kelimelerden en az biri geçiyor mu?
                    any_word_match = any(kw in current_sum_clean for kw in search_keywords)

                    res_obj = {
                        "ID": row.get(key_col, f"S-{i}"),
                        "Status": row.get(stat_col, "N/A"),
                        "Platform": row.get(plat_col, "N/A"),
                        "Özet": current_sum_raw
                    }

                    # --- KARAR MEKANİZMASI ---
                    
                    # A. Exact Listesine Ekleme (Tam öbek geçiyorsa veya %90+ ise)
                    if phrase_match or e_score >= exact_threshold:
                        res_obj["Skor"] = f"%{max(e_score, 100 if phrase_match else 0):.0f}"
                        exact_results.append(res_obj)
                    
                    # B. Semantic Listesine Ekleme (AI Skoru yüksekse)
                    # Kelime eşleşmesi varsa (any_word_match) AI eşiğini otomatik 5 puan düşürür
                    elif s_score >= (semantic_threshold - (5 if any_word_match else 0)):
                        res_obj["Skor"] = f"%{s_score:.0f}"
                        semantic_results.append(res_obj)

            # --- İKİ KOLONLU GÖSTERİM ---
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("🎯 Exact Summary Matches")
                st.caption(f"Özetinde doğrudan '{full_query}' geçen veya çok yakın olanlar")
                if exact_results:
                    st.dataframe(pd.DataFrame(exact_results), use_container_width=True, hide_index=True)
                else:
                    st.info("Karakter bazlı tam eşleşme bulunamadı.")

            with col_right:
                st.subheader("🧠 Semantic Summary Matches")
                st.caption(f"Anlamsal olarak benzer hikayeler (Eşik: %{semantic_threshold})")
                if semantic_results:
                    # Skora göre sırala
                    sem_df = pd.DataFrame(semantic_results).sort_values(by="Skor", ascending=False)
                    st.dataframe(sem_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Anlamsal olarak yakın bir özet bulunamadı.")
                    
    except Exception as e:
        st.error(f"Sistemsel Hata: {e}")
else:
    if not uploaded_file:
        st.info("Devam etmek için lütfen sol menüden Jira CSV dosyasını yükleyin.")
