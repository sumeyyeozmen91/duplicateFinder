import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Sayfa Yapılandırması
st.set_page_config(page_title="Jira Bug Hunter AI", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_ai_model():
    # Türkçe/İngilizce uyumlu hafif ve güçlü model
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Semantik normalizasyon (eylem eşleme)
    text = re.sub(r"\btap\b|\bpress\b|\bselect\b|\btıkla\b|\bseç\b", "click", text)
    # Temizlik (Özel karakterleri at, boşlukları düzenle)
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🕵️ Jira Bug Deduplicator Pro")
st.markdown("---")

# Sidebar
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])
exact_threshold = st.sidebar.slider("Exact Benzerlik Eşiği (%)", 50, 100, 80)
semantic_threshold = st.sidebar.slider("Semantic Benzerlik Eşiği (%)", 50, 100, 75)

# Ana Ekran Girişleri
st.subheader("🔍 2. Yeni Bulgu Kontrolü")
col1, col2 = st.columns(2)
with col1:
    u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Arama kelimeleri veya özet...")
with col2:
    u_desc = st.text_area("Bulgu Açıklaması (Description)", placeholder="Hata detayları...")

if uploaded_file and (u_sum or u_desc):
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("")
        df.columns = [c.strip() for c in df.columns]
        
        # Dinamik Kolon Eşleme
        col_map = {c.lower(): c for c in df.columns}
        sum_col = col_map.get('summary', col_map.get('özet', None))
        desc_col = col_map.get('description', col_map.get('açıklama', None))
        key_col = col_map.get('issue key', col_map.get('key', col_map.get('anahtar', None)))

        if st.button("Derin Analizi Başlat"):
            # Kullanıcı girdilerini hazırla
            raw_query = f"{u_sum} {u_desc}".strip()
            norm_query = clean_text(raw_query)
            # Kelime bazlı arama için parçala
            search_keywords = [clean_text(w) for w in raw_query.split() if len(clean_text(w)) > 2]
            
            results = []
            
            with st.spinner('Hibrit motor (Kelime + Exact + AI) çalışıyor...'):
                # Havuz metinlerini hazırla
                pool_texts = []
                for _, row in df.iterrows():
                    pool_texts.append(str(row[sum_col]) + " " + (str(row[desc_col]) if desc_col else ""))
                
                # Semantic Embedding Hesapla (AI)
                query_emb = model.encode(raw_query, convert_to_tensor=True)
                pool_embs = model.encode(pool_texts, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    p_text_raw = pool_texts[i]
                    p_text_clean = clean_text(p_text_raw)
                    
                    # 1. KADEME: KELİME BAZLI (Contains / Equals)
                    # Girdiğin kelimelerden herhangi biri havuzdaki metnin içinde aynen geçiyor mu?
                    matched_kws = [kw for kw in search_keywords if kw in p_text_clean]
                    
                    # 2. KADEME: EXACT BENZERLİK (%80 Yaklaşık)
                    # Kelime sırasına bakmadan metin bütünlüğünü kontrol eder (Fuzzy)
                    exact_score = fuzz.token_set_ratio(norm_query, p_text_clean)
                    
                    # 3. KADEME: SEMANTIC (Hikaye Benzerliği)
                    # AI, kelimeler farklı olsa da "anlam" benzerliğine bakar
                    sem_score = float(cosine_scores[i]) * 100

                    # Filtreleme ve Etiketleme
                    if matched_kws:
                        results.append({
                            "ID": row[key_col] if key_col else f"Row {i}",
                            "Yöntem": "Kelime (Contains) ✅",
                            "Skor": 100,
                            "Detay": f"Eşleşenler: {', '.join(matched_kws)}",
                            "Özet": row[sum_col]
                        })
                    elif exact_score >= exact_threshold:
                        results.append({
                            "ID": row[key_col] if key_col else f"Row {i}",
                            "Yöntem": f"Exact (%{exact_score:.0f}) 🎯",
                            "Skor": exact_score,
                            "Detay": "Metinler yaklaşık olarak aynı",
                            "Özet": row[sum_col]
                        })
                    elif sem_score >= semantic_threshold:
                        results.append({
                            "ID": row[key_col] if key_col else f"Row {i}",
                            "Yöntem": "Semantic (AI) 🧠",
                            "Skor": sem_score,
                            "Detay": "Hikaye/Anlam benzerliği yüksek",
                            "Özet": row[sum_col]
                        })

            if results:
                st.warning(f"⚠️ Toplam {len(results)} benzer kayıt bulundu!")
                res_df = pd.DataFrame(results).drop_duplicates(subset=['ID']).sort_values(by="Skor", ascending=False)
                st.dataframe(res_df[["ID", "Yöntem", "Detay", "Özet"]], use_container_width=True)
            else:
                st.success("✅ Benzer bir bulgu bulunamadı.")
                
    except Exception as e:
        st.error(f"Hata: {e}")
