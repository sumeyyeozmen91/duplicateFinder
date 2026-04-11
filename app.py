import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Sayfa Ayarları
st.set_page_config(page_title="Jira Bug Hunter AI", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_ai_model():
    # Çok dilli (TR/EN) hızlı model
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Semantik normalizasyon
    text = re.sub(r"\btap\b|\bpress\b|\bselect\b|\btıkla\b|\bseç\b", "click", text)
    text = re.sub(r"\bhata\b|\berror\b|\bfail\b|\bbug\b", "issue", text)
    # Temizlik (Sadece alfanümerik ve Türkçe karakterler)
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🕵️ Jira Bug Deduplicator Pro")
st.markdown("---")

# Sidebar
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])
match_threshold = st.sidebar.slider("Benzerlik Eşiği (%)", 50, 100, 90)

# Ana Ekran
st.subheader("🔍 2. Yeni Bulgu Kontrolü")
col1, col2 = st.columns(2)
with col1:
    u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: Login butonu")
with col2:
    u_desc = st.text_area("Bulgu Açıklaması (Description)", placeholder="Örn: Butona basınca tepki yok")

if uploaded_file and (u_sum or u_desc):
    try:
        # CSV'yi oku ve sütun boşluklarını temizle
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("")
        df.columns = [c.strip() for c in df.columns]
        
        # Sütun İsimlerini Esnek Yakala (TR/EN)
        col_map = {c.lower(): c for c in df.columns}
        sum_col = col_map.get('summary', col_map.get('özet', None))
        desc_col = col_map.get('description', col_map.get('açıklama', None))
        key_col = col_map.get('issue key', col_map.get('key', col_map.get('anahtar', None)))

        if not sum_col:
            st.error("CSV'de 'Summary' veya 'Özet' sütunu bulunamadı!")
            st.stop()

        if st.button("Derin Analizi Başlat"):
            # Kullanıcının sorgusunu birleştir ve temizle
            raw_query = f"{u_sum} {u_desc}".strip()
            norm_query = clean_text(raw_query)
            
            results = []
            
            with st.spinner('Havuz taranıyor...'):
                # Havuz metinlerini hazırla
                pool_texts = []
                for _, row in df.iterrows():
                    txt = str(row[sum_col]) + " " + (str(row[desc_col]) if desc_col else "")
                    pool_texts.append(txt)
                
                # --- AI SEMANTİK ANALİZ ---
                new_emb = model.encode(raw_query, convert_to_tensor=True)
                pool_embs = model.encode(pool_texts, convert_to_tensor=True)
                cosine_scores = util.cos_sim(new_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    p_text_raw = pool_texts[i]
                    p_text_clean = clean_text(p_text_raw)
                    
                    # 1. DOĞRUDAN KELİME İÇERME (Senin istediğin parça eşleşme)
                    # Sorgu metni havuzdaki metnin içinde geçiyor mu?
                    is_contained = norm_query in p_text_clean and len(norm_query) > 2
                    
                    # 2. EXACT-ISH (%90 Benzerlik - Fuzzy)
                    exact_ratio = fuzz.token_set_ratio(norm_query, p_text_clean)
                    
                    # 3. SEMANTİK SKOR (Yapay Zeka Hikaye Benzerliği)
                    semantic_score = float(cosine_scores[i]) * 100

                    # Sonuç Karar Mekanizması
                    if is_contained:
                        results.append({
                            "ID": row[key_col] if key_col else f"Satır {i}",
                            "Yöntem": "İçerik Eşleşmesi (Partial)",
                            "Skor": 100,
                            "Özet": row[sum_col]
                        })
                    elif exact_ratio >= match_threshold:
                        results.append({
                            "ID": row[key_col] if key_col else f"Satır {i}",
                            "Yöntem": f"Benzerlik (%{exact_ratio:.0f})",
                            "Skor": exact_ratio,
                            "Özet": row[sum_col]
                        })
                    elif semantic_score >= 75:
                        results.append({
                            "ID": row[key_col] if key_col else f"Satır {i}",
                            "Yöntem": "Semantic (AI Hikaye)",
                            "Skor": semantic_score,
                            "Özet": row[sum_col]
                        })

            if results:
                st.warning(f"⚠️ Toplam {len(results)} potansiyel çakışma bulundu!")
                # Tekrar edenleri temizle ve skora göre sırala
                res_df = pd.DataFrame(results).drop_duplicates(subset=['ID']).sort_values(by="Skor", ascending=False)
                st.table(res_df[["ID", "Yöntem", "Özet"]])
            else:
                st.success("✅ Benzer bir bulgu bulunamadı.")
                
    except Exception as e:
        st.error(f"Beklenmedik bir hata: {e}")
else:
    st.info("Devam etmek için lütfen Jira CSV havuzunu yükleyin.")
