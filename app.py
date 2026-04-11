import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Sayfa Genişliği ve Başlık
st.set_page_config(page_title="Bug Hunter AI", page_icon="🕵️", layout="wide")

# AI Modeli Önbelleğe Al (Hız için)
@st.cache_resource
def load_ai_model():
    # Çok dilli (TR/EN) ve hızlı model
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_ai_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Semantik normalizasyon (senin mantığın)
    text = re.sub(r"\btap\b|\bpress\b|\bselect\b|\btıkla\b|\bseç\b", "click", text)
    text = re.sub(r"\bhata\b|\berror\b|\bfail\b|\bbug\b", "issue", text)
    # Temizlik
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🕵️ Jira Bug Deduplicator Pro")
st.markdown("---")

# Sidebar: Ayarlar ve Dosya Yükleme
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])
match_threshold = st.sidebar.slider("Exact Similarity Threshold (%)", 70, 100, 90)

# Ana Ekran: Giriş Alanları
st.subheader("🔍 2. Yeni Bulgu Kontrolü")
col1, col2 = st.columns(2)
with col1:
    u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: Buton çalışmıyor")
with col2:
    u_desc = st.text_area("Bulgu Açıklaması (Description)", placeholder="Örn: Sayfada hata kodu alıyorum")

if uploaded_file and (u_sum or u_desc):
    try:
        # CSV'yi oku ve sütun boşluklarını temizle
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("")
        df.columns = [c.strip() for c in df.columns]
        
        # Sütun İsimlerini Esnek Yakala (TR/EN Uyumlu)
        col_map = {c.lower(): c for c in df.columns}
        sum_col = col_map.get('summary', col_map.get('özet', None))
        desc_col = col_map.get('description', col_map.get('açıklama', None))
        key_col = col_map.get('issue key', col_map.get('key', col_map.get('anahtar', None)))

        if not sum_col:
            st.error("CSV'de 'Summary' veya 'Özet' sütunu bulunamadı!")
            st.stop()

        if st.button("Derin Analizi Başlat"):
            new_entry = f"{u_sum} {u_desc}"
            norm_new = clean_text(new_entry)
            
            results = []
            
            with st.spinner('Yapay zeka hikayeleri karşılaştırıyor...'):
                # Havuz metinlerini hazırla (Summary + Description)
                pool_texts = []
                for _, row in df.iterrows():
                    txt = str(row[sum_col]) + " " + (str(row[desc_col]) if desc_col else "")
                    pool_texts.append(txt)
                
                # --- SEMANTİK ANALİZ (AI) ---
                new_emb = model.encode(new_entry, convert_to_tensor=True)
                pool_embs = model.encode(pool_texts, convert_to_tensor=True)
                cosine_scores = util.cos_sim(new_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    p_text_raw = pool_texts[i]
                    p_text_norm = clean_text(p_text_raw)
                    
                    # --- %90 BENZERLİK (Fuzzy Exact) ---
                    exact_ratio = fuzz.token_set_ratio(norm_new, p_text_norm)
                    
                    # --- SEMANTİK SKOR ---
                    semantic_score = float(cosine_scores[i]) * 100

                    # Sonuçları Filtrele
                    if exact_ratio >= match_threshold:
                        results.append({
                            "ID": row[key_col] if key_col else f"Satır {i}",
                            "Yöntem": f"Exact (%{exact_ratio:.0f})",
                            "Skor": exact_ratio,
                            "Özet": row[sum_col]
                        })
                    elif semantic_score >= 75: # Hikaye benzerliği eşiği
                        results.append({
                            "ID": row[key_col] if key_col else f"Satır {i}",
                            "Yöntem": "Semantic (AI)",
                            "Skor": semantic_score,
                            "Özet": row[sum_col]
                        })

            if results:
                st.warning(f"⚠️ Toplam {len(results)} potansiyel çakışma bulundu!")
                res_df = pd.DataFrame(results).sort_values(by="Skor", ascending=False)
                st.table(res_df[["ID", "Yöntem", "Özet"]])
            else:
                st.success("✅ Benzer bir bulgu bulunamadı, güvenle açabilirsiniz.")
                
    except Exception as e:
        st.error(f"Beklenmedik bir hata: {e}")
else:
    st.info("Lütfen sol menüden Jira CSV havuzunu yükleyin.")
