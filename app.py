import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Jira Bug Hunter Pro", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_ai_model():
    # Türkçe/İngilizce uyumlu akıllı model
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
st.markdown("---")

# --- Sidebar: Ayarlar ---
st.sidebar.header("📂 1. Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira CSV Export Yükle", type=["csv"])

st.sidebar.header("⚙️ 2. Filtre Ayarları")
strict_mode = st.sidebar.toggle("Strict Mode (AND)", value=True, help="Açık olduğunda, girdiğiniz TÜM kelimelerin havuzda geçmesini zorunlu kılar.")
exact_threshold = st.sidebar.slider("Exact Benzerlik Eşiği (%)", 30, 100, 80, help="Metinlerin karakter bazlı benzerliğini ölçer.")
semantic_threshold = st.sidebar.slider("Semantic Benzerlik Eşiği (%)", 30, 100, 85, help="Yapay zeka ile anlam benzerliğini ölçer.")

# --- Ana Ekran: Girişler ---
st.subheader("🔍 3. Yeni Bulgu Kontrolü")
col_in1, col_in2 = st.columns(2)
with col_in1:
    u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: voice")
with col_in2:
    u_desc = st.text_area("Bulgu Açıklaması (Description)", placeholder="Örn: emoji")

# --- İşlem Mantığı ---
if uploaded_file and (u_sum or u_desc):
    try:
        # CSV Oku (Boş değerleri 'N/A' ile doldur)
        df = pd.read_csv(uploaded_file, sep=None, engine="python").fillna("N/A")
        df.columns = [c.strip() for c in df.columns]
        
        # Sütun İsimlerini Yakala
        c_map = {c.lower(): c for c in df.columns}
        sum_col = c_map.get('summary', c_map.get('özet', None))
        desc_col = c_map.get('description', c_map.get('açıklama', None))
        key_col = c_map.get('issue key', c_map.get('key', c_map.get('anahtar', None)))
        stat_col = c_map.get('status', c_map.get('durum', None))
        plat_col = c_map.get('platform', c_map.get('işletim sistemi', None))

        if not sum_col:
            st.error("Hata: CSV içinde 'Summary' veya 'Özet' sütunu bulunamadı!")
            st.stop()

        if st.button("Derin Analizi Başlat"):
            # Arama sorgusunu hazırla
            full_query = f"{u_sum} {u_desc}".strip()
            norm_query = clean_text(full_query)
            search_keywords = [clean_text(w) for w in full_query.split() if len(clean_text(w)) > 2]
            
            results = []
            
            with st.spinner('AI motoru ve kelime filtreleri çalışıyor...'):
                # Havuz metinlerini birleştir
                pool_texts = []
                for _, row in df.iterrows():
                    pool_texts.append(str(row[sum_col]) + " " + (str(row[desc_col]) if desc_col else ""))
                
                # Semantic (AI) Embedding Hesapla
                query_emb = model.encode(full_query, convert_to_tensor=True)
                pool_embs = model.encode(pool_texts, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_emb, pool_embs)[0]

                for i, row in df.iterrows():
                    p_text_raw = pool_texts[i]
                    p_text_clean = clean_text(p_text_raw)
                    
                    # 1. KELİME KONTROLÜ (AND Mantığı)
                    matched_kws = [kw for kw in search_keywords if kw in p_text_clean]
                    all_keywords_present = len(matched_kws) == len(search_keywords)
                    
                    # 2. SKOR HESAPLAMALARI
                    # fuzzy.token_set_ratio kelime sırası karışık olsa da benzerliği bulur
                    exact_score = fuzz.token_set_ratio(norm_query, p_text_clean)
                    sem_score = float(cosine_scores[i]) * 100

                    # --- FİLTRELEME KARAR MEKANİZMASI ---
                    
                    # Strict Mode aktifse ve tüm kelimeler yoksa bu kaydı atla
                    if strict_mode and not all_keywords_present:
                        continue
                    
                    match_type = None
                    display_score = 0
                    
                    # Kriter 1: Exact Benzerlik (Slider'a bağlı)
                    if exact_score >= exact_threshold:
                        match_type = f"Exact Match (%{exact_score:.0f})"
                        display_score = exact_score
                    
                    # Kriter 2: Semantic Benzerlik (Slider'a bağlı)
                    elif sem_score >= semantic_threshold:
                        match_type = f"Semantic AI (%{sem_score:.0f})"
                        display_score = sem_score
                    
                    # Eğer slider eşiklerinden biri geçildiyse sonuçlara ekle
                    if match_type:
                        results.append({
                            "ID": row[key_col] if key_col else f"S-{i}",
                            "Yöntem": match_type,
                            "Status": row[stat_col] if stat_col else "N/A",
                            "Platform": row[plat_col] if plat_col else "N/A",
                            "Özet": row[sum_col],
                            "Skor": display_score
                        })

            # --- Sonuç Gösterimi ---
            if results:
                st.warning(f"⚠️ {len(results)} potansiyel çakışma bulundu.")
                res_df = pd.DataFrame(results).drop_duplicates(subset=['ID']).sort_values(by="Skor", ascending=False)
                st.dataframe(
                    res_df[["ID", "Yöntem", "Status", "Platform", "Özet"]], 
                    use_container_width=True, 
                    hide_index=True
                )
            else:
                st.success("✅ Belirtilen eşik değerlerinde benzer bir bulgu bulunamadı.")
                
    except Exception as e:
        st.error(f"Sistemsel Hata: {e}")
else:
    st.info("Devam etmek için lütfen sol menüden Jira CSV dosyasını yükleyin.")
