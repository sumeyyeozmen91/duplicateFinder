import streamlit as st
import pandas as pd
import re

# Sayfa yapılandırması
st.set_page_config(page_title="Jira Bug Deduplicator", page_icon="🐞", layout="wide")

def normalize_and_semantic(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Temizlik: Alfanümerik karakterler dışındakileri at
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    # Semantik Normalizasyon
    text = re.sub(r"\btap\b|\bpress\b|\bselect\b|\btıkla\b|\bseç\b", "click", text)
    text = re.sub(r"\bhata\b|\berror\b|\bfail\b|\bbug\b|\bkusur\b", "issue", text)
    return re.sub(r"\s+", " ", text).strip()

st.title("🐞 Jira Bug Deduplicator")
st.markdown("""
Bu araç, yeni bir bulgu (bug) açmadan önce elinizdeki **Jira havuzunu (CSV)** tarar. 
**Exact (Birebir)** ve **Semantic (Anlamsal)** benzerlikleri bularak mükerrer kayıt açmanızı engeller.
""")

# Sidebar - Dosya Yükleme
st.sidebar.header("📂 Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Jira Export CSV dosyasını yükleyin", type=["csv"])

# Ana Panel - Giriş Alanları
st.subheader("🔍 Yeni Bulgu Kontrolü")
col1, col2 = st.columns(2)
with col1:
    user_summary = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: Uygulama ana ekranda donuyor")
with col2:
    user_desc = st.text_area("Bulgu Açıklaması (Description)", placeholder="Örn: Login olduktan sonra yükleme ikonu gitmiyor...")

if uploaded_file:
    # CSV'yi oku
    try:
        # Jira bazen farklı ayırıcılar kullanabilir, sep=None otomatik algılar
        df_pool = pd.read_csv(uploaded_file, sep=None, engine="python")
        df_pool.columns = [c.strip() for c in df_pool.columns] # Kolon boşluklarını temizle
        
        if st.button("Benzerlikleri Kontrol Et"):
            if not user_summary:
                st.warning("Lütfen kontrol için en azından bir özet (Summary) girin.")
            else:
                norm_user_sum = normalize_and_semantic(user_summary)
                norm_user_desc = normalize_and_semantic(user_desc)
                
                matches = []
                
                with st.spinner('Havuz taranıyor...'):
                    for _, row in df_pool.iterrows():
                        p_sum_raw = str(row.get('Summary', ''))
                        p_desc_raw = str(row.get('Description', ''))
                        p_key = str(row.get('Issue key', row.get('Key', 'N/A')))
                        
                        p_sum_norm = normalize_and_semantic(p_sum_raw)
                        p_desc_norm = normalize_and_semantic(p_desc_raw)
                        
                        # Benzerlik Mantığı
                        is_exact = norm_user_sum == p_sum_norm
                        is_semantic = (len(norm_user_sum) > 5 and norm_user_sum in p_sum_norm) or \
                                      (len(norm_user_desc) > 10 and norm_user_desc in p_desc_norm)
                        
                        if is_exact:
                            matches.append({"Key": p_key, "Tip": "EXACT ✅", "Özet": p_sum_raw})
                        elif is_semantic:
                            matches.append({"Key": p_key, "Tip": "SEMANTIC 🧠", "Özet": p_sum_raw})

                # Sonuçları Göster
                if matches:
                    st.warning(f"Toplam {len(matches)} benzer kayıt bulundu!")
                    st.dataframe(pd.DataFrame(matches), use_container_width=True)
                else:
                    st.success("Benzer bir bulgu bulunamadı. Yeni kayıt açmak için güvenli!")
                    
    except Exception as e:
        st.error(f"Dosya okunurken bir hata oluştu: {e}")
else:
    st.info("Devam etmek için lütfen sol menüden Jira havuzunuzu içeren CSV dosyasını yükleyin.")
