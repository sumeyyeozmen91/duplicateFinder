import streamlit as st
import pandas as pd
import re
import io
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Jira Bug AI Hub", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_model():
    # Çok dilli model: Türkçe ve İngilizce anlam eşlemesinde en yüksek başarıya sahip model
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Sadece harf, rakam ve Türkçe karakterleri koruyarak temizle
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_col_names(df):
    """CSV sütunlarını dinamik olarak eşleştirir."""
    c_map = {c.lower().strip(): c for c in df.columns}
    return {
        "summary": c_map.get('summary', c_map.get('özet', None)),
        "key": c_map.get('issue key', c_map.get('key', c_map.get('anahtar', None))),
        "comp": c_map.get('component', c_map.get('components', c_map.get('bileşen', None))),
        "status": c_map.get('status', c_map.get('durum', None))
    }

# --- Yan Menü (Navigasyon) ---
st.sidebar.title("🚀 Jira AI Hub")
page = st.sidebar.radio("İşlem Modu Seçiniz:", ["Tekli Arama (Manuel Kontrol)", "Toplu Analiz (Bulk Match)"])

# --- MOD 1: TEKLİ ARAMA (MANUEL) ---
if page == "Tekli Arama (Manuel Kontrol)":
    st.title("🔍 Tekli Bulgu Kontrolü")
    st.caption("Yapay zeka kelimelere takılmadan sadece ANLAMA odaklanır.")
    
    master_file = st.sidebar.file_uploader("Ana Havuz CSV (Referans)", type=["csv"], key="single_master")
    exact_threshold = st.sidebar.slider("Exact Eşiği (%)", 50, 100, 90, help="Harf harf (karakter) benzerliği.")
    semantic_threshold = st.sidebar.slider("Semantic Eşiği (AI)", 10, 100, 45, help="Düşürürseniz daha uzak ama ilgili sonuçlar gelir.")

    u_sum = st.text_input("Sorgulanacak Özet (Summary)", placeholder="Örn: arama yapamıyorum")

    if master_file and u_sum:
        df = pd.read_csv(master_file, sep=None, engine="python").fillna("N/A")
        cols = get_col_names(df)
        
        if st.button("Analizi Başlat"):
            norm_query = clean_text(u_sum)
            with st.spinner('Yapay zeka havuzu tarıyor...'):
                summaries = df[cols["summary"]].astype(str).tolist()
                q_emb = model.encode(u_sum, convert_to_tensor=True)
                p_embs = model.encode(summaries, convert_to_tensor=True)
                scores = util.cos_sim(q_emb, p_embs)[0]

                exact_res, sem_res = [], []
                for i, row in df.iterrows():
                    raw_s = str(row[cols["summary"]])
                    clean_s = clean_text(raw_s)
                    
                    e_score = fuzz.partial_ratio(norm_query, clean_s)
                    s_score = float(scores[i]) * 100
                    
                    res = {
                        "ID": row.get(cols["key"], "N/A"),
                        "Status": row.get(cols["status"], "N/A"), 
                        "Component": row.get(cols["comp"], "N/A"), 
                        "Özet": raw_s,
                        "Skor": ""
                    }
                    
                    if (norm_query in clean_s) or e_score >= exact_threshold:
                        res["Skor"] = f"%{max(e_score, 100 if norm_query in clean_s else 0):.0f}"
                        exact_res.append(res)
                    elif s_score >= semantic_threshold:
                        res["Skor"] = f"%{s_score:.0f}"
                        sem_res.append(res)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🎯 Exact Matches")
                st.dataframe(pd.DataFrame(exact_res), use_container_width=True, hide_index=True)
            with c2:
                st.subheader("🧠 Semantic Matches (AI)")
                if sem_res:
                    st.dataframe(pd.DataFrame(sem_res).sort_values(by="Skor", ascending=False), use_container_width=True, hide_index=True)
                else:
                    st.info("Benzer sonuç bulunamadı.")

# --- MOD 2: TOPLU ANALİZ (BULK MATCH) ---
elif page == "Toplu Analiz (Bulk Match)":
    st.title("📊 Toplu Çakışma Analizi")
    st.caption("Yeni Listeyi Ana Havuz ile kıyaslayarak mükerrer kayıtları (ID bazlı) raporlar.")
    
    m_file = st.sidebar.file_uploader("1. ANA HAVUZ CSV (Eski Kayıtlar)", type=["csv"], key="bulk_m")
    n_file = st.sidebar.file_uploader("2. YENİ LİSTE CSV (Kontrol Edilecekler)", type=["csv"], key="bulk_n")
    bulk_threshold = st.sidebar.slider("Benzerlik Eşiği (%)", 10, 100, 55, help="Eşleşme sayılması için gereken minimum AI puanı.")

    if m_file and n_file:
        if st.button("Toplu Analizi Başlat ve Excel Raporu Üret 🚀"):
            # Verileri Oku
            df_m = pd.read_csv(m_file, sep=None, engine="python").fillna("N/A")
            df_n = pd.read_csv(n_file, sep=None, engine="python").fillna("N/A")
            
            mc, nc = get_col_names(df_m), get_col_names(df_n)
            
            with st.spinner('Büyük veri kümesi analiz ediliyor...'):
                m_sums = df_m[mc["summary"]].astype(str).tolist()
                n_sums = df_n[nc["summary"]].astype(str).tolist()
                
                # Vektörel Karşılaştırma (Semantic)
                m_embs = model.encode(m_sums, convert_to_tensor=True)
                n_embs = model.encode(n_sums, convert_to_tensor=True)
                cosine_scores = util.cos_sim(n_embs, m_embs)
                
                report = []
                for i in range(len(n_sums)):
                    # En yakın eşleşmeyi bul
                    best_score = float(cosine_scores[i].max()) * 100
                    best_idx = cosine_scores[i].argmax().item()
                    matched_row_m = df_m.iloc[best_idx]
                    
                    is_duplicate = best_score >= bulk_threshold
                    
                    # Rapor Satırı Yapısı
                    report.append({
                        "Yeni Liste ID": df_n.iloc[i][nc["key"]] if nc["key"] else f"Sıra-{i+1}",
                        "Yeni Liste Özet": n_sums[i],
                        "DURUM": "⚠️ MÜKERRER (DUPE)" if is_duplicate else "✅ YENİ (UNIQUE)",
                        "Benzerlik Skoru": f"%{best_score:.1f}",
                        "Ana Havuz ID": matched_row_m[mc["key"]] if mc["key"] else "Bulunamadı",
                        "Ana Havuz Özet": m_sums[best_idx],
                        "Havuz Component": matched_row_m[mc["comp"]] if mc["comp"] else "N/A",
                        "Havuz Durumu": matched_row_m[mc["status"]] if mc["status"] else "N/A"
                    })
                
                res_df = pd.DataFrame(report)
                
                # Ekranda Göster
                st.subheader("📋 Analiz Sonuç Özeti")
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                # Excel İndirme Hazırlığı
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Çakışma Raporu')
                
                st.download_button(
                    label="Analiz Raporunu İndir (Excel) 📥", 
                    data=output.getvalue(), 
                    file_name="jira_bulk_match_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success(f"Analiz tamamlandı. Toplam {len(n_sums)} kayıt kontrol edildi.")
