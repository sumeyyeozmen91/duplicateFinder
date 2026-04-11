import streamlit as st
import pandas as pd
import re
import io
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- Konfigürasyon ---
st.set_page_config(page_title="Jira Bug AI Hub", page_icon="🕵️", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9çğıöşü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_col_names(df):
    c_map = {c.lower().strip(): c for c in df.columns}
    return {
        "summary": c_map.get('summary', c_map.get('özet', None)),
        "key": c_map.get('issue key', c_map.get('key', c_map.get('anahtar', None))),
        "comp": c_map.get('component', c_map.get('components', c_map.get('bileşen', None))),
        "status": c_map.get('status', c_map.get('durum', None))
    }

# --- Yan Menü (Navigasyon) ---
st.sidebar.title("🚀 Jira AI Hub")
page = st.sidebar.radio("Mod Seçiniz:", ["Tekli Arama (Manual)", "Toplu Analiz (Bulk)"])

# --- MOD 1: TEKLİ ARAMA ---
if page == "Tekli Arama (Manual)":
    st.title("🔍 Tekli Bulgu Kontrolü")
    st.caption("Yeni bir bug'ı havuzda manuel olarak sorgulayın.")
    
    master_file = st.sidebar.file_uploader("Ana Havuz CSV", type=["csv"], key="single_master")
    exact_threshold = st.sidebar.slider("Exact Eşiği (%)", 50, 100, 90)
    semantic_threshold = st.sidebar.slider("Semantic Eşiği (AI)", 20, 100, 60)

    u_sum = st.text_input("Bulgu Özeti (Summary)", placeholder="Örn: arama yapamıyorum")

    if master_file and u_sum:
        df = pd.read_csv(master_file, sep=None, engine="python").fillna("N/A")
        cols = get_col_names(df)
        
        if st.button("Analiz Et"):
            norm_query = clean_text(u_sum)
            search_keywords = [clean_text(w) for w in u_sum.split() if len(clean_text(w)) > 2]
            
            summaries = df[cols["summary"]].astype(str).tolist()
            q_emb = model.encode(u_sum, convert_to_tensor=True)
            p_embs = model.encode(summaries, convert_to_tensor=True)
            scores = util.cos_sim(q_emb, p_embs)[0]

            exact_res, sem_res = [], []
            for i, row in df.iterrows():
                clean_s = clean_text(str(row[cols["summary"]]))
                e_score = fuzz.partial_ratio(norm_query, clean_s)
                s_score = (float(scores[i]) * 100) + (10 if any(k in clean_s for k in search_keywords) else -15)
                
                res = {"ID": row.get(cols["key"], "N/A"), "Status": row.get(cols["status"], "N/A"), 
                       "Component": row.get(cols["comp"], "N/A"), "Özet": row[cols["summary"]]}
                
                if (norm_query in clean_s) or e_score >= exact_threshold:
                    res["Skor"] = f"%{max(e_score, 100 if norm_query in clean_s else 0):.0f}"
                    exact_res.append(res)
                elif s_score >= semantic_threshold:
                    res["Skor"] = f"%{max(0, min(100, s_score)):.0f}"
                    sem_res.append(res)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🎯 Exact Matches")
                st.dataframe(pd.DataFrame(exact_res), hide_index=True)
            with c2:
                st.subheader("🧠 Semantic Matches")
                st.dataframe(pd.DataFrame(sem_res).sort_values(by="Skor", ascending=False) if sem_res else pd.DataFrame(), hide_index=True)

# --- MOD 2: TOPLU ANALİZ ---
elif page == "Toplu Analiz (Bulk)":
    st.title("📊 Toplu Çakışma Analizi")
    st.caption("Bir listeyi ana havuzla karşılaştırıp Excel raporu alın.")
    
    m_file = st.sidebar.file_uploader("Ana Havuz CSV", type=["csv"], key="bulk_m")
    n_file = st.sidebar.file_uploader("Yeni Liste CSV", type=["csv"], key="bulk_n")
    bulk_threshold = st.sidebar.slider("Benzerlik Eşiği (%)", 30, 100, 75)

    if m_file and n_file:
        if st.button("Toplu Analizi Başlat 🚀"):
            df_m = pd.read_csv(m_file, sep=None, engine="python").fillna("N/A")
            df_n = pd.read_csv(n_file, sep=None, engine="python").fillna("N/A")
            
            mc, nc = get_col_names(df_m), get_col_names(df_n)
            
            m_sums = df_m[mc["summary"]].astype(str).tolist()
            n_sums = df_n[nc["summary"]].astype(str).tolist()
            
            m_embs = model.encode(m_sums, convert_to_tensor=True)
            n_embs = model.encode(n_sums, convert_to_tensor=True)
            cosine_scores = util.cos_sim(n_embs, m_embs)
            
            report = []
            for i in range(len(n_sums)):
                best_score = float(cosine_scores[i].max()) * 100
                best_idx = cosine_scores[i].argmax().item()
                match = best_score >= bulk_threshold
                
                report.append({
                    "Yeni Bug": n_sums[i], "Durum": "⚠️ DUPLICATE" if match else "✅ UNIQUE",
                    "Skor": f"%{best_score:.1f}", "Eşleşen Özet": m_sums[best_idx],
                    "Eşleşen ID": df_m.iloc[best_idx][mc["key"]] if mc["key"] else "-"
                })
            
            res_df = pd.DataFrame(report)
            st.dataframe(res_df, use_container_width=True)
            
            # Excel İndirme
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("Raporu İndir (Excel)", data=output.getvalue(), file_name="analiz.xlsx")
