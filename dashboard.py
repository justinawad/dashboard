from urllib.parse import unquote
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import os

import numpy as np
from datetime import datetime
import time
import pickle
from collections import Counter
from rag_engine import process_pdf_and_create_vector_db, get_rag_chain
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import random
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from PIL import Image
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION & STYLING PRO
# ==========================================
icon = Image.open("logo.png")
st.set_page_config(page_title="Market_Visualizer", page_icon=icon, layout="wide", initial_sidebar_state="expanded")

# ---- CSS GLOBAL (cartes + boutons) ----
st.markdown("""
<style>
    /* --- 1. STYLE DES CARTES KPI (WIDGETS) --- */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
    }
    .highlight {
        color: #2563eb;
    }

    /* --- 2. STYLE UNIFORME DES BOUTONS (BLEU PAR DÉFAUT) --- */
    div.stButton > button,
    div.stDownloadButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: 1px solid #2563eb !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        transform: translateY(-2px) !important;
    }

    div.stButton > button:active,
    div.stButton > button:focus:not(:active),
    div.stDownloadButton > button:active {
        background-color: #1e40af !important;
        border-color: #1e40af !important;
        color: white !important;
        box-shadow: none !important;
    }

    /* --- 3. STYLE SPÉCIFIQUE POUR LE BOUTON D'AVIS (KEY=btn_avis) --- */
    .st-key-btn_avis button {
        background-color: #f97316 !important;   /* orange */
        border-color: #f97316 !important;
        color: #ffffff !important;
    }
    .st-key-btn_avis button:hover {
        background-color: #ea580c !important;
        border-color: #ea580c !important;
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)  # [web:18][web:21]

load_dotenv()

# Récupérer la clé
KEYS = os.getenv("GROQ_API_KEY").split(",")
MODEL_NAME = "llama-3.1-8b-instant"

# ---- CSS LAYOUT / TABS ----
st.markdown("""
<style>
    /* Réduire le vide en haut de page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Style des titres H1, H2, H3 */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    /* Style des Cartes KPI (Metric Cards) */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Supprimer l'espace vide sous les graphiques Plotly */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }

    /* Style des onglets (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-top: 2px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS & MOCK ML MODEL
# ==========================================

@st.cache_resource
def load_prediction_model():
    """Loads the real XGBoost model."""
    model_path = "salary_model_xgboost.pkl"

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        st.error("Model file not found. Please put 'salary_model_xgboost.pkl' in the folder.")
        return None

@st.cache_data(show_spinner=False)
def extract_job_keywords(job_text):
    if not job_text or len(job_text) < 10:
        return []

    parser = JsonOutputParser()
    selected_keys = random.choice(KEYS)
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=selected_keys,
        model_name="llama-3.1-8b-instant"
    )

    prompt_text = f"""
    You are a Technical Recruiter. Extract technical keywords from this job description only the skills ex (SQL , python , docker , git).
    Return ONLY a JSON list of objects with keys: "keyword", "type", "importance_score".
    n.b : the importance_score is between 0 to 100 % 
    JOB TEXT:
    {job_text}

    Format instructions: {parser.get_format_instructions()}
    """

    try:
        chain = llm | parser
        response = chain.invoke(prompt_text)
        return response if isinstance(response, list) else []
    except Exception as e:
        st.error("**Oups ! Le model est momentanément très sollicité.**")
        st.warning("""
        En raison d'une forte affluence sur la plateforme (200+ utilisateurs simultanés), 
        le serveur a atteint sa limite de vitesse temporaire.

        **Action :** Pas d'inquiétude ! Tes données sont conservées. 
        Patiente environ **30 à 60 secondes** et retry.
        """)

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    path = "data/cleaned_dataset_domaine_info.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df['salaire_avg'] = pd.to_numeric(df['salaire_avg'], errors='coerce')
        df = df.dropna(subset=['salaire_avg'])

        if 'date_publication' not in df.columns:
            df['date_publication'] = pd.date_range(end=datetime.now(), periods=len(df)).tolist()
        else:
            df['date_publication'] = pd.to_datetime(df['date_publication'])
        df = df[df['metier'] != "#Nextstep - Ingénieur Spécialiste En Développement Logiciel Bancs De Tests"]    
            
        return df
    else:
        st.error(f"File not found: {path}")
        return pd.DataFrame()

def get_top_skills_for_job(df, job_name):
    subset = df[df['metier'] == job_name]
    all_text = ",".join(subset['competences'].dropna().astype(str).tolist())
    skills_list = [s.strip().title() for s in all_text.split(',') if len(s.strip()) > 1]
    counter = Counter(skills_list)
    return [s[0] for s in counter.most_common(5)]

@st.cache_data
def load_geojson(scale="regions"):
    urls = {
        "regions": "https://france-geojson.gregoiredavid.fr/repo/regions.geojson",
        "departements": "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
    }

    try:
        import requests
        r = requests.get(urls[scale])
        return r.json()
    except Exception as e:
        st.error(f"Erreur de chargement de la carte ({scale}): {e}")
        return None

df = load_data()
geojson = load_geojson()

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def rank_cvs(uploaded_files, job_description):
    embed_model = load_embedding_model()

    job_vector = embed_model.embed_query(job_description)
    job_vector = np.array([job_vector])

    results = []

    progress_bar = st.progress(0)
    for i, file in enumerate(uploaded_files):
        try:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            cv_vector = embed_model.embed_query(text)
            cv_vector = np.array([cv_vector])

            score = cosine_similarity(job_vector, cv_vector)[0][0]

            results.append({
                "Nom du Fichier": file.name,
                "Score de Pertinence": round(score * 100, 2),
                "Texte Brut": text[:500] + "..."
            })
        except Exception as e:
            print(f"Erreur fichier {file.name}: {e}")

        progress_bar.progress((i + 1) / len(uploaded_files))

    progress_bar.empty()

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by="Score de Pertinence", ascending=False)

    return df_results

# ==========================================
# 4. SIDEBAR & INPUTS
# ==========================================

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3098/3098090.png", width=50)
st.sidebar.title("Configuration")

# 1. Metier (Category) - Gets list from your CSV
target_metier = st.sidebar.selectbox(
    "Métier",
    options=sorted(df['metier'].unique()),
    index=None,
    placeholder="Choisissez un métier..."
)

# 2. Region - Gets list from your CSV
target_region = st.sidebar.selectbox("Région", options=sorted(df['region'].unique()))

exp_options = ["Junior (0-2 ans)", "Intermédiaire (2-5 ans)", "Senior (5+ ans)", "Non spécifié"]
target_experience = st.sidebar.selectbox("Expérience", options=exp_options)

# 4. Job Title
target_title = st.sidebar.text_input("Intitulé du Poste", "Choisissez un post...")

# 5. Description
target_desc = st.sidebar.text_area("Description de l'offre", height=200, placeholder="Collez la description ici...")

known_skills = ["Python", "SQL", "Java", "AWS", "Azure", "Docker", "Kubernetes", "React", "Terraform"]
target_skills = st.sidebar.multiselect("Compétences Clés (Filtre Dashboard)", known_skills)

predict_btn = st.sidebar.button(" Lancer la Prédiction", use_container_width=True)

# Session State
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'extracted_skills' not in st.session_state:
    st.session_state['extracted_skills'] = []

# ==========================================
# 5. MAIN LOGIC FLOW
# ==========================================

st.title("Market_Visualizer • Talent Intelligence & Compensation Analytics")
st.markdown(f"Analyse de marché pour **{target_metier}** en **{target_region}**.")
st.divider()

# ==========================================
# 6. LOGIQUE DE PRÉDICTION & LOADING
# ==========================================

if predict_btn:
    if not target_desc:
        st.error("⚠️ Veuillez coller une description d'offre.")
    else:
        with st.status(" Initialisation ", expanded=True) as status:
            st.write("Connexion ...")
            time.sleep(0.8)

            st.write(" Analyse sémantique description...")
            ai_skills = extract_job_keywords(target_desc)
            st.session_state['extracted_skills'] = ai_skills
            time.sleep(0.5)

            st.write("Agrégation des données régionales...")
            skills_str = ", ".join([s['keyword'] for s in ai_skills]) if ai_skills else ""

            model = load_prediction_model()

            if model:
                text_input = (
                    str(target_title) + " " +
                    str(target_metier) + " " +
                    str(skills_str) + " " + str(target_experience) + " " + str(target_region) + " " +
                    str(target_desc)
                )

                input_df = pd.DataFrame({
                    'text_features': [text_input],
                    'metier': [target_metier],
                    'experience': [target_experience],
                    'region': [target_region]
                })

                try:
                    pred_val = model.predict(input_df)[0]
                    pred_min = pred_val * 0.9
                    pred_max = pred_val * 1.1
                    st.session_state['prediction'] = {"val": pred_val, "range": (pred_min, pred_max)}
                    status.update(label="✅ Analyse terminée avec succès !", state="complete", expanded=False)

                    dates = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    token_encoded = st.query_params.get("us", "")
                    token_user = unquote(token_encoded)
                    requete_historique = requests.post(
                        "https://predict-production-28b1.up.railway.app/api/predict/historique",
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {token_user}'
                        },
                        json={
                            "salaire_predit": int(pred_val),
                            "salaire_min": int(pred_min),
                            "salaire_mensuel": int(pred_max),
                            "niveau_experience": f"{input_df['experience'].iloc[0]}",
                            "date_predit": dates,
                            "description": f"{input_df['desc'].iloc[0]}",
                            "competences": f"{input_df['competences'].iloc[0]}",
                            "region": f"{input_df['region'].iloc[0]}",
                            "titre": f"{input_df['metier'].iloc[0]}"
                        }
                    )

                except Exception as e:
                    status.update(label="❌ Erreur critique", state="error")
                    st.error(f"Erreur : {e}")
            else:
                st.error("Modèle introuvable.")

# ==========================================
# 7. AFFICHAGE DU DASHBOARD (BLOC UNIQUE)
# ==========================================

if st.session_state['prediction']:
    pred = st.session_state['prediction']

    st.write("")

    # --- A. HEADER & BOUTON DOWNLOAD (HAUT GAUCHE) ---
    col_dl, col_title = st.columns([1, 4])

    with col_dl:
        import datetime as dt
        date_jour = dt.datetime.now().strftime("%d/%m/%Y")

        skills_html = ""
        if st.session_state['extracted_skills']:
            for s in st.session_state['extracted_skills']:
                skills_html += f"<span class='badge'>{s['keyword']} ({s['importance_score']}%)</span>"
        else:
            skills_html = "<i>Aucune compétence technique spécifique détectée.</i>"

        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Rapport - {target_title}</title>
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f4f9; padding: 40px; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}

                .header {{ border-bottom: 2px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }}
                .logo {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
                .date {{ color: #888; font-size: 14px; }}

                .salary-box {{ background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 40px; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2); }}
                .salary-title {{ text-transform: uppercase; font-size: 14px; letter-spacing: 1px; opacity: 0.9; margin-bottom: 5px; }}
                .salary-amount {{ font-size: 48px; font-weight: 700; margin: 0; }}
                .salary-range {{ font-size: 18px; margin-top: 10px; opacity: 0.9; background: rgba(255,255,255,0.2); display: inline-block; padding: 5px 15px; border-radius: 20px; }}

                .section-title {{ font-size: 18px; font-weight: 700; color: #1e293b; margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #2563eb; padding-left: 10px; }}

                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .info-item {{ background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; }}
                .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 5px; }}
                .value {{ font-size: 16px; font-weight: 600; color: #0f172a; }}

                .badge {{ display: inline-block; background-color: #e0f2fe; color: #0369a1; padding: 6px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; margin-right: 5px; margin-bottom: 5px; }}

                .desc-box {{ background: #fff; border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; font-size: 14px; line-height: 1.6; color: #475569; white-space: pre-wrap; }}

                .footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">Market_Visualiser</div>
                    <div class="date">Rapport généré le {date_jour}</div>
                </div>

                <div class="salary-box">
                    <div class="salary-title">Estimation du Salaire Annuel Brute </div>
                    <div class="salary-amount">{pred['val']:,.0f} €</div>
                    <div class="salary-range">Fourchette : {pred['range'][0]:,.0f} € - {pred['range'][1]:,.0f} €</div>
                </div>

                <div class="section-title">Détails du Poste</div>
                <div class="grid">
                    <div class="info-item"><span class="label">Intitulé</span><span class="value">{target_title}</span></div>
                    <div class="info-item"><span class="label">Catégorie Métier</span><span class="value">{target_metier}</span></div>
                    <div class="info-item"><span class="label">Région</span><span class="value">{target_region}</span></div>
                    <div class="info-item"><span class="label">Expérience</span><span class="value">{target_experience}</span></div>
                </div>

                <div class="section-title">Analyse Sémantique (Compétences Clés)</div>
                <div>
                    {skills_html}
                </div>

                <div class="section-title">Description Originale</div>
                <div class="desc-box">{target_desc}</div>

                <div class="footer">
                    Ce rapport a été généré automatiquement .<br>
                    &copy; 2026 Market_Visualizer - Tous droits réservés.
                </div>
            </div>
        </body>
        </html>
        """

        st.download_button(
            label=" Télécharger Rapport Complet",
            data=html_report,
            file_name=f"Rapport_Complet_{target_metier}.html",
            mime="text/html",
            use_container_width=True,
            key="btn_download_full_report"
        )

    with col_title:
        st.markdown(f"###  Analyse : **{target_metier}** en **{target_region}**")

    st.markdown("---")

    # --- B. CARTE ---
    col_map_filters, col_map_viz = st.columns([1, 3])

    with col_map_filters:
        st.markdown("####  Carte")
        view_mode = st.radio("Zoom :", ["Région", "Département"], index=0, key="radio_map_zoom")
        st.write("")
        metric_label = st.radio(" Indicateur :", ["Médiane", "Moyenne"], index=0, key="radio_map_metric")
        metric_func = 'median' if "Médiane" in metric_label else 'mean'

    with col_map_viz:
        if view_mode == "Région":
            map_df = df[df['metier'] == target_metier].groupby('region')['salaire_avg'].agg(metric_func).reset_index()
            geojson = load_geojson("regions")
            loc_col, key_json, zoom_lvl = 'region', "properties.nom", 4.5
        else:
            map_df = df[df['metier'] == target_metier].groupby('departement')['salaire_avg'].agg(metric_func).reset_index()
            map_df['departement'] = map_df['departement'].astype(str).str.zfill(2)
            geojson = load_geojson("departements")
            loc_col, key_json, zoom_lvl = 'departement', "properties.code", 5.0

        if geojson:
            fig_map = px.choropleth_mapbox(
                map_df, geojson=geojson, locations=loc_col, featureidkey=key_json,
                color='salaire_avg', color_continuous_scale="Blues",
                mapbox_style="carto-positron", zoom=zoom_lvl, center={"lat": 46.6, "lon": 2.2},
                opacity=0.7, labels={'salaire_avg': f'{metric_label} (€)'}
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=450)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.error("Carte indisponible.")

    # --- C. KPIs ---
    st.markdown("### Indicateurs Clés")

    c1, c2, c3, c4 = st.columns(4)

    mask_context = (df['metier'] == target_metier) & (df['region'] == target_region)
    market_data = df[mask_context]
    market_val = market_data['salaire_avg'].median() if not market_data.empty else 0
    sample_size = len(market_data)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Salaire Estimé (Brut)</div>
            <div class="metric-value highlight">{pred['val']:,.0f} €</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Marché ({target_region})</div>
            <div class="metric-value">{market_val:,.0f} €</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fourchette Probable</div>
            <div class="metric-value" style="font-size: 1.2rem;">{pred['range'][0]:,.0f} - {pred['range'][1]:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("---")

    # --- D. ANALYSE DÉTAILLÉE ---
    col_bench, col_ai = st.columns([1, 1], gap="medium")

    with col_bench:
        with st.container(border=True):
            st.markdown("#### Comparateur de Métiers")
            all_metiers = sorted(df['metier'].unique())
            try:
                default_ix = all_metiers.index("Data Engineer")
            except:
                default_ix = 0
            comp_metier = st.selectbox("Comparer avec :", all_metiers, index=default_ix, key="sb_compare_metier")

            df_curr = df[df['metier'] == target_metier]
            df_comp = df[df['metier'] == comp_metier]
            med_curr = df_curr['salaire_avg'].median()
            med_comp = df_comp['salaire_avg'].median()

            fig_comp = go.Figure(data=[
                go.Bar(name=target_metier, x=[target_metier], y=[med_curr], marker_color='#2563eb', text=f"{med_curr:,.0f}€", textposition='auto'),
                go.Bar(name=comp_metier, x=[comp_metier], y=[med_comp], marker_color='#94a3b8', text=f"{med_comp:,.0f}€", textposition='auto')
            ])
            fig_comp.update_layout(title="Salaire Médian Comparé", height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

            st.markdown("**Top Compétences :**")
            skills_curr = get_top_skills_for_job(df, target_metier)
            skills_comp = get_top_skills_for_job(df, comp_metier)
            c1, c2 = st.columns(2)
            with c1:
                for s in skills_curr[:4]:
                    st.caption(f"🔹 {s}")
            with c2:
                for s in skills_comp[:4]:
                    st.caption(f"🔸 {s}")

    with col_ai:
        with st.container(border=True):
            st.markdown("#### Analyse des competences de votre Offre ")
            if st.session_state['extracted_skills']:
                skills_df = pd.DataFrame(st.session_state['extracted_skills']).sort_values('importance_score', ascending=True)
                fig_skills = px.bar(
                    skills_df, x="importance_score", y="keyword", orientation='h',
                    color="importance_score", color_continuous_scale="Teal",
                    title="Mots-clés détectés"
                )
                fig_skills.update_layout(
                    title=None,
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(showticklabels=False, title=None),
                    yaxis=dict(title=None)
                )
                st.plotly_chart(fig_skills, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Collez une description pour voir l'analyse.")

    st.markdown("---")

# ==========================================
# 8. CENTRE DE CARRIÈRE & RECRUTEMENT
# ==========================================
st.header("Centre de Recrutement & Carrière")

tab_candidat, tab_recruteur = st.tabs([" Espace Candidat (optimiser vos candidature)", " Espace Recruteur (Tri CVs)"])

with tab_candidat:
    st.info("** Posez vos questions sur votre compatibilité avec le poste.")
    col_cv, col_chat = st.columns([1, 2])

    with col_cv:
        uploaded_cv = st.file_uploader("Analysez votre CV (PDF)", type=["pdf"], key="cv_upload_candidat")
        if uploaded_cv:
            st.success("✅ CV chargé")

    with col_chat:
        if "messages_candidat" not in st.session_state:
            st.session_state.messages_candidat = []
        if "vector_store_candidat" not in st.session_state:
            st.session_state.vector_store_candidat = None

        if uploaded_cv and st.session_state.vector_store_candidat is None:
            with st.spinner(" Analyse du CV..."):
                st.session_state.vector_store_candidat = process_pdf_and_create_vector_db(uploaded_cv)
                st.rerun()

        chat_container = st.container(height=350)
        with chat_container:
            for msg in st.session_state.messages_candidat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if uploaded_cv:
            b1, b2, b3 = st.columns(3)
            action_prompt = None
            if b1.button(" Analyser CV", use_container_width=True):
                action_prompt = "Analyse mon CV par rapport à ce poste. Points forts/faibles ?"
            if b2.button(" Comparer", use_container_width=True):
                action_prompt = f"Donne mon score de compatibilité pour le poste de {target_metier}."
            if b3.button("Lettre Motiv'", use_container_width=True):
                action_prompt = "Rédige une lettre de motivation courte."

            if action_prompt:
                st.session_state.messages_candidat.append({"role": "user", "content": action_prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(action_prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("réfléchit..."):
                            selected_key = random.choice(KEYS)
                            qa_chain = get_rag_chain(st.session_state.vector_store_candidat, selected_key)

                            full_query = f"ACTION: {action_prompt} | OFFRE: {target_desc[:500]}"
                            response = qa_chain.invoke(full_query)

                            st.session_state.messages_candidat.append({"role": "assistant", "content": response})
                            st.markdown(response)

with tab_recruteur:
    st.markdown("#### 📂 Analyse d'un dossier de resume")
    st.write(f"Identifiez instantanément les meilleurs profils pour le poste de **{target_metier}**.")
    uploaded_files = st.file_uploader("Glissez-déposez les CVs (PDF)", type=['pdf'], accept_multiple_files=True, key="bulk_upload")

    if uploaded_files:
        if st.button(f" Analyser {len(uploaded_files)} CVs", key="btn_launch_bulk"):
            with st.spinner("Classement en cours..."):
                ranked_df = rank_cvs(uploaded_files, target_desc)
                st.success("Terminé !")
                if len(ranked_df) >= 3:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🥇 Top 1", ranked_df.iloc[0]['Nom du Fichier'], f"{ranked_df.iloc[0]['Score de Pertinence']}%")
                    c2.metric("🥈 Top 2", ranked_df.iloc[1]['Nom du Fichier'], f"{ranked_df.iloc[1]['Score de Pertinence']}%")
                    c3.metric("🥉 Top 3", ranked_df.iloc[2]['Nom du Fichier'], f"{ranked_df.iloc[2]['Score de Pertinence']}%")

                st.dataframe(
                    ranked_df[['Score de Pertinence', 'Nom du Fichier']],
                    use_container_width=True,
                    column_config={
                        "Score de Pertinence": st.column_config.ProgressColumn(
                            "Pertinence", format="%.1f%%", min_value=0, max_value=100
                        )
                    }
                )

# ==========================================
# 9. SECTION AVIS UTILISATEUR (OUVERTE PAR LE BOUTON SIDEBAR)
# ==========================================
# Bouton pour ouvrir le pop-up
# --- Modal d'avis ---
@st.dialog("Donnez votre avis", width="medium")
def avis_dialog():
    st.write("Nous serions ravis d'avoir votre avis sur l'application")
    times = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    satisfaction = st.slider("Votre satisfaction globale", 1, 5, 4, 1)
    commentaire = st.text_area(
        "Commentaire",
        placeholder="Dites-nous ce que vous aimez ou ce qu'on peut améliorer..."
    )

    col1, col2 = st.columns(2)
    with col1:
        envoyer = st.button("Envoyer mon avis", type="primary", use_container_width=True)
    with col2:
        annuler = st.button("Annuler", use_container_width=True)

    if envoyer:
        # Exemple : appel API pour enregistrer l'avis (à adapter)
        try:
            payload = {
                "dates":f"{times}",
                "satisfaction": f"{satisfaction}",
                "commentaire": f"{commentaire}",
            }
            r = requests.post(
                "https://predict-production-28b1.up.railway.app/api/predict/feedback",
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {token_user}'
                },
                json=payload,
                timeout=10,
            )
            if r.status_code == 200:
                st.success("Merci pour votre avis !")
            else:
                st.warning("Avis envoyé, mais le serveur a retourné une réponse inattendue.")
        except Exception as e:
            st.error(f"Erreur lors de l'envoi de l'avis : {e}")

        # Ferme le modal en forçant un rerun
        st.rerun()

    if annuler:
        # Ferme simplement le modal
        st.rerun()

if "last_feedback" in st.session_state:
    fb = st.session_state["last_feedback"]
    st.write(f"Dernier avis : {fb}")

# Bouton qui ouvre le modal
if st.sidebar.button("Donner mon avis",use_container_width=True,key="btn_avis"):
    avis_dialog()

