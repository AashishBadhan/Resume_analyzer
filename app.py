import streamlit as st
import fitz
import re
import pandas as pd
import joblib
import os
import zipfile
import gdown
import shutil
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Smart ATS | Premium", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

def toggle_theme():
    st.session_state.theme = 'Light' if st.session_state.theme == 'Dark' else 'Dark'

base_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family: 'Plus Jakarta Sans', sans-serif;}
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {background-color: transparent !important;}
.main-header {font-size: 46px; font-weight: 800; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 5px; animation: fadeIn 1s ease-in;}
.sub-header {text-align: center; font-size: 18px; margin-bottom: 40px; font-weight: 600;}
@keyframes fadeIn {from {opacity: 0; transform: translateY(-20px);} to {opacity: 1; transform: translateY(0);}}
.stButton>button {width: 100%; border-radius: 10px; font-weight: 800; transition: all 0.4s ease; border: none; padding: 0.7rem; font-size: 16px;}
.stButton>button:hover {transform: scale(1.02); box-shadow: 0 8px 20px rgba(79, 172, 254, 0.4);}
[data-testid="stMetricValue"] {font-size: 32px !important; font-weight: 800 !important;}
[data-testid="stMetricLabel"] {font-size: 15px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 1px;}
div[data-testid="metric-container"] {border-radius: 16px; padding: 25px; text-align: center; transition: all 0.3s ease;}
div[data-testid="metric-container"]:hover {transform: translateY(-5px);}
.streamlit-expanderHeader {font-size: 18px; font-weight: 800; border-radius: 10px; transition: all 0.3s ease; padding: 15px !important;}
"""

if st.session_state.theme == 'Dark':
    theme_css = """
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span, .stApp li, .stApp label {color: #ffffff !important;}
    .sub-header {color: #a0aec0 !important;}
    [data-testid="stSidebar"] {background-color: #0b0f19; border-right: 1px solid #1f2937;}
    div[data-testid="metric-container"] {background: linear-gradient(145deg, #111827, #1f2937); border: 1px solid #374151; box-shadow: 0 10px 25px rgba(0,0,0,0.5);}
    [data-testid="stMetricValue"] {color: #00f2fe !important;}
    [data-testid="stMetricLabel"] {color: #9ca3af !important;}
    .streamlit-expanderHeader {color: #00f2fe !important; background-color: #111827 !important; border: 1px solid #374151;}
    .stButton>button {background: linear-gradient(90deg, #1e3a8a 0%, #111827 100%); color: white !important;}
    .stTextArea textarea {background-color: #111827 !important; color: #ffffff !important; border: 1px solid #374151 !important; border-radius: 10px;}
    div[role="tablist"] button[aria-selected="false"] p {color: #9ca3af !important;}
    div[role="tablist"] button[aria-selected="true"] p {color: #00f2fe !important; font-weight: 800 !important;}
    """
else:
    theme_css = """
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span, .stApp li, .stApp label {color: #0f172a !important;}
    .sub-header {color: #64748b !important;}
    [data-testid="stSidebar"] {background-color: #f8fafc; border-right: 1px solid #e2e8f0;}
    div[data-testid="metric-container"] {background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.05);}
    [data-testid="stMetricValue"] {color: #2563eb !important;}
    [data-testid="stMetricLabel"] {color: #64748b !important;}
    .streamlit-expanderHeader {color: #1e40af !important; background-color: #f1f5f9 !important; border: 1px solid #e2e8f0;}
    .stButton>button {background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%); color: white !important;}
    .stTextArea textarea {background-color: #ffffff !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; border-radius: 10px;}
    div[role="tablist"] button[aria-selected="false"] p {color: #64748b !important;}
    div[role="tablist"] button[aria-selected="true"] p {color: #2563eb !important; font-weight: 800 !important;}
    """

st.markdown(f"<style>{base_css}{theme_css}</style>", unsafe_allow_html=True)

@st.cache_resource
def download_model_if_missing():
    model_dir = os.path.abspath('./distilbert_resume_model')
    config_file = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_file):
        if os.path.exists(model_dir): shutil.rmtree(model_dir)
        file_id = '1cjxek02nIA36_8lmC-B66HwYjPR6wsyS' 
        output = 'model.zip'
        temp_extract_dir = os.path.abspath('./temp_model_extract')
        try:
            with st.spinner("⏳ First time setup: Downloading Heavy AI Model..."):
                gdown.download(id=file_id, output=output, quiet=False)
                os.makedirs(temp_extract_dir, exist_ok=True)
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                os.remove(output)
                found_model_path = None
                for root, dirs, files in os.walk(temp_extract_dir):
                    if 'config.json' in files:
                        found_model_path = root
                        break
                os.makedirs(model_dir, exist_ok=True)
                for item in os.listdir(found_model_path):
                    shutil.move(os.path.join(found_model_path, item), os.path.join(model_dir, item))
                shutil.rmtree(temp_extract_dir)
        except Exception as e:
            st.error(f"🚨 Download failed: {str(e)}")
            st.stop()

download_model_if_missing()

@st.cache_resource
def load_ai_model():
    base_model_path = os.path.abspath('./distilbert_resume_model')
    le = joblib.load('label_encoder.pkl')
    bert_analyzer = pipeline("text-classification", model=base_model_path, tokenizer=base_model_path)
    return le, bert_analyzer

le, bert_analyzer = load_ai_model()

def clean_text(text):
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'[^\w\s\.\@\-\+]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_phone(text):
    patterns = [
        r'(?:\+91|91|0)?[\s\-]?\d{5}[\s\-]?\d{5}',
        r'\+?\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}',
        r'\b\d{10}\b',
        r'\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b'
    ]
    for p in patterns:
        matches = re.findall(p, text)
        if matches:
            for m in matches:
                clean_m = re.sub(r'[\s\-\(\)]', '', m)
                if 10 <= len(clean_m) <= 13: return m.strip()
    return "Not Found"

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not Found"

def extract_name(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:15]:
        match = re.search(r'(?i)\b(?:name|candidate|applicant|first name)\s*[:\-]\s*([A-Za-z\s]+)', line)
        if match: return match.group(1).title().strip()
    ignore = ['resume', 'cv', 'curriculum', 'vitae', 'profile', 'email', 'phone', 'address', 'mobile', 'dob', 'date', 'page', 'linkedin', 'github', 'summary', 'objective']
    for line in lines[:10]:
        if len(line.split()) > 4 or len(line) > 30: continue
        if any(w in line.lower() for w in ignore): continue
        if re.search(r'\d', line): continue
        if '@' in line or '.com' in line: continue
        words = line.split()
        if 1 < len(words) <= 3: return line.title()
    for line in lines[:5]:
        words = line.split()
        if 1 < len(words) <= 3 and not re.search(r'\d', line) and not '@' in line: return line.title()
    return "Unknown Candidate"

def extract_experience(text):
    exp_pattern = r'(?i)\b([0-9]{1,2}(?:\.[0-9]{1,2})?)\s*(?:\+)?\s*(?:years?|yrs?|months?)\s*(?:of)?\s*(?:experience|exp)?\b'
    matches = re.findall(exp_pattern, text)
    if matches:
        max_exp = max([float(m) for m in matches])
        return f"{max_exp} Years" if max_exp > 1 else "Fresher (< 1 Year)"
    return "Fresher / Not Specified"

def extract_skills(text):
    tech_skills = ['python', 'java', 'c++', 'react', 'node.js', 'sql', 'aws', 'docker', 'kubernetes', 'machine learning', 'nlp', 'html', 'css', 'javascript', 'django', 'flask', 'fastapi', 'git', 'excel', 'powerbi']
    found = [s for s in tech_skills if re.search(r'\b' + re.escape(s) + r'\b', text.lower())]
    return ", ".join(found).title() if found else "Not Explicitly Listed"

def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    raw_text = ""
    for page in doc: raw_text += page.get_text("text")
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    name = extract_name(raw_text)
    exp = extract_experience(raw_text)
    skills = extract_skills(raw_text)
    return raw_text, email, phone, name, exp, skills

def get_match_score(jd, resume):
    if not jd or not resume: return 0
    vectors = TfidfVectorizer(stop_words='english').fit_transform([jd, resume])
    score = cosine_similarity(vectors)[0][1] * 100
    return round(score, 2)

st.markdown("<div class='main-header'>AI-Powered Resume Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Next-Gen Deep Learning Categorization & Suitability Ranking Engine</div>", unsafe_allow_html=True)

with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135692.png", width=45)
    with col2:
        st.markdown("<h3 style='margin-top: 5px; color: #00f2fe;'>Control Panel</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.button("🌓 Toggle Light/Dark Mode", on_click=toggle_theme, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 1. Target Job Description")
    jd_input = st.text_area("Paste JD here", height=200, placeholder="Paste required skills, tech stack, and role details here...", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📂 2. Upload Talent Pool")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### 🛠️ AI System Diagnostics")
    st.markdown("🟢 **NLP Engine:** DistilBERT Active")
    st.markdown("🟢 **Heuristics:** 5-Tier Extraction On")
    st.markdown("🟢 **Deduplication:** Armed & Ready")

if uploaded_files:
    candidates = []
    progress_text = "🧠 Neural Network is actively parsing candidate vectors..."
    my_bar = st.progress(0, text=progress_text)
    for i, file in enumerate(uploaded_files):
        raw_text, email, phone, name, experience, skills = process_pdf(file)
        cleaned_text = clean_text(raw_text)
        if len(cleaned_text) < 50:
            category, score = "Invalid/Scanned PDF", 0
        else:
            prediction = bert_analyzer(cleaned_text, truncation=True, max_length=512)
            label_id = int(prediction[0]['label'].split('_')[-1])
            category = le.inverse_transform([label_id])[0]
            score = get_match_score(clean_text(jd_input), cleaned_text) if jd_input else 0
        candidates.append({
            "File Name": file.name, "Name": name, "Domain": category,
            "Exp": experience, "Skills": skills, "Score": score, 
            "Email": email, "Phone": phone, "Raw": raw_text
        })
        my_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
    my_bar.empty()
    df = pd.DataFrame(candidates).sort_values(by="Score", ascending=False)
    
    valid_emails = df[df['Email'] != 'Not Found']
    valid_phones = df[df['Phone'] != 'Not Found']
    dupes = pd.concat([
        valid_emails[valid_emails.duplicated(subset=['Email'], keep=False)],
        valid_phones[valid_phones.duplicated(subset=['Phone'], keep=False)]
    ]).drop_duplicates(subset=['File Name'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Resumes Processed", len(df))
    c2.metric("🎯 Highest Suitability", f"{df['Score'].max()}%" if not df.empty else "0%")
    c3.metric("💼 Senior Profiles (>3 Yrs)", len(df[df['Exp'].str.contains(r'[3-9]|[1-9][0-9]', na=False)]))
    c4.metric("⚠️ Duplicates Blocked", len(dupes))
    
    st.markdown("<br>", unsafe_allow_html=True)

    def render_candidate_card(row, index, tab_key):
        with st.expander(f"{'🏆' if row['Score'] > 75 else '👤'} {row['Name']} | 🎯 Match: {row['Score']}% | 💼 {row['Domain']}"):
            rc1, rc2, rc3 = st.columns([1, 1, 1])
            rc1.markdown(f"**📧 Email:** `{row['Email']}`\n\n**📞 Phone:** `{row['Phone']}`")
            rc2.markdown(f"**⏳ Experience:** `{row['Exp']}`\n\n**⚡ Key Skills:** `{row['Skills']}`")
            rc3.progress(row['Score']/100, text=f"Algorithm Suitability: {row['Score']}%")
            st.markdown("---")
            st.text_area("Parsed Raw Vector Data", row['Raw'][:1000] + "...", height=150, key=f"txt_{tab_key}_{index}")

    t1, t2, t3, t4 = st.tabs(["📊 Ranked Talent Pool", "👨‍💻 Experienced Leads", "🌱 Entry Level / Freshers", "🚨 Deduplication Alerts"])

    with t1:
        for i, row in df.iterrows(): render_candidate_card(row, i, "t1")
    with t2:
        exp_df = df[~df['Exp'].str.contains("Fresher", case=False, na=False)]
        if exp_df.empty: st.info("No experienced candidates found in this batch.")
        for i, row in exp_df.iterrows(): render_candidate_card(row, i, "t2")
    with t3:
        fresh_df = df[df['Exp'].str.contains("Fresher", case=False, na=False)]
        if fresh_df.empty: st.info("No freshers found in this batch.")
        for i, row in fresh_df.iterrows(): render_candidate_card(row, i, "t3")
    with t4:
        if dupes.empty: 
            st.success("✅ Database Integrity Verified! No redundant applications found.")
        else:
            st.warning("⚠️ Alert: The system has quarantined the following redundant applications.")
            for i, row in dupes.iterrows(): render_candidate_card(row, i, "t4")
else:
    st.info("👈 System is idling. Please upload candidate vectors (PDFs) from the Control Panel to begin processing.")
