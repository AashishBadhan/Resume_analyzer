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
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {background-color: transparent !important;}
.main-header {font-size: 42px; font-weight: 800; background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 10px;}
.sub-header {text-align: center; font-size: 18px; margin-bottom: 40px;}
.stButton>button {width: 100%; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; border: 1px solid rgba(255,255,255,0.1);}
.stButton>button:hover {transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2);}
[data-testid="stMetricValue"] {font-size: 28px !important; font-weight: 800 !important;}
[data-testid="stMetricLabel"] {font-size: 14px !important; font-weight: 600 !important;}
div[data-testid="metric-container"] {border-radius: 12px; padding: 20px; text-align: center; transition: all 0.3s ease;}
.streamlit-expanderHeader {font-size: 16px; font-weight: 700; border-radius: 8px; transition: all 0.3s ease;}
</style>
"""

if st.session_state.theme == 'Dark':
    theme_css = """
    <style>
    .stApp {background-color: #0e1117; color: #fafafa;}
    .sub-header {color: #aaa;}
    [data-testid="stSidebar"] {background-color: #262730; border-right: 1px solid #333;}
    div[data-testid="metric-container"] {background: #1e1e1e; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.4);}
    [data-testid="stMetricValue"] {color: #4facfe !important;}
    [data-testid="stMetricLabel"] {color: #aaa !important;}
    .streamlit-expanderHeader {color: #4facfe; background-color: #1e1e1e;}
    .stButton>button {background: linear-gradient(90deg, #1e3c72 0%, #1e1e1e 100%); color: white;}
    </style>
    """
else:
    theme_css = """
    <style>
    .stApp {background-color: #ffffff; color: #111111;}
    .sub-header {color: #666;}
    [data-testid="stSidebar"] {background-color: #f8f9fa; border-right: 1px solid #e0e0e0;}
    div[data-testid="metric-container"] {background: white; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
    [data-testid="stMetricValue"] {color: #1e3c72 !important;}
    [data-testid="stMetricLabel"] {color: #555 !important;}
    .streamlit-expanderHeader {color: #1e3c72; background-color: #f1f5f9;}
    .stButton>button {background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white;}
    </style>
    """

st.markdown(base_css + theme_css, unsafe_allow_html=True)

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
    le = joblib.load('label_encoder.pkl')
    base_model_path = os.path.abspath('./distilbert_resume_model')
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
    phone_pattern = re.compile(r'(?:(?:\+|0{0,2})91[\s\-]?)?(?:(?:\d{5}[\s\-]?\d{5})|(?:\d{3}[\s\-]?\d{3}[\s\-]?\d{4})|(?:\d{10}))')
    matches = phone_pattern.findall(text)
    for match in matches:
        clean_match = re.sub(r'[\s\-]', '', match)
        if len(clean_match) >= 10 and len(clean_match) <= 13:
            return "+" + clean_match if not clean_match.startswith('+') and len(clean_match) > 10 else clean_match
    return "Not Found"

def extract_email(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not Found"

def extract_name(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:20]:
        match = re.search(r'(?i)\b(?:name|first name|candidate)\s*[:\-]\s*([A-Za-z\s]+)', line)
        if match: return match.group(1).title().strip()
    ignore_words = ['resume', 'cv', 'curriculum', 'vitae', 'profile', 'email', 'phone', 'address', 'mobile', 'dob', 'date', 'page', 'linkedin', 'github']
    for line in lines[:10]:
        if len(line.split()) > 4 or len(line) > 30: continue
        if any(word in line.lower() for word in ignore_words): continue
        if re.search(r'\d', line): continue
        if '@' in line or '.com' in line: continue
        words = line.split()
        if 1 <= len(words) <= 3:
            return line.title()
    return "Unknown Candidate"

def extract_experience(text):
    exp_pattern = r'(?i)\b([0-9]{1,2}(?:\.[0-9]{1,2})?)\s*(?:\+)?\s*(?:years?|yrs?|months?)\b'
    matches = re.findall(exp_pattern, text)
    if matches:
        max_exp = max([float(m) for m in matches])
        return f"{max_exp} Years" if max_exp > 1 else "Fresher (< 1 Year)"
    return "Fresher / Not Specified"

def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    raw_text = ""
    for page in doc: raw_text += page.get_text("text")
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    name = extract_name(raw_text)
    exp = extract_experience(raw_text)
    return raw_text, email, phone, name, exp

def get_match_score(jd, resume):
    if not jd or not resume: return 0
    vectors = TfidfVectorizer(stop_words='english').fit_transform([jd, resume])
    score = cosine_similarity(vectors)[0][1] * 100
    return round(score, 2)

st.markdown("<div class='main-header'>AI-Powered Resume Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Next-Gen Deep Learning Resume Categorization & Ranking Engine</div>", unsafe_allow_html=True)

with st.sidebar:
    st.button("🌓 Toggle Light/Dark Mode", on_click=toggle_theme)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135692.png", width=80)
    st.markdown("### ⚙️ Recruitment Control Panel")
    jd_input = st.text_area("🎯 Job Description (JD)", height=250, placeholder="Paste required skills, tech stack, and role details here...")
    st.markdown("---")
    uploaded_files = st.file_uploader("📂 Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    candidates = []
    progress_text = "🧠 DistilBERT is analyzing candidates..."
    my_bar = st.progress(0, text=progress_text)
    for i, file in enumerate(uploaded_files):
        raw_text, email, phone, name, experience = process_pdf(file)
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
            "Exp": experience, "Score": score, "Email": email,
            "Phone": phone, "Raw": raw_text
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
    c1.metric("📄 Total Scanned", len(df))
    c2.metric("🎯 Top Match Score", f"{df['Score'].max()}%" if not df.empty else "0%")
    c3.metric("💼 Senior Pros (>3 Yrs)", len(df[df['Exp'].str.contains(r'[3-9]|[1-9][0-9]', na=False)]))
    c4.metric("⚠️ Duplicates Found", len(dupes))
    
    st.markdown("<br>", unsafe_allow_html=True)

    def render_candidate_card(row, index, tab_key):
        with st.expander(f"{'🏆' if row['Score'] > 75 else '👤'} {row['Name']} | 🎯 Match: {row['Score']}% | 💼 {row['Domain']}"):
            rc1, rc2, rc3 = st.columns([1, 1, 1])
            rc1.markdown(f"**📧 Email:** `{row['Email']}`\n\n**📞 Phone:** `{row['Phone']}`")
            rc2.markdown(f"**⏳ Experience:** `{row['Exp']}`\n\n**📂 File:** `{row['File Name']}`")
            rc3.progress(row['Score']/100, text=f"Suitability: {row['Score']}%")
            st.markdown("---")
            st.text_area("Extracted Resume Text (Preview)", row['Raw'][:1000] + "...", height=150, key=f"txt_{tab_key}_{index}")

    t1, t2, t3, t4 = st.tabs(["📊 All Ranked Candidates", "👨‍💻 Experienced Only", "🌱 Freshers", "🚨 Duplicate Alerts"])

    with t1:
        for i, row in df.iterrows(): render_candidate_card(row, i, "t1")
    with t2:
        exp_df = df[~df['Exp'].str.contains("Fresher", case=False, na=False)]
        if exp_df.empty: st.info("No experienced candidates found.")
        for i, row in exp_df.iterrows(): render_candidate_card(row, i, "t2")
    with t3:
        fresh_df = df[df['Exp'].str.contains("Fresher", case=False, na=False)]
        if fresh_df.empty: st.info("No freshers found.")
        for i, row in fresh_df.iterrows(): render_candidate_card(row, i, "t3")
    with t4:
        if dupes.empty: 
            st.success("✅ Database Clean! No redundant applications found.")
        else:
            st.warning("⚠️ Warning: The following candidates submitted multiple applications.")
            for i, row in dupes.iterrows(): render_candidate_card(row, i, "t4")
else:
    st.info("👈 System Ready. Please upload candidate resumes in PDF format from the sidebar to begin processing.")
