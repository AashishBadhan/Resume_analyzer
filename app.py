import streamlit as st
import fitz
import re
import os
import zipfile
import gdown
import pandas as pd
import joblib
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI ATS Dashboard", page_icon="👔", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: 600; background-color: #2e66ff; color: white; border: none; padding: 0.5rem 1rem;}
    .stButton>button:hover {background-color: #1a4cdb;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_model_if_missing():
    if not os.path.exists('./distilbert_resume_model'):
        st.info("First time setup: Downloading AI Model (takes 1-2 minutes)...")
        file_id = '1f4lGmcd-U5d5fgqOgV2HHdZi7E1d_Gjj'
        url = f'https://drive.google.com/uc?id={file_id}'
        output = 'model.zip'
        gdown.download(url, output, quiet=False)
        
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(output) 
        st.success("Model downloaded successfully!")

# Is function ko call karna zaroori hai
download_model_if_missing()

def load_ai_model():
    le = joblib.load('label_encoder.pkl')
    bert_analyzer = pipeline(
        "text-classification", 
        model='./distilbert_resume_model', 
        tokenizer='./distilbert_resume_model'
    )
    return le, bert_analyzer

le, bert_analyzer = load_ai_model()

def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_info(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    email = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phone = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    
    return text, email[0] if email else "N/A", phone[0] if phone else "N/A"

def get_match_score(jd, resume):
    if not jd or not resume: 
        return 0
    vectors = TfidfVectorizer().fit_transform([jd, resume])
    return cosine_similarity(vectors)[0][1] * 100

st.markdown("<h1>👔 AI Applicant Tracking System</h1>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ ATS Settings")
    jd_input = st.text_area("Paste Job Description (JD)", height=250, placeholder="Enter required skills...")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    candidates = []
    
    with st.spinner("🧠 AI is analyzing resumes..."):
        for file in uploaded_files:
            raw_text, email, phone = extract_info(file)
            cleaned = clean_text(raw_text)
            
            if len(cleaned) < 50:
                category = "Unreadable/Invalid Format"
                score = 0
            else:
                prediction = bert_analyzer(cleaned, truncation=True, max_length=512)
                label_id = int(prediction[0]['label'].split('_')[-1])
                category = le.inverse_transform([label_id])[0]
                score = get_match_score(clean_text(jd_input), cleaned) if jd_input else 0
            
            candidates.append({
                "Name": file.name.replace('.pdf', ''),
                "Category": category,
                "Match Score (%)": round(score, 2),
                "Email": email,
                "Phone": phone,
                "Text": cleaned
            })

    df = pd.DataFrame(candidates).sort_values(by="Match Score (%)", ascending=False)
    
    valid_emails = df[df['Email'] != 'N/A']
    dupes = valid_emails[valid_emails.duplicated(subset=['Email'], keep=False)]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants", len(df))
    col2.metric("Highest Match Score", f"{df['Match Score (%)'].max()}%" if not df.empty else "0%")
    col3.metric("Duplicates Detected", len(dupes))
    
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🏆 Full Time", "🎓 Interns", "⚠️ Duplicates", "💼 Experienced", "🌱 Freshers", "⏳ Internship History"
    ])

    with t1:
        valid_df = df[df['Category'] != "Unreadable/Invalid Format"]
        st.dataframe(valid_df.drop(columns=['Text']), use_container_width=True, hide_index=True)
        if st.button("✉️ Send Offer/Interview Email"):
            st.success("Drafts created!")

    with t2:
        interns = df[df['Text'].str.contains(r'\b(intern|internship|trainee)\b', case=False, regex=True) & ~df['Text'].str.contains(r'\b(managed|mentored|led)\b.*\binterns\b', case=False, regex=True)]
        st.dataframe(interns.drop(columns=['Text']), use_container_width=True, hide_index=True)

    with t3:
        st.dataframe(dupes.drop(columns=['Text']), use_container_width=True, hide_index=True)
        if dupes.empty:
            st.info("No duplicates found.")

    with t4:
        exp = df[df['Text'].str.contains(r'\b[1-9][0-9]?\+?\s*(years|yrs)\b', case=False, regex=True)]
        st.dataframe(exp.drop(columns=['Text']), use_container_width=True, hide_index=True)

    with t5:
        fresh = df[~df['Text'].str.contains(r'\b[1-9][0-9]?\+?\s*(years|yrs)\b', case=False, regex=True)]
        st.dataframe(fresh.drop(columns=['Text']), use_container_width=True, hide_index=True)

    with t6:
        hist = df[df['Text'].str.contains(r'\binternship\b', case=False, regex=True)]
        st.dataframe(hist.drop(columns=['Text']), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Detailed Candidate View")
    selected_candidate = st.selectbox("Select a candidate to view full resume text:", df['Name'].tolist())
    if selected_candidate:
        resume_content = df[df['Name'] == selected_candidate]['Text'].values[0]
        st.text_area(f"Resume Text for {selected_candidate}:", value=resume_content, height=400)

else:
    st.info("👈 Please upload candidate resumes (PDF) from the sidebar.")
