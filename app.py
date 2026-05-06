import streamlit as st
import fitz  # PyMuPDF
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


st.set_page_config(
    page_title="AI ATS Dashboard", 
    page_icon="👔", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;} 
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: 600; background-color: #2e66ff; color: white; border: none; padding: 0.5rem 1rem;}
    .stButton>button:hover {background-color: #1a4cdb;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center;}
    /* Expander Styling */
    .streamlit-expanderHeader {font-size: 16px; font-weight: bold; color: #1a4cdb;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_model_if_missing():
    model_dir = os.path.abspath('./distilbert_resume_model')
    config_file = os.path.join(model_dir, 'config.json')
    
    if not os.path.exists(config_file):
        # Purana aadha-adhura folder hatao
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        file_id = '1cjxek02nIA36_8lmC-B66HwYjPR6wsyS' 
        output = 'model.zip'
        temp_extract_dir = os.path.abspath('./temp_model_extract')
        
        try:
            gdown.download(id=file_id, output=output, quiet=False)
            
            if not zipfile.is_zipfile(output):
                st.error("🚨 Error: Downloaded file is not a valid ZIP. Please check Drive link permissions.")
                st.stop()
        
            # 1. Zip ko ek temporary folder mein extract karo
            os.makedirs(temp_extract_dir, exist_ok=True)
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            os.remove(output)
            
            # 2. SMART SEARCH: config.json ko dhoondho (chahe kitna bhi andar chupa ho)
            found_model_path = None
            for root, dirs, files in os.walk(temp_extract_dir):
                if 'config.json' in files:
                    found_model_path = root
                    break
            
            if not found_model_path:
                st.error("🚨 Error: 'config.json' file not found inside the ZIP! Did you zip the right folder in Colab?")
                shutil.rmtree(temp_extract_dir)
                st.stop()
                
            # 3. Asli files ko nikal kar sahi jagah move karo
            os.makedirs(model_dir, exist_ok=True)
            for item in os.listdir(found_model_path):
                source_item = os.path.join(found_model_path, item)
                dest_item = os.path.join(model_dir, item)
                shutil.move(source_item, dest_item)
                
            # 4. Kachra (temp folder) saaf karo
            shutil.rmtree(temp_extract_dir)
            
            st.success("✅ AI Model correctly located, extracted, and activated!")
            
        except Exception as e:
            st.error(f"🚨 Download/Extraction failed: {str(e)}")
            st.stop()

download_model_if_missing()

@st.cache_resource
def load_ai_model():
    le = joblib.load('label_encoder.pkl')
    base_model_path = os.path.abspath('./distilbert_resume_model')
    
    # Check if config.json is hidden inside nested folders
    actual_model_path = base_model_path
    if not os.path.exists(os.path.join(base_model_path, 'config.json')):
        for root, dirs, files in os.walk(base_model_path):
            if 'config.json' in files:
                actual_model_path = root
                break
                
    bert_analyzer = pipeline("text-classification", model=actual_model_path, tokenizer=actual_model_path)
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
    
    # Extract Name (Smart Heuristic)
    ignore_words = ["resume", "curriculum vitae", "cv", "bio-data"]
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
    extracted_name = "Unknown"
    for line in lines:
        if line.lower() not in ignore_words:
            extracted_name = line.title()
            break

    # Extract Experience
    exp_match = re.search(r'\b([1-9][0-9]?)\+?\s*(?:years|yrs)\b', text, re.IGNORECASE)
    experience = f"{exp_match.group(1)} Years" if exp_match else "Fresher"
    
    return text, email[0] if email else "N/A", phone[0] if phone else "N/A", extracted_name, experience

def get_match_score(jd, resume):
    if not jd or not resume: return 0
    vectors = TfidfVectorizer().fit_transform([jd, resume])
    return cosine_similarity(vectors)[0][1] * 100

st.markdown("<h1>👔 AI Resume Analyzer System</h1>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Recruitment Settings")
    jd_input = st.text_area("Paste Job Description (JD)", height=250, placeholder="Enter required skills...")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    candidates = []
    with st.spinner("🧠 AI is analyzing resumes..."):
        for file in uploaded_files:
            raw_text, email, phone, name, experience = extract_info(file)
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
                "File Name": file.name,
                "Extracted Name": name,
                "Predicted Domain": category,
                "Experience": experience,
                "JD Match Score (%)": round(score, 2),
                "Email": email,
                "Phone": phone,
                "Raw Text": raw_text
            })

    df = pd.DataFrame(candidates).sort_values(by="JD Match Score (%)", ascending=False)
    
    # Duplicate Detection
    email_dupes = df[(df['Email'] != 'N/A') & (df['Email'].duplicated(keep=False))]
    phone_dupes = df[(df['Phone'] != 'N/A') & (df['Phone'].duplicated(keep=False))]
    dupes = pd.concat([email_dupes, phone_dupes]).drop_duplicates(subset=['File Name'])
    
    # Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants", len(df))
    col2.metric("Highest JD Match", f"{df['JD Match Score (%)'].max()}%" if not df.empty else "0%")
    col3.metric("Duplicates Detected", len(dupes))
    st.markdown("<br>", unsafe_allow_html=True)

    # Reusable UI component to show candidates via Expanders
    def display_candidates(candidate_df):
        if candidate_df.empty:
            st.info("No candidates found in this category.")
            return
            
        for _, row in candidate_df.iterrows():
            with st.expander(f"👤 {row['Extracted Name']} | 💼 {row['Experience']} | 🎯 Score: {row['JD Match Score (%)']}%"):
                c1, c2 = st.columns(2)
                c1.write(f"**Predicted Domain:** {row['Predicted Domain']}")
                c1.write(f"**File Name:** {row['File Name']}")
                c2.write(f"**Email:** {row['Email']}")
                c2.write(f"**Phone:** {row['Phone']}")
                
                st.markdown("---")
                st.markdown("#### 📄 Extracted Resume Text")
                st.text_area("Resume Content", value=row['Raw Text'], height=250, disabled=True, label_visibility="collapsed", key=row['File Name'])

    # Smart Tabs
    t1, t2, t3, t4 = st.tabs(["🏆 All Matches", "💼 Experienced Pros", "🌱 Freshers", "⚠️ Duplicates"])

    with t1:
        display_candidates(df[df['Predicted Domain'] != "Unreadable/Invalid Format"])

    with t2:
        exp = df[df['Experience'] != "Fresher"]
        display_candidates(exp)

    with t3:
        fresh = df[df['Experience'] == "Fresher"]
        display_candidates(fresh)

    with t4:
        display_candidates(dupes)

else:
    st.info("👈 Waiting for action! Please upload candidate resumes from the sidebar to begin.")
    
