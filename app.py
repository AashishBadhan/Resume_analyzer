import streamlit as st
import fitz
import re
import pandas as pd
import joblib
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

@st.cache_resource
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

st.title("AI RESUME ANALYZER")

with st.sidebar:
    st.header("Job Settings")
    jd_input = st.text_area("Paste Job Description (JD)", height=200)
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    candidates = []
    
    for file in uploaded_files:
        raw_text, email, phone = extract_info(file)
        cleaned = clean_text(raw_text)
        
        if not cleaned:
            cleaned = "empty document format"
        
        prediction = bert_analyzer(cleaned, truncation=True, max_length=512)
        label_id = int(prediction[0]['label'].split('_')[-1])
        category = le.inverse_transform([label_id])[0]
        
        score = get_match_score(clean_text(jd_input), cleaned) if jd_input else 0
        
        candidates.append({
            "Name": file.name,
            "Category": category,
            "Match Score (%)": round(score, 2),
            "Email": email,
            "Phone": phone,
            "Text": cleaned
        })

    df = pd.DataFrame(candidates).sort_values(by="Match Score (%)", ascending=False)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Full Time", "Interns", "Duplicates", "Experienced", "Freshers", "Internship History"
    ])

    with t1:
        st.subheader("Selected Candidates (Full-Time)")
        st.dataframe(df)
        if st.button("Send Email to All"):
            st.success("Drafts created for all selected candidates!")

    with t2:
        st.subheader("Potential Interns")
        interns = df[df['Text'].str.contains('intern|trainee', case=False)]
        st.dataframe(interns)

    with t3:
        st.subheader("Duplicate Applications")
        dupes = df[df.duplicated(subset=['Email'], keep=False)]
        st.dataframe(dupes)

    with t4:
        st.subheader("Most Experienced")
        exp = df[df['Text'].str.contains('experience|years', case=False)]
        st.dataframe(exp)

    with t5:
        st.subheader("Freshers")
        fresh = df[~df['Text'].str.contains('experience|years', case=False)]
        st.dataframe(fresh)

    with t6:
        st.subheader("Candidates with Prior Internships")
        hist = df[df['Text'].str.contains('internship', case=False)]
        st.dataframe(hist)

else:
    st.info("Please upload resumes from the sidebar.")