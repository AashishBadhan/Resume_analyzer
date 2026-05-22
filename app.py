import os
import re
import fitz
import joblib
import shutil
import zipfile
import gdown
import pandas as pd
import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "processed" not in st.session_state:
    st.session_state.processed = False

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()


def toggle_theme():
    st.session_state.theme = (
        "light" if st.session_state.theme == "dark" else "dark"
    )

# --- BASE CSS ---
base_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header {
    visibility: hidden;
}
.stApp {
    overflow-x: hidden;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
section[data-testid="stSidebar"] {
    min-width: 330px !important;
    max-width: 330px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: -2px;
}
.sub-title {
    text-align: center;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.card {
    padding: 1.4rem;
    border-radius: 24px;
    margin-bottom: 1rem;
}
.stButton > button {
    width: 100%;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    border: none;
    font-weight: 700;
    transition: 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
}
div[data-testid="metric-container"] {
    border-radius: 24px;
    padding: 1rem;
}
.resume-box {
    border-radius: 24px;
    padding: 20px;
    margin-bottom: 18px;
    transition: 0.3s ease;
}
.resume-box:hover {
    transform: translateY(-4px);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 600;
}
.stProgress > div > div > div > div {
    border-radius: 20px;
}
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#22d3ee, #06b6d4);
    border-radius: 20px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
</style>
"""

# --- DARK THEME CSS ---
dark_css = """
<style>
.stApp {
    background: #050816;
    color: #ffffff;
}
.main-title {
    background: linear-gradient(90deg, #22d3ee, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(34,211,238,0.25);
}
.sub-title {
    color: #cbd5e1;
}

/* Fixed Sidebar CSS for Dark Theme */
section[data-testid="stSidebar"] {
    background: #0b1220 !important;
    border-right: 1px solid rgba(34,211,238,0.15);
}
/* Specifically target text elements instead of using wildcard '*' */
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}

/* Fix File Uploader UI in Dark Theme */
section[data-testid="stFileUploadDropzone"] {
    background-color: #111827 !important;
    border: 1px dashed rgba(34,211,238,0.4) !important;
    border-radius: 14px;
}

.card, .resume-box, div[data-testid="metric-container"], .stTabs [data-baseweb="tab"] {
    background: #0f172a;
    border: 1px solid rgba(34,211,238,0.10);
    box-shadow: 0 0 18px rgba(34,211,238,0.06);
    backdrop-filter: blur(10px);
}
.resume-box:hover {
    border: 1px solid rgba(34,211,238,0.30);
    box-shadow: 0 0 25px rgba(34,211,238,0.12);
}
.stButton > button {
    background: linear-gradient(90deg, #22d3ee, #06b6d4);
    color: #041018 !important;
    font-weight: 700;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #67e8f9, #22d3ee);
}
.stTextArea textarea, .stTextInput input {
    background: #111827 !important;
    color: white !important;
    border: 1px solid rgba(34,211,238,0.18) !important;
    border-radius: 14px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #22d3ee, #06b6d4) !important;
    color: #041018 !important;
    font-weight: 700;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22d3ee, #06b6d4);
}
</style>
"""

# --- LIGHT THEME CSS ---
light_css = """
<style>
.stApp {
    background: #f8fafc;
    color: #0f172a;
}
.main-title {
    background: linear-gradient(90deg, #0891b2, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* Fixed Sidebar CSS for Light Theme */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(6,182,212,0.15);
}
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] label {
    color: #0f172a !important;
}

/* Fix File Uploader UI in Light Theme */
section[data-testid="stFileUploadDropzone"] {
    background-color: #f1f5f9 !important;
    border: 1px dashed rgba(6,182,212,0.4) !important;
    border-radius: 14px;
}

.card, .resume-box, div[data-testid="metric-container"], .stTabs [data-baseweb="tab"] {
    background: white;
    border: 1px solid rgba(6,182,212,0.10);
    box-shadow: 0 0 12px rgba(6,182,212,0.06);
}
.stButton > button {
    background: linear-gradient(90deg, #22d3ee, #06b6d4);
    color: white !important;
}
.stTextArea textarea, .stTextInput input {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid rgba(6,182,212,0.18) !important;
    border-radius: 14px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #22d3ee, #06b6d4) !important;
    color: white !important;
}
</style>
"""


st.markdown(base_css, unsafe_allow_html=True)

if st.session_state.theme == "dark":
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def ensure_model_exists():
    model_dir = "distilbert_resume_model"
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return model_dir

    zip_path = "resume_model.zip"
    extract_dir = "temp_model"
    file_id = "1cjxek02nIA36_8lmC-B66HwYjPR6wsyS"

    try:
        with st.spinner("Downloading AI model..."):
            gdown.download(id=file_id, output=zip_path, quiet=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            found_path = None
            for root, _, files in os.walk(extract_dir):
                if "config.json" in files:
                    found_path = root
                    break

            if not found_path:
                st.error("Model extraction failed")
                st.stop()

            os.makedirs(model_dir, exist_ok=True)
            for item in os.listdir(found_path):
                shutil.move(
                    os.path.join(found_path, item),
                    os.path.join(model_dir, item)
                )
            shutil.rmtree(extract_dir)
            if os.path.exists(zip_path):
                os.remove(zip_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    return model_dir


@st.cache_resource(show_spinner=False)
def load_models():
    model_path = ensure_model_exists()
    try:
        label_encoder = joblib.load("label_encoder.pkl")
    except Exception:
        st.error("label_encoder.pkl missing")
        st.stop()

    try:
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path
        )
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    return label_encoder, classifier


label_encoder, classifier = load_models()


def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"[^\w\s@.+-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_email(text):
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return emails[0] if emails else "Not Found"


def extract_phone(text):
    pattern = r"(?:\+91[-\s]?)?[6-9]\d{9}"
    phones = re.findall(pattern, text)
    if phones:
        number = re.sub(r"\D", "", phones[0])
        if len(number) == 10:
            return f"+91 {number}"
    return "Not Found"


def extract_name(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    blacklist = [
        "resume", "curriculum", "vitae", "developer",
        "engineer", "email", "phone", "linkedin", "github"
    ]
    for line in lines[:12]:
        lower = line.lower()
        if any(word in lower for word in blacklist):
            continue
        if len(line.split()) <= 4 and not re.search(r"\d", line):
            return line.title()
    return "Unknown Candidate"


def extract_experience(text):
    text = text.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|year|yrs|yr)", text)
    if not matches:
        return "Fresher"
    years = max(float(x) for x in matches)
    return f"{years:.1f} Years"


def calculate_match_score(jd, resume):
    if not jd.strip() or not resume.strip():
        return 0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([jd, resume])
        similarity = cosine_similarity(vectors)[0][1]
        return round(float(similarity * 100), 2)
    except Exception:
        return 0


def read_pdf(file):
    try:
        file.seek(0)
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        return text.strip()
    except Exception:
        return None


def analyze_resume(file, jd_text):
    raw_text = read_pdf(file)
    if not raw_text or len(raw_text) < 30:
        return {
            "File": file.name,
            "Name": "Unreadable Resume",
            "Email": "Not Found",
            "Phone": "Not Found",
            "Experience": "Unknown",
            "Domain": "Invalid PDF",
            "Score": 0,
            "Preview": "Could not extract text"
        }
    
    cleaned = clean_text(raw_text)
    
    try:
        prediction = classifier(cleaned[:4000], truncation=True, max_length=512)
        label = prediction[0]["label"]
        if "_" in label:
            label_id = int(label.split("_")[-1])
            category = label_encoder.inverse_transform([label_id])[0]
        else:
            category = label
    except Exception:
        category = "Unknown"

    score = calculate_match_score(clean_text(jd_text), cleaned)

    return {
        "File": file.name,
        "Name": extract_name(raw_text),
        "Email": extract_email(raw_text),
        "Phone": extract_phone(raw_text),
        "Experience": extract_experience(raw_text),
        "Domain": category,
        "Score": score,
        "Preview": raw_text[:2500]
    }


st.markdown('<div class="main-title">AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced AI Candidate Screening & Resume Ranking</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## ⚡ Control Panel")
    st.button("🌓 Toggle Theme", on_click=toggle_theme, use_container_width=True)
    st.markdown("---")
    
    jd_input = st.text_area("📄 Job Description", height=220)
    uploaded_files = st.file_uploader("📂 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} Resume(s) Uploaded")
    
    st.markdown("---")
    analyze_button = st.button("🚀 Analyze Resumes", use_container_width=True)


if analyze_button:
    if not uploaded_files:
        st.warning("Please upload resumes")
    else:
        progress = st.progress(0)
        results = []
        
        for index, file in enumerate(uploaded_files):
            result = analyze_resume(file, jd_input)
            results.append(result)
            progress.progress((index + 1) / len(uploaded_files))
            
        progress.empty()
        df = pd.DataFrame(results)
        df = df.sort_values(by="Score", ascending=False)
        st.session_state.results = df
        st.session_state.processed = True


if st.session_state.processed and not st.session_state.results.empty:
    df = st.session_state.results
    top_score = int(df["Score"].max())
    avg_score = round(df["Score"].mean(), 1)
    duplicate_count = len(df[df["Email"].duplicated(keep=False)])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resumes", len(df))
    col2.metric("Top Match", f"{top_score}%")
    col3.metric("Average Score", f"{avg_score}%")
    col4.metric("Duplicates", duplicate_count)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tabs = st.tabs(["All Candidates", "Top Matches", "Freshers"])

    def render_cards(dataframe):
        if dataframe.empty:
            st.info("No candidates found")
            return
            
        for _, row in dataframe.iterrows():
            emoji = "🏆" if row["Score"] >= 75 else "👨‍💻"
            st.markdown('<div class="resume-box">', unsafe_allow_html=True)
            c1, c2 = st.columns([3,1])
            with c1:
                st.markdown(f"## {emoji} {row['Name']}")
                st.markdown(f"**Domain:** {row['Domain']}")
                st.markdown(f"**Experience:** {row['Experience']}")
                st.markdown(f"**Email:** {row['Email']}")
                st.markdown(f"**Phone:** {row['Phone']}")
                st.markdown(f"**Resume:** {row['File']}")
            with c2:
                st.metric("Match Score", f"{row['Score']}%")
                st.progress(min(float(row["Score"]) / 100, 1.0))
            
            with st.expander("Preview Resume"):
                st.text(row["Preview"])
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[0]:
        render_cards(df)
    with tabs[1]:
        render_cards(df[df["Score"] >= 60])
    with tabs[2]:
        render_cards(df[df["Experience"].str.contains("Fresher", case=False, na=False)])

    csv = df.drop(columns=["Preview"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Report",
        data=csv,
        file_name="resume_analysis_report.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.markdown(
        """
        <div class="card">
            <h2>🚀 Features</h2>
            <ul style="line-height:2;">
                <li>AI Resume Classification</li>
                <li>Modern Dashboard UI</li>
                <li>JD Similarity Matching</li>
                <li>Resume Ranking</li>
                <li>Duplicate Detection</li>
                <li>PDF Resume Parsing</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
