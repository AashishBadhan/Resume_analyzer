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
    page_title="AI Resume Analyzer Pro",
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
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


base_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    min-width: 340px !important;
    max-width: 340px !important;
    border-right: 1px solid rgba(128,128,128,0.15);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

section[data-testid="stSidebar"] .stTextArea textarea {
    min-height: 220px !important;
}


.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.5rem;
}

.sub-title {
    text-align: center;
    font-size: 1rem;
    opacity: 0.8;
    margin-bottom: 2rem;
}

.card {
    padding: 1.2rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}

.stButton > button {
    width: 100% !important;
    border-radius: 14px;
    font-weight: 700;
    border: none;
    transition: 0.25s ease;
    padding: 0.75rem 1rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

div[data-testid="metric-container"] {
    border-radius: 18px;
    padding: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}

.resume-box {
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 14px;
}

.highlight {
    font-weight: 700;
}
</style>
"""


dark_css = """
<style>
.stApp {
    background: #0b1120;
    color: #ffffff;
}

.main-title {
    background: linear-gradient(90deg,#38bdf8,#8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card,
div[data-testid="metric-container"],
.resume-box,
.stTabs [data-baseweb="tab"] {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.08);
}

.stButton > button {
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
}

.stTextArea textarea,
.stTextInput input {
    background: #111827 !important;
    color: white !important;
}
</style>
"""


light_css = """
<style>
.stApp {
    background: #f8fafc;
    color: #0f172a;
}

.main-title {
    background: linear-gradient(90deg,#2563eb,#9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card,
div[data-testid="metric-container"],
.resume-box,
.stTabs [data-baseweb="tab"] {
    background: white;
    border: 1px solid rgba(15,23,42,0.08);
}

.stButton > button {
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
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
    config_path = os.path.join(model_dir, "config.json")

    if os.path.exists(config_path):
        return model_dir

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

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
        st.error(f"Failed to initialize model: {e}")
        st.stop()

    return model_dir


@st.cache_resource(show_spinner=False)
def load_models():
    model_path = ensure_model_exists()

    try:
        label_encoder = joblib.load("label_encoder.pkl")
    except:
        st.error("label_encoder.pkl file missing")
        st.stop()

    try:
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path
        )
    except Exception as e:
        st.error(f"Unable to load AI model: {e}")
        st.stop()

    return label_encoder, classifier


label_encoder, classifier = load_models()


def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'[^\w\s@.+-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()



def extract_email(text):
    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    return emails[0] if emails else "Not Found"



def extract_phone(text):
    pattern = r'(?:\+91[-\s]?)?[6-9]\d{9}'
    phones = re.findall(pattern, text)

    if phones:
        number = re.sub(r'\D', '', phones[0])

        if len(number) == 10:
            return f'+91 {number}'

        return f'+{number}'

    return "Not Found"



def extract_name(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    blacklist = [
        'resume', 'curriculum', 'vitae', 'developer', 'engineer',
        'email', 'phone', 'linkedin', 'github', 'profile'
    ]

    for line in lines[:12]:
        line_clean = line.lower()

        if any(word in line_clean for word in blacklist):
            continue

        if len(line.split()) <= 4 and not re.search(r'\d', line):
            return line.title()

    return "Unknown Candidate"



def extract_experience(text):
    text = text.lower()

    patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*(?:years|year|yrs|yr)',
        r'(\d+(?:\.\d+)?)\+?\s*(?:months|month)'
    ]

    values = []

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                values.append(float(match))
            except:
                pass

    if not values:
        return "Fresher"

    highest = max(values)

    if highest < 1:
        return "Fresher"

    return f"{highest:.1f} Years"



def calculate_match_score(jd, resume):
    if not jd.strip() or not resume.strip():
        return 0

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([jd, resume])
        similarity = cosine_similarity(vectors)[0][1]
        return round(float(similarity * 100), 2)
    except:
        return 0



def read_pdf(file):
    text = ""

    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")

        for page in pdf:
            text += page.get_text()

        pdf.close()

    except:
        return None

    return text.strip()



def analyze_resume(file, jd_text):
    raw_text = read_pdf(file)

    if not raw_text or len(raw_text.strip()) < 30:
        return {
            "File": file.name,
            "Name": "Unreadable Resume",
            "Email": "Not Found",
            "Phone": "Not Found",
            "Experience": "Unknown",
            "Domain": "Invalid PDF",
            "Score": 0,
            "Preview": "Unable to extract text"
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

    except:
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
st.markdown('<div class="sub-title">Advanced Resume Ranking, Screening and AI-Based Candidate Matching</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    theme_col1, theme_col2 = st.columns([1, 2])

    with theme_col1:
        st.button(
            "🌓",
            on_click=toggle_theme,
            use_container_width=True
        )

    with theme_col2:
        st.markdown(
            f"<div style='padding-top:8px;font-weight:600;'>Current Theme: {st.session_state.theme.title()}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 📄 Job Description")

    jd_input = st.text_area(
        "Job Description",
        height=240,
    )

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF resumes"
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} Resume(s) Uploaded")

    st.markdown("---")

    analyze_button = st.button(
        "Analyze Resumes",
        use_container_width=True
    )


if analyze_button:
    if not uploaded_files:
        st.warning("Please upload at least one resume")
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

    duplicate_emails = df[
        (df["Email"] != "Not Found") &
        (df["Email"].duplicated(keep=False))
    ]

    duplicate_phones = df[
        (df["Phone"] != "Not Found") &
        (df["Phone"].duplicated(keep=False))
    ]

    duplicates = pd.concat([
        duplicate_emails,
        duplicate_phones
    ]).drop_duplicates()

    senior_candidates = df[
        ~df["Experience"].str.contains("Fresher", case=False, na=False)
    ]

    freshers = df[
        df["Experience"].str.contains("Fresher", case=False, na=False)
    ]

    top_score = int(df["Score"].max()) if not df.empty else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Resumes", len(df))
    col2.metric("Top Match", f"{top_score}%")
    col3.metric("Experienced", len(senior_candidates))
    col4.metric("Duplicates", len(duplicates))

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "All Candidates",
        "Top Matches",
        "Freshers",
        "Duplicates"
    ])

    def render_cards(dataframe):
        if dataframe.empty:
            st.info("No candidates found")
            return

        for _, row in dataframe.iterrows():
            emoji = "🏆" if row['Score'] >= 75 else "👨‍💻"

            with st.container():
                st.markdown('<div class="resume-box">', unsafe_allow_html=True)

                c1, c2 = st.columns([3, 1])

                with c1:
                    st.markdown(f"### {emoji} {row['Name']}")
                    st.markdown(f"**Domain:** {row['Domain']}")
                    st.markdown(f"**Experience:** {row['Experience']}")
                    st.markdown(f"**Email:** {row['Email']}")
                    st.markdown(f"**Phone:** {row['Phone']}")
                    st.markdown(f"**Resume:** {row['File']}")

                with c2:
                    st.metric("Match Score", f"{row['Score']}%")
                    st.progress(min(float(row['Score']) / 100, 1.0))

                with st.expander("Resume Preview"):
                    st.text(row['Preview'])

                st.markdown('</div>', unsafe_allow_html=True)

    with tabs[0]:
        render_cards(df)

    with tabs[1]:
        render_cards(df[df['Score'] >= 60])

    with tabs[2]:
        render_cards(freshers)

    with tabs[3]:
        if duplicates.empty:
            st.success("No duplicate resumes detected")
        else:
            render_cards(duplicates)

    csv = df.drop(columns=["Preview"]).to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Analysis Report",
        data=csv,
        file_name="resume_analysis_report.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.markdown(
        """
        <div class="card">
            <h3>Features</h3>
            <ul>
                <li>AI-based resume classification</li>
                <li>Automatic email and phone extraction</li>
                <li>Duplicate resume detection</li>
                <li>JD similarity matching</li>
                <li>Modern responsive UI</li>
                <li>Exportable candidate report</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
