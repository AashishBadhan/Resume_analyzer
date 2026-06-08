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
    initial_sidebar_state="expanded",
)


# -----------------------------
# Session state
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "processed" not in st.session_state:
    st.session_state.processed = False

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "last_input_signature" not in st.session_state:
    st.session_state.last_input_signature = None


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


# -----------------------------
# Styling
# -----------------------------
base_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
/* Hide only Streamlit menu and footer.
   IMPORTANT: Do NOT hide header/toolbar/buttons because Streamlit keeps
   the sidebar open/reopen control there after the sidebar is collapsed. */
#MainMenu, footer {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
.stApp {
    overflow-x: hidden;
}

/* Responsive main page layout */
.block-container {
    width: 100% !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: clamp(1rem, 3vw, 3rem) !important;
    padding-right: clamp(1rem, 3vw, 3rem) !important;
}

/* Responsive sidebar width without breaking Streamlit's collapsed/open button. */
section[data-testid="stSidebar"][aria-expanded="true"] {
    width: min(330px, 92vw) !important;
}
section[data-testid="stSidebar"] .block-container {
    max-width: 100% !important;
    padding-top: 1rem;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

.main-title {
    font-size: clamp(2rem, 6vw, 3.5rem);
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: clamp(-1.5px, -0.2vw, -2px);
    line-height: 1.1;
    word-break: break-word;
}
.sub-title {
    text-align: center;
    font-size: clamp(0.9rem, 2.5vw, 1rem);
    margin-bottom: clamp(1rem, 3vw, 2rem);
}
.card {
    padding: clamp(1rem, 3vw, 1.4rem);
    border-radius: 24px;
    margin-bottom: 1rem;
    overflow-wrap: anywhere;
}
.stButton > button {
    width: 100%;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    border: none;
    font-weight: 700;
    transition: 0.3s ease;
    white-space: normal !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
}
div[data-testid="metric-container"] {
    border-radius: 24px;
    padding: clamp(0.8rem, 2vw, 1rem);
    min-height: 105px;
}
.resume-box {
    width: 100%;
    max-width: 100%;
    border-radius: 24px;
    padding: clamp(14px, 3vw, 20px);
    margin-bottom: 18px;
    transition: 0.3s ease;
    overflow: hidden;
    overflow-wrap: anywhere;
}
.resume-box:hover {
    transform: translateY(-4px);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 600;
    white-space: normal;
}
.stProgress > div > div > div > div {
    border-radius: 20px;
}
.small-muted {
    opacity: 0.8;
    font-size: 0.95rem;
}

/* Keep every component inside the visible width */
.element-container, .stMarkdown, .stText, .stDataFrame, .stDownloadButton {
    max-width: 100% !important;
}
img, svg, canvas, iframe {
    max-width: 100% !important;
}
pre, code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* Responsive columns and cards for tablets/mobiles */
@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .main-title {
        font-size: 2.4rem;
        letter-spacing: -1px;
    }
    .resume-box:hover, .stButton > button:hover {
        transform: none;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 12px;
        font-size: 0.9rem;
    }
}

@media (max-width: 640px) {
    .main-title {
        font-size: 2rem;
    }
    .sub-title {
        font-size: 0.9rem;
    }
    div[data-testid="metric-container"] {
        min-height: auto;
    }
    .resume-box h2, .resume-box h3 {
        font-size: 1.2rem !important;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 92vw !important;
    }
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
section[data-testid="stSidebar"] {
    background: #0b1220 !important;
    border-right: 1px solid rgba(34,211,238,0.15);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}
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
st.markdown(dark_css if st.session_state.theme == "dark" else light_css, unsafe_allow_html=True)


# -----------------------------
# Model loading
# -----------------------------
def safe_extract_zip(zip_ref, extract_dir):
    """Extract model zip safely and avoid accidental path traversal."""
    extract_abs = os.path.abspath(extract_dir)
    for member in zip_ref.infolist():
        member_path = os.path.abspath(os.path.join(extract_dir, member.filename))
        if not member_path.startswith(extract_abs + os.sep) and member_path != extract_abs:
            raise ValueError("Unsafe file path found inside model zip.")
    zip_ref.extractall(extract_dir)


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
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)

            download_result = gdown.download(id=file_id, output=zip_path, quiet=True)

            if download_result is None or not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
                st.error("Model download failed. Please check Google Drive file permissions.")
                st.stop()

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safe_extract_zip(zip_ref, extract_dir)

            found_path = None
            encoder_found_path = None
            for root, _, files in os.walk(extract_dir):
                if "config.json" in files and found_path is None:
                    found_path = root
                if "label_encoder.pkl" in files and encoder_found_path is None:
                    encoder_found_path = os.path.join(root, "label_encoder.pkl")

            if not found_path:
                st.error("Model extraction failed. config.json was not found in the model zip.")
                st.stop()

            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            shutil.copytree(found_path, model_dir)

            # Some zips keep label_encoder.pkl outside the actual Transformers model folder.
            # Copy it into the model folder so classification labels do not break after extraction cleanup.
            model_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            if encoder_found_path and not os.path.exists(model_encoder_path):
                shutil.copy2(encoder_found_path, model_encoder_path)

            shutil.rmtree(extract_dir, ignore_errors=True)
            if os.path.exists(zip_path):
                os.remove(zip_path)
    except zipfile.BadZipFile:
        st.error("Downloaded model file is not a valid zip. Please check the Google Drive model file.")
        st.stop()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    return model_dir


def find_label_encoder(model_path):
    possible_paths = [
        "label_encoder.pkl",
        os.path.join(model_path, "label_encoder.pkl"),
        os.path.join("distilbert_resume_model", "label_encoder.pkl"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_models():
    model_path = ensure_model_exists()

    encoder_path = find_label_encoder(model_path)
    if encoder_path is None:
        st.error(
            "label_encoder.pkl missing. Put label_encoder.pkl beside app.py or inside the model folder."
        )
        st.stop()

    try:
        label_encoder = joblib.load(encoder_path)
    except Exception as e:
        st.error(f"Could not load label_encoder.pkl: {e}")
        st.stop()

    try:
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
        )
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    return label_encoder, classifier


# Models are loaded only after the user clicks Analyze.
# This keeps the frontend fast and avoids startup crashes on Streamlit Cloud.


# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"[^\w\s@.+-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def normalize_key(value):
    value = str(value or "").strip().lower()
    if value in {"", "not found", "unknown", "unknown candidate", "unreadable resume"}:
        return ""
    return re.sub(r"\s+", "", value)


def extract_email(text):
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text or "")
    return emails[0].lower() if emails else "Not Found"


def extract_phone(text):
    raw = text or ""
    patterns = [
        r"(?:\+91[-\s]?)?[6-9]\d{9}",
        r"(?:\+91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}",
        r"(?:\+91[-\s]?)?[6-9]\d{2}[-\s]?\d{3}[-\s]?\d{4}",
    ]
    for pattern in patterns:
        phones = re.findall(pattern, raw)
        for phone in phones:
            digits = re.sub(r"\D", "", phone)
            if digits.startswith("91") and len(digits) == 12:
                digits = digits[2:]
            if len(digits) == 10 and digits[0] in "6789":
                return f"+91 {digits}"
    return "Not Found"


def extract_name(text):
    lines = [line.strip() for line in (text or "").split("\n") if line.strip()]
    blacklist = [
        "resume", "curriculum", "vitae", "developer", "engineer", "email",
        "phone", "mobile", "contact", "linkedin", "github", "address",
        "objective", "profile", "summary", "skills", "experience", "education",
        "certification", "project", "portfolio", "dob", "gender",
    ]

    for line in lines[:20]:
        name_match = re.search(r"(?:candidate\s*)?name\s*[:\-]\s*([A-Za-z][A-Za-z .]{1,45})", line, re.IGNORECASE)
        if name_match:
            return re.sub(r"\s+", " ", name_match.group(1)).strip().title()

    for line in lines[:15]:
        lower = line.lower()
        if any(word in lower for word in blacklist):
            continue
        if re.search(r"[@:/\\]|\d", line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and len(line) <= 45:
            return line.title()
    return "Unknown Candidate"

def extract_experience(text):
    text = (text or "").lower()

    if re.search(r"\bfresher\b|entry[-\s]?level|no prior experience", text):
        return "Fresher"

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|year|yrs|yr)\s*(?:of)?\s*(?:experience|exp)?",
        r"experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|year|yrs|yr)",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    if matches:
        years = max(float(x) for x in matches)
        if years <= 0:
            return "Fresher"
        return f"{years:.1f} Years"

    month_matches = re.findall(r"(\d+)\+?\s*(?:months|month|mos|mo)\s*(?:of)?\s*(?:experience|exp|internship)?", text)
    if month_matches:
        months = max(int(x) for x in month_matches)
        if months >= 12:
            return f"{months / 12:.1f} Years"
        return f"{months} Months"

    return "Fresher"

def calculate_match_score(jd, resume):
    jd = jd or ""
    resume = resume or ""
    if not jd.strip() or not resume.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([jd, resume])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(max(0.0, min(float(similarity * 100), 100.0)), 2)
    except ValueError:
        return 0.0
    except Exception:
        return 0.0


def decode_prediction_label(label, label_encoder):
    label_text = str(label or "").strip()
    if not label_text:
        return "Unknown"

    classes = getattr(label_encoder, "classes_", None)
    if classes is not None and label_text in set(map(str, classes)):
        return label_text

    # Handles LABEL_0, LABEL-0, label_0, or plain numeric labels.
    match = re.search(r"(\d+)$", label_text)
    if match:
        label_id = int(match.group(1))
        try:
            if classes is not None and 0 <= label_id < len(classes):
                return str(label_encoder.inverse_transform([label_id])[0])
        except Exception:
            return "Unknown"

    return label_text


def make_input_signature(jd_text, files):
    file_signature = tuple(
        (
            getattr(file, "name", ""),
            getattr(file, "size", None),
            getattr(file, "type", ""),
        )
        for file in (files or [])
    )
    return (clean_text(jd_text), file_signature)


def read_pdf(file):
    try:
        file.seek(0)
        pdf_bytes = file.read()
        if not pdf_bytes:
            return None

        text_parts = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                try:
                    pdf.authenticate("")
                except Exception:
                    return None

            for page in pdf:
                text_parts.append(page.get_text("text") or "")

        return "\n".join(text_parts).strip()
    except Exception:
        return None


def analyze_resume(file, jd_text, label_encoder, classifier):
    raw_text = read_pdf(file)
    if not raw_text or len(raw_text) < 30:
        return {
            "File": file.name,
            "Name": "Unreadable Resume",
            "Email": "Not Found",
            "Phone": "Not Found",
            "Experience": "Unknown",
            "Domain": "Invalid PDF",
            "Score": 0.0,
            "Duplicate Status": "Not Checked",
            "Preview": "Could not extract text from this PDF.",
        }

    cleaned = clean_text(raw_text)

    try:
        prediction = classifier(cleaned[:4000], truncation=True, max_length=512)
        if isinstance(prediction, list) and prediction and isinstance(prediction[0], list):
            prediction = prediction[0]

        if not isinstance(prediction, list) or not prediction:
            category = "Unknown"
        else:
            label = prediction[0].get("label", "Unknown")
            category = decode_prediction_label(label, label_encoder)
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
        "Duplicate Status": "Unique",
        "Preview": raw_text[:2500],
    }


def add_duplicate_status(df):
    if df.empty:
        return df

    df = df.copy()
    df["Email Key"] = df["Email"].apply(normalize_key)
    df["Phone Key"] = df["Phone"].apply(normalize_key)

    valid_email = df["Email Key"] != ""
    valid_phone = df["Phone Key"] != ""

    duplicate_by_email = valid_email & df.duplicated(subset=["Email Key"], keep=False)
    duplicate_by_phone = valid_phone & df.duplicated(subset=["Phone Key"], keep=False)

    df["Duplicate Status"] = "Unique"
    df.loc[duplicate_by_email & duplicate_by_phone, "Duplicate Status"] = "Duplicate Email and Phone"
    df.loc[duplicate_by_email & ~duplicate_by_phone, "Duplicate Status"] = "Duplicate Email"
    df.loc[~duplicate_by_email & duplicate_by_phone, "Duplicate Status"] = "Duplicate Phone"

    return df.drop(columns=["Email Key", "Phone Key"], errors="ignore")


def get_duplicate_df(df):
    if df.empty or "Duplicate Status" not in df.columns:
        return pd.DataFrame()
    return df[df["Duplicate Status"].str.contains("Duplicate", case=False, na=False)]


def render_cards(dataframe, show_duplicate_status=False):
    if dataframe.empty:
        st.info("No candidates found")
        return

    for _, row in dataframe.iterrows():
        try:
            score = float(row.get("Score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        duplicate_status = row.get("Duplicate Status", "Unique")
        emoji = "⚠️" if "Duplicate" in str(duplicate_status) else ("🏆" if score >= 75 else "👨‍💻")

        st.markdown('<div class="resume-box">', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"## {emoji} {row.get('Name', 'Unknown Candidate')}")
            st.markdown(f"**Domain:** {row.get('Domain', 'Unknown')}")
            st.markdown(f"**Experience:** {row.get('Experience', 'Unknown')}")
            st.markdown(f"**Email:** {row.get('Email', 'Not Found')}")
            st.markdown(f"**Phone:** {row.get('Phone', 'Not Found')}")
            st.markdown(f"**Resume:** {row.get('File', 'Unknown File')}")
            if show_duplicate_status:
                st.warning(f"Duplicate Status: {duplicate_status}")
        with c2:
            st.metric("Match Score", f"{score}%")
            st.progress(max(0.0, min(score / 100, 1.0)))

        with st.expander("Preview Resume"):
            st.text(row.get("Preview", "No preview available"))
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="main-title">AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Advanced AI Candidate Screening & Resume Ranking</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚡ Control Panel")
    st.button("🌓 Toggle Theme", on_click=toggle_theme, use_container_width=True)
    st.markdown("---")

    jd_input = st.text_area("📄 Job Description", height=220)
    uploaded_files = st.file_uploader(
        "📂 Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} Resume(s) Uploaded")

    st.markdown("---")
    analyze_button = st.button("🚀 Analyze Resumes", use_container_width=True)

current_input_signature = make_input_signature(jd_input, uploaded_files)
if (
    st.session_state.processed
    and st.session_state.last_input_signature is not None
    and current_input_signature != st.session_state.last_input_signature
    and not analyze_button
):
    st.session_state.processed = False
    st.session_state.results = pd.DataFrame()

if analyze_button:
    if not jd_input.strip():
        st.warning("Please enter job description")
        st.session_state.processed = False
        st.session_state.results = pd.DataFrame()
    elif not uploaded_files:
        st.warning("Please upload resumes")
        st.session_state.processed = False
        st.session_state.results = pd.DataFrame()
    else:
        label_encoder, classifier = load_models()
        progress = st.progress(0)
        results = []

        for index, file in enumerate(uploaded_files):
            result = analyze_resume(file, jd_input, label_encoder, classifier)
            results.append(result)
            progress.progress((index + 1) / len(uploaded_files))

        progress.empty()
        df = pd.DataFrame(results)
        df = add_duplicate_status(df)
        df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.session_state.results = df
        st.session_state.processed = True
        st.session_state.last_input_signature = current_input_signature

if st.session_state.processed and not st.session_state.results.empty:
    df = st.session_state.results.copy()
    duplicate_df = get_duplicate_df(df)

    top_score = int(float(df["Score"].max())) if not df.empty else 0
    avg_score = round(float(df["Score"].mean()), 1) if not df.empty else 0
    duplicate_count = len(duplicate_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resumes", len(df))
    col2.metric("Top Match", f"{top_score}%")
    col3.metric("Average Score", f"{avg_score}%")
    col4.metric("Duplicates", duplicate_count)

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs(["All Candidates", "Top Matches", "Experienced Pros", "Freshers", "Duplicates"])

    with tabs[0]:
        render_cards(df)

    with tabs[1]:
        render_cards(df[df["Score"] >= 50])

    with tabs[2]:
        experienced_df = df[~df["Experience"].astype(str).str.contains("Fresher|Unknown", case=False, na=False)]
        render_cards(experienced_df)

    with tabs[3]:
        fresher_df = df[df["Experience"].astype(str).str.contains("Fresher", case=False, na=False)]
        render_cards(fresher_df)

    with tabs[4]:
        st.markdown("### ⚠️ Duplicate Applications")
        st.markdown(
            '<p class="small-muted">Duplicates are detected using repeated email addresses or phone numbers. "Not Found" values are ignored.</p>',
            unsafe_allow_html=True,
        )
        render_cards(duplicate_df, show_duplicate_status=True)

    export_df = df.drop(columns=["Preview"], errors="ignore")
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Report",
        data=csv,
        file_name="resume_analysis_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.markdown(
        """
        <div class="card">
            <h2>🚀 Features</h2>
            <ul style="line-height:2;">
                <li>AI Resume Classification</li>
                <li>Modern Dashboard UI with Dark/Light Theme</li>
                <li>JD Similarity Matching</li>
                <li>Resume Ranking</li>
                <li>Experienced and Fresher Filtering</li>
                <li>Duplicate Detection using Email and Phone</li>
                <li>PDF Resume Parsing</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
