import streamlit as st
from st_audiorec import st_audiorec
import shutil
import os
import time
import requests
import docx
import pytesseract
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import base64

# ─── Poppler Path ────────────────────────────────────────────────────────────
if os.name == "nt":
    poppler_path = r"C:\Users\sailo\OneDrive\Desktop\pdf-chat-main\poppler-24.08.0\Library\bin"
else:
    poppler_path = "/usr/bin"

# ─── Environment Setup ───────────────────────────────────────────────────────
load_dotenv()
if "TESSERACT_PATH" in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_PATH"]
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if "question_answer_history" not in st.session_state:
    st.session_state.question_answer_history = []
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "sources_loaded" not in st.session_state:
    st.session_state.sources_loaded = False

# ─── Language Map ─────────────────────────────────────────────────────────────
language_map = {
    "Telugu": "te", "English": "en", "Hindi": "hi",
    "Tamil": "ta", "Spanish": "es", "French": "fr"
}

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
/* Removed 'header' so the sidebar button comes back */
#MainMenu, footer { visibility: hidden; }

/* Make the header transparent so it doesn't mess up your dark theme */
[data-testid="stHeader"] { background-color: transparent !important; }
/* Top Bar */
.top-bar {
    background: linear-gradient(90deg, #1a3a5c, #0d47a1, #1565c0);
    padding: 18px 32px 14px;
    border-radius: 0 0 16px 16px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.top-bar h1 { font-size:1.7rem; font-weight:700; color:#ffffff !important; margin:0; letter-spacing:-0.3px; }
.top-bar p  { font-size:0.85rem; color:#90caf9 !important; margin:2px 0 0; }

/* Badges */
.badge-on  { background:#0d3321; color:#3fb950; border:1px solid #238636; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; display:inline-block; }
.badge-off { background:#1c1c1c; color:#8b949e; border:1px solid #30363d; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; display:inline-block; }

/* Headings & Cards */
.sb-heading { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#58a6ff !important; margin:18px 0 10px; padding-bottom:6px; border-bottom:1px solid #21262d; }
.chat-card { background:#161b22; border:1px solid #21262d; border-radius:12px; padding:0; margin-bottom:12px; overflow:hidden; }
.chat-q { background:#1c2333; padding:14px 18px; font-size:14px; color:#79c0ff; font-weight:500; border-bottom:1px solid #21262d; }
.chat-a { padding:14px 18px; font-size:14px; color:#e6edf3; line-height:1.7; }
.chat-q-label { font-size:11px; color:#58a6ff; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
.chat-a-label { font-size:11px; color:#3fb950; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
.hero { text-align:center; padding:60px 20px 50px; color:#8b949e; }
.hero .icon { font-size:56px; margin-bottom:16px; }
.hero h2 { font-size:1.4rem; color:#58a6ff !important; margin-bottom:8px; }
.hero p  { font-size:0.9rem; color:#6e7681 !important; }
.hero-pill { display:inline-block; background:#1c2333; border:1px solid #30363d; border-radius:20px; padding:6px 14px; font-size:12px; color:#8b949e !important; margin:4px; }
.input-card { background:#161b22; border:1px solid #21262d; border-radius:14px; padding:20px 24px; margin-bottom:20px; }

/* --- FIXED INPUT BOX VISIBILITY --- */
[data-testid="stTextInput"] input {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important; /* White text when typing */
    padding: 10px 14px !important;
    font-size: 14px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #8b949e !important; /* Visible light grey for placeholder */
    opacity: 1 !important;
}

/* Selectbox */
.stSelectbox > div > div { background:#0d1117 !important; border:1px solid #30363d !important; border-radius:8px !important; color:#e6edf3 !important; }

/* --- FIXED UPLOAD SECTION --- */
[data-testid="stFileUploadDropzone"] {
    background-color: #0d1117 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: #8b949e !important; /* Fix the dark text inside the box */
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: #161b22 !important;
    border-color: #58a6ff !important;
}

/* Buttons & Misc */
.stButton > button { background:linear-gradient(135deg,#1565c0,#0d47a1) !important; color:#ffffff !important; border:none !important; border-radius:8px !important; padding:10px 22px !important; font-size:14px !important; font-weight:600 !important; width:100%; transition:opacity 0.2s; }
.stButton > button:hover { opacity:0.88 !important; }
.stRadio label, .stCheckbox label { color:#c9d1d9 !important; font-size:14px !important; }
.stNumberInput > div > div > input { background:#0d1117 !important; border:1px solid #30363d !important; border-radius:8px !important; color:#e6edf3 !important; }
div[data-testid="stMarkdownContainer"] p { color:#c9d1d9; }
.stSuccess { background:#0d3321 !important; border:1px solid #238636 !important; border-radius:8px !important; color:#3fb950 !important; }
.stError   { background:#2d0f0f !important; border:1px solid #f85149 !important; border-radius:8px !important; color:#f85149 !important; }
.stInfo    { background:#0c2a43 !important; border:1px solid #1f6feb !important; border-radius:8px !important; color:#79c0ff !important; }
hr { border-color:#21262d !important; margin:20px 0 !important; }
.section-title { font-size:15px; font-weight:600; color:#e6edf3 !important; margin:20px 0 12px; display:flex; align-items:center; gap:8px; }
.footer { position:fixed; bottom:0; left:0; width:18rem; background:#0d1117; border-top:1px solid #21262d; padding:10px 16px; font-size:11px; color:#6e7681 !important; text-align:center; }
.footer a { color:#58a6ff !important; text-decoration:none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  BACKEND FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def text_to_speech(text, language):
    try:
        lang_code = language_map.get(language, "en")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        audio_base64 = base64.b64encode(audio_file.read()).decode()
        st.markdown(f'''
            <audio controls autoplay style="width:100%;border-radius:8px;margin-top:8px;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Text-to-speech failed: {e}")


def extract_text_from_image(image):
    if shutil.which("tesseract") is None:
        st.error("Tesseract not found. Please install it.")
        return ""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            pdf.seek(0)
            pdf_reader = PdfReader(BytesIO(pdf_bytes))
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text
                else:
                    try:
                        images = convert_from_bytes(
                            pdf_bytes, first_page=page_num+1,
                            last_page=page_num+1, poppler_path=poppler_path)
                        for image in images:
                            text += extract_text_from_image(image)
                    except Exception as e:
                        st.error(f"OCR failed on page {page_num+1}: {e}")
        except Exception as e:
            st.error(f"PDF processing failed: {e}")
    return text


def get_docx_text(docx_docs):
    text = ""
    for docx_file in docx_docs:
        try:
            document = docx.Document(docx_file)
            for para in document.paragraphs:
                text += para.text + "\n"
            for rel in document.part._rels:
                rel_part = document.part._rels[rel]
                if "image" in rel_part.target_ref:
                    try:
                        image = Image.open(BytesIO(rel_part.target_part.blob))
                        text += "\n" + extract_text_from_image(image)
                    except Exception as e:
                        st.error(f"OCR failed for Word image: {e}")
        except Exception as e:
            st.error(f"Word doc processing failed: {e}")
    return text


def get_url_text_selenium(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(5)
        html_content = driver.page_source
        driver.quit()
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.error(f"Selenium failed for {url}: {e}")
        return ""


def get_url_text(urls):
    text = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:104.0) Gecko/20100101 Firefox/104.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/"
    }
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                text += get_url_text_selenium(url)
                continue
            if url.lower().endswith(".pdf"):
                pdf_bytes = response.content
                pdf_reader = PdfReader(BytesIO(pdf_bytes))
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text
                    else:
                        try:
                            images = convert_from_bytes(
                                pdf_bytes, first_page=page_num+1,
                                last_page=page_num+1, poppler_path=poppler_path)
                            for image in images:
                                text += extract_text_from_image(image)
                        except Exception as e:
                            st.error(f"OCR failed on URL PDF page {page_num+1}: {e}")
            elif url.lower().endswith((".jpg", ".jpeg", ".png")):
                image = Image.open(BytesIO(response.content))
                text += extract_text_from_image(image)
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                text += soup.get_text(separator=" ", strip=True)
        except Exception as e:
            st.error(f"Failed to process {url}: {e}")
    return text


def get_text_chunks(text):
    # RATE LIMIT FIX: Reduced chunk size from 50000 to 10000 to prevent token limits
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    # FIXED: updated to current gemini embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are provided with the following context and a question. Please craft a thorough,
    well-organized answer that covers every relevant detail. Your response should be at
    least 300 words long and include tables, bullet points, or other structured formats
    as needed to enhance clarity. If the provided context does not cover some aspects of
    the question, supplement your answer with credible information from Google.

    Your answer must:
    - Explain the topic comprehensively, covering all pertinent aspects.
    - Include tables or structured data to summarize key information where applicable.
    - Use clear headings, subheadings, bullet points, or numbered lists.
    - Elaborate with in-depth details, examples, and analysis.
    - Ensure the explanation is logically structured and easy to follow.

    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def search_google(query, retries=5):
    model = genai.GenerativeModel('gemini-2.0-flash')
    for attempt in range(retries):
        try:
            return model.generate_content(f"Give me a summary of: {query}").text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep((2 ** attempt) + 1)
            else:
                st.error(f"Search error: {e}")
                return "Could not find information."
    return "Could not find information."


def translate_text(text, target_language):
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model.generate_content(
        f"Translate the following text to {target_language}:\n\n{text}"
    ).text


def user_input(user_question, target_language):
    # FIXED: updated to current gemini embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    try:
        new_db = FAISS.load_local(
            os.path.join(os.getcwd(), "faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    if (
        "cannot be answered from the given context" in response["output_text"].lower()
        or "not found in the provided context" in response["output_text"].lower()
    ):
        google_answer = search_google(user_question)
        response["output_text"] = (
            "This question cannot be answered from the given context. "
            "Here is an answer sourced from Google:\n\n" + google_answer
        )

    # RATE LIMIT FIX: Add a 2-second delay before making the translation API call
    time.sleep(2) 
    translated_answer = translate_text(response["output_text"], target_language)
    
    st.session_state.question_answer_history.append({
        "question": user_question,
        "answer": translated_answer
    })
    return translated_answer


# ─────────────────────────────────────────────────────────────────────────────
#  LANGUAGE LIST
# ─────────────────────────────────────────────────────────────────────────────
all_languages = (
    "English", "Telugu", "Hindi", "Tamil", "Spanish", "French", "German",
    "Chinese", "Gujarati", "Marathi", "Punjabi", "Kannada", "Malayalam",
    "Odia", "Assamese", "Urdu", "Konkani", "Maithili", "Sindhi", "Nepali",
    "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Arabic",
    "Dutch", "Greek", "Swedish", "Turkish", "Vietnamese", "Polish", "Bengali",
    "Afrikaans", "Albanian", "Armenian", "Azerbaijani", "Basque", "Belarusian",
    "Bosnian", "Bulgarian", "Catalan", "Croatian", "Czech", "Danish", "Estonian",
    "Finnish", "Georgian", "Hebrew", "Hungarian", "Icelandic", "Indonesian",
    "Irish", "Latvian", "Lithuanian", "Macedonian", "Malagasy", "Maltese",
    "Mongolian", "Montenegrin", "Norwegian", "Pashto", "Persian", "Romanian",
    "Serbian", "Slovak", "Slovenian", "Somali", "Swahili", "Tajik", "Tatar",
    "Thai", "Tibetan", "Turkmen", "Ukrainian", "Uzbek", "Welsh", "Yoruba",
    "Zulu", "Amharic", "Chechen", "Esperanto", "Fijian", "Galician", "Hausa",
    "Igbo", "Javanese", "Kinyarwanda", "Kurdish", "Lao", "Luxembourgish",
    "Oromo", "Quechua", "Samoan", "Shona", "Sinhala", "Tagalog", "Tigrinya",
    "Wolof", "Xhosa", "Zaza"
)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Top Header Bar ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="top-bar">
        <div style="font-size:32px;">🤖</div>
        <div>
            <h1>Personal AI Research Assistant</h1>
            <p>Chat with PDFs, Word docs &amp; URLs · Multilingual · by Eppa Sai Vardhan Reddy </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:

        st.markdown("<div style='padding:4px 0 16px'>", unsafe_allow_html=True)
        if st.session_state.sources_loaded:
            st.markdown('<span class="badge-on">● Sources loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-off">○ No sources loaded</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Documents
        st.markdown('<div class="sb-heading">📁 Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF or Word files",
            accept_multiple_files=True,
            type=["pdf", "docx"],
            label_visibility="collapsed"
        )
        if st.button("⚡ Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_text = ""
                    pdf_files  = [f for f in uploaded_files if f.type == "application/pdf"]
                    docx_files = [f for f in uploaded_files if "wordprocessingml" in f.type]
                    if pdf_files:
                        raw_text += get_pdf_text(pdf_files)
                    if docx_files:
                        raw_text += get_docx_text(docx_files)
                    if raw_text.strip():
                        get_vector_store(get_text_chunks(raw_text))
                        st.session_state.sources_loaded = True
                        st.success(f"✓ {len(uploaded_files)} file(s) processed successfully")
                    else:
                        st.error("No text could be extracted from the uploaded files.")
            else:
                st.error("Please upload at least one file.")

        # URLs
        st.markdown('<div class="sb-heading">🌐 Web URLs</div>', unsafe_allow_html=True)
        num_urls = st.number_input(
            "Number of URLs", min_value=1, max_value=10, value=1, step=1
        )
        url_list = []
        for i in range(int(num_urls)):
            url = st.text_input(
                f"URL {i+1}",
                placeholder="https://...",
                label_visibility="collapsed",
                key=f"url_{i}"
            )
            url_list.append(url)
        if st.button("⚡ Process URLs"):
            urls = [u.strip() for u in url_list if u.strip()]
            if urls:
                with st.spinner("Scraping and processing URLs..."):
                    raw_text = get_url_text(urls)
                    if raw_text.strip():
                        get_vector_store(get_text_chunks(raw_text))
                        st.session_state.sources_loaded = True
                        st.success(f"✓ {len(urls)} URL(s) processed successfully")
                    else:
                        st.error("No text could be extracted from the provided URLs.")
            else:
                st.error("Please enter at least one valid URL.")

        # Settings
        st.markdown('<div class="sb-heading">⚙️ Settings</div>', unsafe_allow_html=True)
        audio_output = st.checkbox("🔊 Enable audio output")
        if st.button("🗑️ Clear conversation"):
            st.session_state.question_answer_history = []
            st.rerun()

        st.markdown("""
        <div class="footer">
            <a href="https://www.linkedin.com/in/eppa-sai-vardhan-reddy-5b71213a4" target="_blank">Connect with the author</a><br>
            © Eppa Sai Vardhan Reddy
        </div>
        """, unsafe_allow_html=True)

    # ── Main Content ─────────────────────────────────────────────────────────
    col_main, _ = st.columns([1, 0.001])
    with col_main:

        # Input card
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            input_mode = st.radio("Mode", ["✏️ Type", "🎙️ Speak"], label_visibility="collapsed")
        with c2:
            language = st.selectbox("Response language", all_languages, label_visibility="collapsed")

        if "Speak" in input_mode:
            st.info("Record your question — it will be auto-transcribed.")
            audio_bytes = st_audiorec()
            if audio_bytes is not None:
                try:
                    r = sr.Recognizer()
                    with sr.AudioFile(BytesIO(audio_bytes)) as source:
                        audio_data = r.record(source)
                    transcribed = r.recognize_google(
                        audio_data, language=language_map.get(language, "en-IN"))
                    st.session_state.transcribed_text = transcribed
                    st.success("Transcription complete.")
                except Exception as e:
                    st.error(f"Transcription error: {e}")
            user_question = st.text_input(
                "Recorded question (editable)",
                value=st.session_state.get("transcribed_text", ""),
                placeholder="Your recorded question will appear here...",
                label_visibility="collapsed"
            )
        else:
            user_question = st.text_input(
                "Ask a question",
                placeholder="Ask anything about your loaded documents or URLs...",
                label_visibility="collapsed"
            )

        submitted = st.button("🚀 Ask Assistant")
        st.markdown("</div>", unsafe_allow_html=True)

        # Handle submission
        if submitted:
            if not user_question.strip():
                st.error("Please enter or record a question first.")
            elif not st.session_state.sources_loaded:
                st.error("Please load at least one document or URL using the sidebar first.")
            else:
                with st.spinner("Thinking..."):
                    answer = user_input(user_question, language)
                if audio_output and answer:
                    text_to_speech(answer, language)
                st.rerun()

        # Conversation history
        if not st.session_state.question_answer_history:
            st.markdown("""
            <div class="hero">
                <div class="icon">💬</div>
                <h2>Ready to answer your questions</h2>
                <p>Load a document or URL using the sidebar, then ask anything below.</p>
                <div style="margin-top:20px">
                    <span class="hero-pill">📄 PDF support</span>
                    <span class="hero-pill">📝 Word docs</span>
                    <span class="hero-pill">🌐 Web URLs</span>
                    <span class="hero-pill">🎙️ Voice input</span>
                    <span class="hero-pill">🔊 Audio output</span>
                    <span class="hero-pill">🌍 100+ languages</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="section-title">
                <span>💬</span> Conversation
                <span style="font-size:12px;color:#6e7681;font-weight:400;margin-left:auto">
                    {len(st.session_state.question_answer_history)} exchange(s)
                </span>
            </div>
            """, unsafe_allow_html=True)

            for pair in reversed(st.session_state.question_answer_history):
                answer_html = pair['answer'].replace('\n', '<br>')
                st.markdown(f"""
                <div class="chat-card">
                    <div class="chat-q">
                        <div class="chat-q-label">You asked</div>
                        {pair['question']}
                    </div>
                    <div class="chat-a">
                        <div class="chat-a-label">Assistant</div>
                        {answer_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
