import os
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import chromadb
import tiktoken
from dotenv import load_dotenv
from google import genai

# --------------------------- Load Gemini API Key ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Add it in .env file.")
    st.stop()

# Initialize Gemini Client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")
    st.stop()

LLM_MODEL = "gemini-2.5-flash"

# --------------------------- ChromaDB Setup ---------------------------
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
except:
    chroma_client = chromadb.EphemeralClient()

collection = chroma_client.get_or_create_collection(
    name="web_qa_chunks",
    metadata={"hnsw:space": "cosine"}
)

# --------------------------- Helper Functions ---------------------------
def scrape_requests(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        content_tags = soup.find_all(["p", "h1", "h2", "h3"])
        return "\n\n".join([tag.get_text(strip=True) for tag in content_tags if tag.get_text(strip=True)])
    except Exception as e:
        return f"__SCRAPE_ERROR__:{str(e)}"

def scrape_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    try:
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        content_tags = soup.find_all(["p", "h1", "h2", "h3"])
        return "\n\n".join([tag.get_text(strip=True) for tag in content_tags if tag.get_text(strip=True)])
    finally:
        driver.quit()

def chunk_text(text: str, max_tokens=800, overlap=80) -> List[str]:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunks.append(enc.decode(tokens[start:end]))
            start = end - overlap if end < len(tokens) else end
        return chunks
    except:
        words = text.split()
        chunks, chunk = [], []
        for w in words:
            chunk.append(w)
            if len(chunk) >= 300:
                chunks.append(" ".join(chunk))
                chunk = chunk[-50:]
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

def get_embedding(text: str):
    TARGET_DIM = 512
    vec = [float(ord(c) % 256) for c in text[:TARGET_DIM]]
    if len(vec) < TARGET_DIM:
        vec += [0.0] * (TARGET_DIM - len(vec))
    return vec

def add_chunks_to_chroma(url, chunks):
    ids = [f"{url}__chunk__{i}" for i in range(len(chunks))]
    metadatas = [{"url": url, "chunk_index": i} for i in range(len(chunks))]
    embeddings = [get_embedding(c) for c in chunks]

    try:
        collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    except Exception as e:
        st.warning(f"Chroma error: {e}")

def query_chroma(question, url, top_k=4):
    q_emb = get_embedding(question)
    try:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents","metadatas"],
            where={"url": url}  # Only current URL
        )
        return list(zip(results["documents"][0], results["metadatas"][0]))
    except Exception as e:
        st.warning(f"Chroma query error: {e}")
        return []

def build_context(results):
    return "\n\n---\n\n".join(
        [f"Source: {meta.get('url')}\n{doc}" for doc, meta in results]
    )

def ask_gemini(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly."
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[prompt],
            config={"temperature": 0}
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="Web Q&A Bot", layout="wide")

# Custom colors & styles
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f0f9ff, #e0f7fa);
        color: #0c4b8e;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #ef4444;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        height: 40px;
        font-size: 16px;
        padding-left: 10px;
    }
    .stCheckbox>div>div>input {
        width: 20px;
        height: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Web Q&A Bot")
st.markdown("Ask questions about **any website** and get answers powered by **Gemini AI**!")

with st.container():
    st.subheader("Step 1: Enter website URL & your question")
    col1, col2 = st.columns([2,3])
    with col1:
        url = st.text_input("Website URL", placeholder="https://example.com")
    with col2:
        question = st.text_input("Your Question", placeholder="What is this website about?")

    use_selenium = st.checkbox("Use Selenium if website uses JavaScript")

    st.markdown("---")
    scrape_button = st.button("🚀 Scrape & Ask")

if scrape_button:
    if not url or not question:
        st.error("Please enter both URL and question.")
        st.stop()

    with st.spinner("🔍 Scraping website..."):
        raw = scrape_requests(url)

        if raw.startswith("__SCRAPE_ERROR__") and use_selenium:
            st.warning("Requests failed. Trying Selenium…")
            raw = scrape_selenium(url)

        if raw.startswith("__SCRAPE_ERROR__"):
            st.error(f"Failed to scrape website: {raw.replace('__SCRAPE_ERROR__:', '')}")
            st.stop()

    st.success("✅ Website scraped successfully!")

    # Chunk and store in Chroma (current URL only)
    chunks = chunk_text(raw)
    add_chunks_to_chroma(url, chunks)
    st.info(f"🗂 {len(chunks)} chunks added to ChromaDB for this URL.")

    # Query Chroma for context (current URL only)
    results = query_chroma(question, url)
    if not results:
        st.warning("⚠️ No relevant context found in this website.")
        context = ""
    else:
        context = build_context(results)

    # Ask Gemini AI
    with st.spinner("🤖 Asking Gemini AI..."):
        answer = ask_gemini(question, context)

    st.markdown("---")
    st.subheader("💡 Answer:")
    st.markdown(f"<div style='background-color:#e0f7fa; padding:15px; border-radius:10px;'>{answer}</div>", unsafe_allow_html=True)
