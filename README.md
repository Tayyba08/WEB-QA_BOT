# 🌐 Web Q&A Bot — Enhanced

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-✓-orange?logo=streamlit)](https://streamlit.io/)  
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-blue?logo=openai)](https://openai.com/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

**Short Description (350 characters):**  
An interactive Web Q&A Bot built with Python, Streamlit, OpenAI GPT, and ChromaDB. Scrape website content, chunk long text, store embeddings, and ask AI-powered questions. View answers, download them as .txt, and track history with an intuitive web interface.

---

The **Web Q&A Bot** is an interactive web application that allows users to scrape website content and ask AI-powered questions. Built with **Python**, **Streamlit**, **OpenAI GPT**, and **ChromaDB**, it integrates web scraping, vector embeddings, and natural language processing for fast and accurate responses.

## 🚀 Features

- Scrape content from **static pages** using BeautifulSoup and **dynamic pages** using Selenium.
- Automatically **chunks long text** to fit model input limits.
- Stores embeddings in **ChromaDB** for fast retrieval and caching.
- Queries **OpenAI GPT** to answer questions with context-aware responses.
- **Downloadable answers** as text files.
- **History tracking** of recent Q&A interactions.
- Intuitive **Streamlit interface** with sidebar options for example websites, chunk retrieval settings, and clearing history.

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/web-qa-bot.git
cd web-qa-bot

2. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # macOS/Linux


3. Install dependencies:

```bash
pip install -r requirements.txt


4. Add your OpenAI API key in .env:

```bash
OPENAI_API_KEY=your_api_key_here


Note: For Streamlit Cloud deployment, use .streamlit/secrets.toml instead of .env.

🖥️ Usage

Run the Streamlit app:
```bash
streamlit run app.py

Enter a website URL and your question.

Optionally enable Selenium for JS-rendered pages.

Click Scrape & Ask.

View the answer, download it as .txt, and track recent history.

📦 Requirements

Python 3.12+

streamlit

beautifulsoup4

requests

selenium

webdriver-manager

chromadb

openai==0.28.0

tiktoken

python-dotenv

🗂️ Folder Structure

web-qa-bot/
│
├─ app.py
├─ requirements.txt
├─ README.md
├─ .env (optional)
├─ venv/           # excluded from GitHub
├─ chroma_db/      # excluded from GitHub

🤝 Contributing

Fork the repository.

Create a new branch: git checkout -b feature-name

Make your changes and commit: git commit -m "Add feature"

Push to your branch: git push origin feature-name

Open a pull request.

📄 License

This project is licensed under the MIT License.

🙏 Acknowledgements

OpenAI — GPT models and embeddings

Streamlit — Web interface

ChromaDB — Vector database for embeddings