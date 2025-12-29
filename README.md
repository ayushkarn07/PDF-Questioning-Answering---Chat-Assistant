# 📄 PDF Question Answering System (RAG + Groq)

A **PDF Question Answering web application** built using **Retrieval-Augmented Generation (RAG)**.  
Users can upload large PDFs (50–60+ pages) and ask natural language questions to get **accurate, context-based answers**.

The app is deployed using **Streamlit Cloud** and powered by **Groq LLM**, **FAISS**, and **HuggingFace embeddings**.

---
# URL
 pdfaayush.streamlit.app/

## 🚀 Features

- Upload PDF documents
- Semantic search using FAISS vector database
- Context-aware answers using Groq LLM (LLaMA 3.1)
- Supports large PDFs (research papers, reports, books)
- Chat-style question answering interface
- Secure API key handling using Streamlit Secrets
- Deployed on Streamlit Cloud

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (LLaMA 3.1 8B Instant)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS
- **Framework**: LangChain (modular packages)
- **Deployment**: Streamlit Cloud

---
1.  **Install Dependencies**:
    ```bash
    pip install requirements.txt
    ```
2. Environment Setup
    ```bash
    os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
    ```
3. Run the Application
   ```bash
    streamlit run app.py
---
# SnapShots
<img width="959" height="416" alt="image" src="https://github.com/user-attachments/assets/ca637600-0303-439d-af59-db64742a48a4" />
<img width="957" height="415" alt="image" src="https://github.com/user-attachments/assets/65e85500-0ff6-416f-9b22-bd8df14e1cf7" />




