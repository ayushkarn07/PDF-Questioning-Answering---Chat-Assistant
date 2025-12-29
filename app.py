import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Question Answering System")
st.caption("Groq + RAG | Ask questions from long PDFs (50–60 pages supported)")

# ======================================
# SESSION STATE
# ======================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ======================================
# SIDEBAR - PDF UPLOAD
# ======================================
st.sidebar.header("📂 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

# ======================================
# PROCESS PDF
# ======================================
if uploaded_file:
    with st.spinner("Processing PDF... ⏳"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.sidebar.success("✅ PDF processed successfully!")

# ======================================
# GROQ LLM
# ======================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ======================================
# PROMPT TEMPLATE
# ======================================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant answering questions strictly based on the given context.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ======================================
# CHAT INPUT
# ======================================
question = st.chat_input("Ask a question from the PDF...")

if question and st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    st.session_state.chat_history.append({
        "question": question,
        "answer": response.content
    })

# ======================================
# DISPLAY CHAT
# ======================================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

if not uploaded_file:
    st.info("👈 Upload a PDF from the sidebar to start asking questions.")
