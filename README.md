# 🤖 Web Knowledge AI Assistant

A modular, multi-source Retrieval-Augmented Generation (RAG) system built with **Streamlit, LangChain, FAISS, and Llama 3.1 (Groq)**.

This application allows users to ingest knowledge from multiple sources and interact with it using a conversational AI interface powered by vector similarity search.

---

## 🚀 Live Demo

🎥 Demo Video:  
👉 **YOUR_VIDEO_LINK_HERE**

---

## ✨ Features

### 📂 Multi-Source Ingestion
- 🌐 Web URLs
- 📄 PDF files
- 📑 TXT files
- 📊 CSV files
- 📝 Direct text input

### 🧠 Retrieval-Augmented Generation (RAG)
- FAISS vector database
- HuggingFace MiniLM embeddings
- Context-grounded AI responses
- Top-k similarity retrieval
- Source transparency with similarity scores

### 💬 Conversational Interface
- Chat-style UI
- Memory persistence within session
- Reset memory functionality
- Streaming-style answer display

### 📊 Analytics Dashboard
- Total documents processed
- Total chunks created
- Average chunk length
- Model configuration display

---

## 🏗 Project Architecture

```
EquityResearchTool/
│
├── core/
│   ├── chain.py
│   ├── embeddings.py
│   ├── loader.py
│   ├── memory.py
│   ├── splitter.py
│   └── vectorstore.py
│
├── data/
│   ├── faiss_index/
│   ├── vector_index.pkl
│   └── vectorstore.pkl
│
├── main.py
├── requirements.txt
├── style.css
├── .env
└── FAISS.ipynb
```

---

## ⚙️ System Workflow

1. Load documents from selected sources
2. Split into semantic chunks
3. Generate embeddings using MiniLM
4. Store embeddings in FAISS
5. Retrieve top-k similar chunks
6. Generate response using Llama 3.1 via Groq
7. Display answer with source references

---

## 🛠 Tech Stack

- Python 3.10
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq (Llama 3.1 8B)
- Sentence Transformers
- Python-dotenv

---

## 🧩 Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/web-rag-ai-assistant.git
cd web-rag-ai-assistant
```

---

### 2️⃣ Create Environment

```bash
conda create -n web_rag python=3.10 -y
conda activate web_rag
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 5️⃣ Run Application

```bash
streamlit run main.py
```

Open browser:

```
http://localhost:8501
```

---

## 📊 Example Usage

1. Add URL / Upload documents / Paste text
2. Click "Process Information"
3. Switch to Chat tab
4. Ask a question
5. View answer + similarity scores + sources

---

## 🔒 Security Note

- FAISS index files are ignored via `.gitignore`
- `.env` file is not tracked
- No unsafe pickle deserialization enabled
- Designed for local trusted execution

---

## 🎯 Why This Project?

Unlike basic RAG demos, this system:

- Uses modular architecture
- Separates ingestion, embeddings, vectorstore, and chain logic
- Supports multi-modal knowledge ingestion
- Includes session-based memory
- Displays similarity scores for transparency
- Structured for future FastAPI + Docker deployment

---

## 🚀 Future Improvements

- Token-level streaming output
- Light/Dark theme toggle
- Retrieval confidence scoring
- Dockerized deployment
- FastAPI backend integration
- Evaluation framework for hallucination detection
- Multi-user session support

---

## 👨‍💻 Author

**Jay Shimpi**  
AI & Data Science Student 

---

## ⭐ If You Found This Useful

Star the repository and share feedback!

