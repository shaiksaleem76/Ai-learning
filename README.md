# GenAI Learning & RAG Chatbot

A hands-on Generative AI learning project built using Python, Ollama, Streamlit, embeddings and ChromaDB.

This project explores how Large Language Models (LLMs) work and how to build Retrieval-Augmented Generation (RAG) applications using locally running AI models.

## 🚀 Features

- Local AI chatbot powered by Ollama.
- Interactive chatbot interface using Streamlit.
- Streaming AI responses.
- Conversation history management.
- Token usage tracking.
- Text embeddings using `nomic-embed-text`.
- Semantic search using cosine similarity.
- Vector storage and retrieval using ChromaDB.
- RAG-based question answering using retrieved documents.
- Document relevance filtering using distance thresholds.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Ollama | Run LLMs locally |
| Llama 3.2 1B | Chat model |
| Nomic Embed Text | Generate embeddings |
| Streamlit | Chatbot interface |
| ChromaDB | Vector database |
| Requests | API communication |

## 📁 Project Structure

```text
ai-learning/
│
├── app.py
├── rag_chatbot.py
├── ollama_practice.py
├── embedding_practice.py
├── semantic_search.py
├── vector_database.py
├── api_practice.py
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/shaiksaleem76/Ai-learning.git
cd Ai-learning
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama:

https://ollama.com/download

### 5. Download required models

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

### 6. Run the chatbot

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## 🧠 Learning Roadmap

- [x] AI and Generative AI fundamentals
- [x] Python fundamentals
- [x] Working with APIs and JSON
- [x] Running LLMs locally using Ollama
- [x] Building an interactive chatbot
- [x] Streaming AI responses
- [x] Understanding tokens and embeddings
- [x] Semantic search
- [x] Vector databases using ChromaDB
- [x] Building a basic RAG chatbot
- [ ] Document loading and processing
- [ ] Text chunking and optimization
- [ ] Advanced RAG techniques
- [ ] LangChain and Pydantic
- [ ] FastAPI integration
- [ ] AI agents and tool calling

## 🎯 Future Goal

Build a Smart AI Assistant capable of:

- Answering general questions.
- Understanding and searching PDF documents.
- Extracting structured information from documents.
- Integrating external APIs.
- Automating tasks using AI agents.

## 👨‍💻 Author

**Shaik Saleem**

Learning Generative AI, Python, LLMs and AI application development.

GitHub: https://github.com/shaiksaleem76
