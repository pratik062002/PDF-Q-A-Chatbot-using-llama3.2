# PDF Chatbot with Llama 3.2

A simple yet powerful PDF chatbot application that uses Ollama's Llama 3.2 model to answer questions about your documents. All processing happens locally on your PC with no cloud dependencies or API keys required.

## 🌟 Features

- **📄 PDF Upload** - Upload and process PDF documents easily
- **🤖 AI-Powered Responses** - Uses Llama 3.2 model for intelligent answers
- **🔍 Vector Search** - Retrieves relevant document sections using FAISS
- **💬 Chat History** - Maintains conversation history within the session
- **🔒 Privacy-Focused** - All data stays on your local machine
- **⚡ No API Keys Required** - Run completely offline
- **🚀 Simple & Lightweight** - Minimal dependencies and fast setup

## 📋 Prerequisites

- **Python 3.8+**
- **Ollama** installed and running ([Download from ollama.ai](https://ollama.ai))
- **Llama 3.2 model** downloaded via Ollama

## 🛠️ Installation & Setup

### Step 1: Install Ollama

Download and install from [https://ollama.ai](https://ollama.ai)

### Step 2: Download Llama 3.2 Model

```powershell
ollama pull llama3.2
```

This downloads approximately 2.0 GB of model files.

### Step 3: Install Python Dependencies

```powershell
cd e:\Chatbot
pip install -r requirements.txt
```

Or install manually:
```powershell
pip install streamlit langchain langchain-community PyPDF2 huggingface-hub sentence-transformers
```

### Step 4: Start Ollama Server

Make sure Ollama is running in the background:
```powershell
ollama serve
```

## 🚀 Running the Chatbot

```powershell
cd e:\Chatbot
streamlit run chatbot.py
```

The app will open at: `http://localhost:8501`

## 💻 How to Use

1. **Open the Application**
   - Navigate to `http://localhost:8501` in your browser

2. **Upload a PDF**
   - Click "Train Your File"
   - Select a PDF from your computer
   - Wait for processing confirmation

3. **Ask Questions**
   - Type your question in "Enter Your Prompt"
   - Click "Submit"
   - Get answers based on the document content

4. **View Chat History**
   - All conversations appear below
   - Clear history with the chat reset button

## 📁 Project Structure

```
e:\Chatbot\
├── chatbot.py                 # Main Streamlit application
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── db_utils.py               # Database utilities (optional)
├── app.py                    # Alternative OpenAI-powered app (optional)
└── Ollama model code.txt     # Reference code
```

## 📊 Code Overview

### Main Components

**1. Model Initialization**
```python
model = Ollama(model='llama3.2')
```
Initializes the Llama 3.2 model from Ollama

**2. PDF Processing**
```python
def pdf_to_chunks(pdf_file):
    # Extract text and split into chunks
```
Reads PDF and splits text into manageable chunks (1000 chars with 200 char overlap)

**3. Vector Store Creation**
```python
def create_vector_store(chunks):
    # Create FAISS embeddings
```
Creates embeddings using HuggingFace and stores in FAISS for similarity search

**4. Question Answering**
Uses RetrievalQA chain to:
- Search relevant document sections
- Pass to Llama 3.2 for answer generation
- Return contextual responses

## ⚙️ Configuration

### Model Selection

Current: **Llama 3.2** (optimized for local performance)

To use other models:
```powershell
ollama pull mistral
ollama pull llama2
ollama pull neural-chat
```

Then update `chatbot.py` line 10:
```python
model = Ollama(model='mistral')  # Change model name
```

### Chunk Size Adjustment

In `chatbot.py`, modify chunk parameters:
```python
text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # Adjust for larger/smaller chunks
    chunk_overlap=200     # Overlap for context
)
```

## 🔍 How It Works

### RAG (Retrieval Augmented Generation)

1. **PDF Upload** → Extract and split text into chunks
2. **Vectorization** → Convert chunks to embeddings using HuggingFace
3. **Storage** → Store vectors in FAISS database
4. **Query** → Convert user question to vector
5. **Retrieval** → Find 4 most relevant chunks from FAISS
6. **Generation** → Send chunks + question to Llama 3.2
7. **Response** → Return AI-generated answer

## 🔧 Troubleshooting

### Error: "Failed to initialize Ollama model"
**Solution:**
```powershell
# Check if Ollama is running
ollama serve

# Verify model is installed
ollama list

# Pull model if needed
ollama pull llama3.2
```

### Error: "Port 8501 already in use"
**Solution:**
```powershell
# Use different port
streamlit run chatbot.py --server.port 8502
```

### Slow Response Times
- **First response** may take 30-60 seconds (model loading)
- **Subsequent responses** will be faster
- Consider using a GPU for acceleration
- Adjust chunk size to reduce search time

### PDF Not Extracting Text
- Ensure PDF is text-based (not image/scanned)
- Try opening the PDF in Adobe Reader first
- Check PDF file size (very large PDFs may cause issues)

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | latest | Web UI framework |
| langchain | latest | LLM orchestration |
| langchain-community | latest | Community integrations |
| PyPDF2 | latest | PDF text extraction |
| FAISS | latest | Vector similarity search |
| HuggingFace-hub | latest | Embeddings provider |
| sentence-transformers | latest | Text embeddings |

## 🌐 Network Access

- **Local:** `http://localhost:8501`
- **Network:** `http://192.168.0.100:8501` (adjust IP as needed)

## 📝 Limitations & Notes

- **Text-based PDFs only** - Scanned/image PDFs won't work
- **Local processing** - No cloud storage or external API calls
- **Session-based** - Chat history resets when app restarts
- **Single user** - Designed for single-user local deployment
- **Model size** - Llama 3.2 requires ~4-8GB RAM

## 🔜 Future Improvements

- [ ] Support for multiple document formats (DOCX, TXT, MD)
- [ ] Multi-PDF support
- [ ] Export chat history
- [ ] Custom system prompts
- [ ] Performance metrics dashboard
- [ ] Web-based deployment
- [ ] Chat persistence to database

## 🚀 Performance Tips

1. **Use SSD storage** - Faster model loading
2. **Close other applications** - Free up RAM
3. **Use GPU** - If available, significantly faster inference
4. **Smaller PDFs** - Faster processing
5. **Adjust chunk size** - Smaller chunks = faster search but less context

## 📞 Support

If you encounter issues:

1. Check Ollama status: `ollama list`
2. Verify model is loaded: `ollama list llama3.2`
3. Check console output for errors
4. Try restarting Ollama: `ollama serve`

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Credits

- **Ollama** - Local LLM infrastructure
- **Streamlit** - Web UI framework
- **LangChain** - AI orchestration
- **PyPDF2** - PDF processing
- **FAISS** - Vector search
- **Llama 3.2** - Language model

---

**Version:** 1.0  
**Last Updated:** February 22, 2026  
**Status:** ✅ Production Ready
