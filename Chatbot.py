import streamlit as st
from langchain_community.llms import Ollama  # Correct import
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize the Llama 3.2 model (ensure your model is available via `ollama list`)
model = Ollama(model='llama3.2')  # Using your Llama 3.2 model

# Function to break PDF into chunks
def pdf_to_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit app
st.title("PDF Chatbot with Llama 3.2")

# File upload
uploaded_file = st.file_uploader("Train Your File", type="pdf")
if uploaded_file:
    chunks = pdf_to_chunks(uploaded_file)
    vector_store = create_vector_store(chunks)
    st.write("✅ File processed and stored in vector database.")

    # Input prompt
    user_input = st.text_input("Enter Your Prompt:")

    # Display result and store chat history
    if st.button("Submit") and user_input:
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
        result = qa_chain.run(user_input)

        # Append chat history
        st.session_state['chat_history'].append((user_input, result))

        # Display the result
        st.write(f"**Bot:** {result}")

    # Display chat history
    if st.session_state['chat_history']:
        st.write("### Chat History")
        for user_msg, bot_msg in st.session_state['chat_history']:
            st.write(f"**You:** {user_msg}")
            st.write(f"**Bot:** {bot_msg}")
            st.write("---")
