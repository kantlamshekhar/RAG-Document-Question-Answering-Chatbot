import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from time import sleep

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_processing(path):
    # Document Loading
    loader = PyPDFLoader(path)
    doc = loader.load() # Load the PDF document
    # print(f"Loaded {len(doc)} pages from the PDF document.")

    # Splitting the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(doc) # Split the document into chunks
    # print(f"Loaded {len(docs)} chunks from the PDF document.")

    # Embeddings and Vector Store Creation
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    ) # Create embeddings using Google Generative AI
    vector_db = InMemoryVectorStore.from_documents(documents = docs, embedding=embeddings) # Create an in-memory vector store from the document chunks 

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True

# Building the Streamlit App
st.title("Document Question Answering Chatbot")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

# File Uploaad Section
if not st.session_state.document_uploaded:
    file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())
        
        with st.spinner("Processing the uploaded document..."):
            document_processing("./uploaded_document.pdf") # Process the uploaded document
        
        st.markdown("Document uploaded successfully!")
        sleep(2)
        st.rerun() # Rerun the app to update the state after document processing

# Chat Section
if st.session_state.document_uploaded and st.session_state.vector_db:
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]
        st.chat_message(role).markdown(content) # Display previous messages in the chat interface
    
    query = st.chat_input("Ask a question about the document:")
    if query:
        st.session_state.messages.append({"role": "user", "content": query}) # Store the user's question in the session state
        st.chat_message("user").markdown(query) # Display the user's question in the chat interface
        # Retrieve relevant chunks from the vector store
        documents = st.session_state.vector_db.similarity_search(query, k=3) # Retrieve top 3 relevant chunks
        context = " "

        for doc in documents:
            context += doc.page_content + "\n" # Concatenate the content of the retrieved chunks to form the context
        
        # Create a prompt for the chat model
        prompt = f"""
        You are a helpful assistant that answers questions based on the provided context. Use the context to answer the question accurately. If the answer is not present in the context, respond with "I don't know."
        Context: {context}
        Question: {query}
        Answer:
        """
        try:
            chat_model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
            ) # Initialize the chat model using Google Generative AI
            answer = chat_model.invoke(prompt) # Get the answer from the chat model
            st.session_state.messages.append({"role": "ai", "content": answer}) # Store the AI's answer in the session state
            st.chat_message("ai").markdown(f"**Answer:** {answer}") # Display the answer
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                st.error("❌ API Quota Exceeded: Your free tier quota has been exhausted. Please wait for your quota to reset or check your billing details at: https://ai.dev/rate-limit")
            else:
                st.error(f"❌ Error: {str(e)}")