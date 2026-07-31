# RAG-Document-Question-Answering-Chatbot

Overview
This project is a Retrieval-Augmented Generation (RAG) based Document
Question Answering Chatbot built with Python, Streamlit, LangChain, and
Google Gemini.

Users can: - Upload a PDF document - Ask questions about the uploaded
document - Receive AI-generated answers based only on the document
content

The application extracts text from the PDF, splits it into chunks,
creates embeddings, stores them in an in-memory vector database,
retrieves the most relevant content, and uses Google Gemini to generate
accurate answers.

Features

-   PDF document upload
-   Automatic text extraction
-   Document chunking using RecursiveCharacterTextSplitter
-   Semantic search using Gemini Embeddings
-   In-memory vector database
-   AI-powered question answering with Gemini 2.0 Flash
-   Chat interface built with Streamlit
-   Session state for chat history
-   Basic error handling for API quota issues

Technologies Used

-   Python
-   Streamlit
-   LangChain
-   Google Gemini API
-   Google Generative AI Embeddings
-   PyPDFLoader
-   RecursiveCharacterTextSplitter
-   InMemoryVectorStore
-   python-dotenv

Project Workflow

1.  Upload a PDF document.
2.  Extract text from the PDF.
3.  Split the text into smaller chunks.
4.  Generate embeddings for each chunk.
5.  Store embeddings in an in-memory vector database.
6.  Ask a question.
7.  Retrieve the most relevant chunks.
8.  Send the context and question to Gemini.
9.  Display the AI-generated answer.

Installation

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GOOGLE_API_KEY=YOUR_API_KEY

Run the application:

streamlit run app.py

Future Improvements

-   Support PDF files
-   Persistent vector database (FAISS/Chroma)
-   Conversation memory
-   Source citations
-   Deployment on Streamlit Cloud or Render

Author
Developed as an AI Engineer portfolio project demonstrating RAG,
LangChain, Vector Search, and Google Gemini integration.
