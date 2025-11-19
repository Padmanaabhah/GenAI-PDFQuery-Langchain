PDF Question-Answering System (RAG + Groq + LangChain + ChromaDB)

This project allows you to upload any PDF and ask natural-language questions about its content using a Retrieval-Augmented Generation (RAG) pipeline. The system automatically processes the PDF, splits it into chunks, generates embeddings, stores them in a vector database, and uses a Groq-powered LLM to generate accurate, grounded answers — including follow-up question support.

🚀 Features

Upload any PDF and index it automatically

Ask questions in natural language

Supports conversational follow-up questions

Ultra-fast responses using Groq Llama-3.3-70B

Local vector storage via ChromaDB

Clean interface powered by Streamlit

Modular backend (rag.py) and frontend (app.py)

🛠️ Technology Used

Python 3, Streamlit

LangChain (chains, retrievers, RAG pipeline)

Groq LLM (ChatGroq client)

ChromaDB for vector storage

HuggingFace Embeddings

PyPDF2 for PDF extraction

📁 Project Structure
📦 project-root
 ┣ 📄 app.py        # Streamlit UI
 ┣ 📄 rag.py        # RAG logic, vector DB, LLM chain
 ┣ 📄 requirements.txt
 ┗ 📄 README.md

▶️ How to Run
pip install -r requirements.txt
streamlit run app.py

💡 Use Cases

News article analysis

Legal or medical document interrogation

Research paper summarization

Personal knowledge management
