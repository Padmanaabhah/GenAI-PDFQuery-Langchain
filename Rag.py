from uuid import uuid4
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 
from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_chroma import Chroma 
import PyPDF2 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain


load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
COLLECTION_NAME = "pdfdata"

llm=None
vector_store=None
qa_chain = None
chat_history = []

def set_db_path(path: str):
    """Set the directory where Chroma DB will be stored."""
    global VECTORSTORE_DIR
    VECTORSTORE_DIR = path


def initialize_components():
    """Initialize Groq LLM and Chroma vector store (only once)."""
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500,
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )

        #VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
        )

def store_pdf_in_chromadb(pdf_path):
    """
    Reads the PDF file at pdf_path, removes all existing ChromaDB collections,
    splits the text using LangChain, and stores the chunks in a collection named 'pdfdata'.
    ChromaDB will be stored locally at chroma_db_path.
    """
    global qa_chain


    initialize_components()
    

    # Reset existing vector store/collection
    vector_store.reset_collection()

    # Read PDF content
    pdf_text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            pdf_text += page.extract_text() or ""

    # Split text into chunks using LangChain
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(pdf_text)
    
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": pdf_path, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents, ids=uuids)

    print(f"PDF chunks stored in ChromaDB collection 'pdfdata'. Total chunks: {len(documents)}")
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )


def generate_answer(question: str):
    """Run conversational Retrieval QA with sources over the vector store."""
    global qa_chain, chat_history

    if vector_store is None:
        raise RuntimeError("Vector database is not initialized")

    if qa_chain is None:
        raise RuntimeError("QA chain is not initialized")

    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    answer = result.get("answer", "")
    source_docs = result.get("source_documents", [])

    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in source_docs
    })

    # Update chat history so follow-ups work
    chat_history.append((question, answer))

    return answer, sources

# store_pdf_in_chromadb(r"C:\Users\padmp\Downloads\th.th_bangalore.18_11_2025.pdf") 

# while True:
#     question = input("\nEnter your question (or type 'exit' to quit): ")

#     if question.lower() in ["exit", "quit", "q"]:
#         print("Goodbye!")
#         break

#     answer, sources = generate_answer(question)

#     print("\nAnswer:", answer)
#     print("Sources:", sources)


