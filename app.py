import os
import tempfile
import streamlit as st

from Rag import set_db_path, store_pdf_in_chromadb, generate_answer

st.set_page_config(page_title="PDF Reader GenAI", page_icon="📄")
st.title("📄 PDF Q&A (Groq + Chroma + LangChain)")

st.markdown(
    "Upload a PDF, choose where to store the Chroma DB, then ask questions about the document."
)

# ---- DB path input ----
default_db_dir = "C:\\PADMP\\Personal\\Projects\\PDF_Reader_GenAI\\chroma_db"
db_path = st.text_input(
    "Path to save Chroma DB",
    value=default_db_dir,
    help="This folder will be used to store the local Chroma database.",
)

# ---- PDF upload ----
uploaded_pdf = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    help="Select the PDF you want to index and query.",
)

# Keep a flag in session to know if PDF has been processed
if "pdf_indexed" not in st.session_state:
    st.session_state.pdf_indexed = False

# ---- Button to process PDF ----
if st.button("📥 Process & Save PDF to DB"):
    if not uploaded_pdf:
        st.error("Please upload a PDF file first.")
    elif not db_path.strip():
        st.error("Please specify a valid path to save the DB.")
    else:
        try:
            # Ensure directory exists
            os.makedirs(db_path, exist_ok=True)

            # Tell rag.py where to store the DB
            set_db_path(db_path)

            # Save uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                temp_pdf_path = tmp.name

            # Call your existing function from rag.py
            with st.spinner("Processing PDF and saving data to ChromaDB..."):
                store_pdf_in_chromadb(temp_pdf_path)

            # Mark as indexed
            st.session_state.pdf_indexed = True
            st.success(
                f"✅ Processed PDF and stored chunks in ChromaDB at:\n`{db_path}`"
            )

            # Optionally clean up temp file
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass

        except Exception as e:
            st.session_state.pdf_indexed = False
            st.error(f"Error while processing PDF: {e}")

st.markdown("---")

# ---- Q&A section ----
st.subheader("💬 Ask a question about the PDF")

question = st.text_input(
    "Your question",
    placeholder="e.g., Any news on 'Namma Metro'?",
)

if st.button("🔍 Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.pdf_indexed:
        st.warning("Please upload and process a PDF first.")
    else:
        try:
            answer, sources = generate_answer(question)

            st.markdown("### ✅ Answer")
            st.write(answer)

            st.markdown("### 📄 Sources")
            if sources:
                for src in sources:
                    st.write(f"- {src}")
            else:
                st.write("No specific sources returned.")
        except Exception as e:
            st.error(f"Error while generating answer: {e}")
