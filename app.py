import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, dotenv_values
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_or_create_vector_store(text_chunks=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Try loading the existing FAISS index
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.success("Loaded existing FAISS index.")
    except Exception:
        # Create a new index if loading fails
        if text_chunks:
            st.warning("No FAISS index found or corrupted. Creating a new one...")
            vector_store = get_vector_store(text_chunks)
        else:
            raise ValueError("No text chunks provided to create a new FAISS index.")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context say, "answer is not in the context", don't provide the wrong answer.\n\n
    Context: \n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = genai.GenerativeModel(model_name="gemini-pro", temperature= 1.0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response["output_text"])

def main():
    print('hi')

    st.set_page_config(page_title="Research Paper Analyzer")
    st.header("Chat with paper using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files and Click on the Submit & Process", type=["pdf"], accept_multiple_files=True)
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.warning("No text extracted from PDFs. Please check the files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = load_or_create_vector_store(text_chunks)
                        st.success("Processing completed!")
                except Exception as e:
                    st.error(f"An error occurred while processing the files: {e}")

if __name__ == "__main__":
    main()
