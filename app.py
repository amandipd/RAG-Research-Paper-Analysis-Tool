import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_embeddings(chunks, client, model):
    response = client.embeddings(model=model, input=chunks)
    return [data.embedding for data in response.data]

def create_or_load_vector_store(chunks, client, model):
    """Create a FAISS vector store."""
    embeddings = get_embeddings(chunks, client, model)
    try:
        return FAISS.load_local("faiss_index")
    except Exception:
        vector_store = FAISS.from_embeddings(zip(chunks, embeddings), embedding=None)
        vector_store.save_local("faiss_index")
        return vector_store

def create_qa_chain(client, model):
    """Create a question-answering chain with a custom prompt."""
    prompt = PromptTemplate(
        template="""
        Answer the question based on the context provided. If the answer is not in the context, say "answer is not in the context".
        Context: {context}
        Question: {question}
        Answer:
        """,
        input_variables=["context", "question"]
    )
    
    def qa_function(input_data):
        context = "\n".join([doc.page_content for doc in input_data["input_documents"]])
        messages = [
            ChatMessage(role="system", content=prompt.format(context=context, question=input_data["question"])),
            ChatMessage(role="user", content=input_data["question"])
        ]
        response = client.chat(model=model, messages=messages)
        return {"output_text": response.choices[0].message.content}
    
    return qa_function

def main():
    st.set_page_config(page_title="AI Research Paper Analysis Chatbot")
    st.title("Chat with Your Research PDFs")

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        process_files = st.button("Process Files")

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    embedding_model = "mistral-embed"
    chat_model = "mistral-tiny"  # or "mistral-small", "mistral-medium" depending on your needs
    client = MistralClient(api_key=api_key)

    if process_files and pdf_docs:
        with st.spinner("Processing files..."):
            text = get_pdf_text(pdf_docs)
            if not text.strip():
                st.warning("No text found in PDFs. Please check your files.")
                return
            chunks = get_text_chunks(text)
            vector_store = create_or_load_vector_store(chunks, client, embedding_model)
        st.success("Files processed successfully!")

    user_question = st.text_input("Ask a question about the uploaded PDFs")
    if user_question:
        try:
            vector_store = FAISS.load_local("faiss_index")
            relevant_docs = vector_store.similarity_search(user_question)
            qa_chain = create_qa_chain(client, chat_model)
            response = qa_chain({"input_documents": relevant_docs, "question": user_question})
            st.write("**Answer:**", response["output_text"])
        except FileNotFoundError:
            st.error("Vector store not found. Please process files first.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
