import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from mistralai.client import MistralClient
from dotenv import load_dotenv
import os
from typing import List

class MistralEmbeddings(Embeddings):
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.chunk_size = 16000

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        for word in words:
            if current_size + len(word) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for text in texts:
            chunks = self.chunk_text(text)
            chunk_embeddings = []
            for chunk in chunks:
                response = self.client.embeddings(model=self.model, input=[chunk])
                chunk_embeddings.append(response.data[0].embedding)
            if len(chunk_embeddings) > 1:
                avg_embedding = [sum(e) / len(e) for e in zip(*chunk_embeddings)]
                all_embeddings.append(avg_embedding)
            else:
                all_embeddings.extend(chunk_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        chunks = self.chunk_text(text)
        chunk_embeddings = []
        for chunk in chunks:
            response = self.client.embeddings(model=self.model, input=[chunk])
            chunk_embeddings.append(response.data[0].embedding)
        if len(chunk_embeddings) > 1:
            return [sum(e) / len(e) for e in zip(*chunk_embeddings)]
        return chunk_embeddings[0]

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

def create_vector_store(chunks, embeddings):
    return FAISS.from_texts(chunks, embedding=embeddings)

def create_qa_chain(client, model):
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
            {"role": "system", "content": prompt.format(context=context, question=input_data["question"])},
            {"role": "user", "content": input_data["question"]}
        ]
        response = client.chat(model=model, messages=messages)
        return {"output_text": response.choices[0].message.content}
    
    return qa_function

def main():
    st.set_page_config(page_title="AI Research Paper Analysis Chatbot")
    st.title("Chat with Your Research PDFs")

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    embedding_model = "mistral-embed"
    chat_model = "mistral-tiny"
    client = MistralClient(api_key=api_key)
    embeddings = MistralEmbeddings(client, embedding_model)

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        process_files = st.button("Process Files")

    if process_files and pdf_docs:
        with st.spinner("Processing files..."):
            text = get_pdf_text(pdf_docs)
            if not text.strip():
                st.warning("No text found in PDFs. Please check your files.")
                return
            chunks = get_text_chunks(text)
            st.session_state.vector_store = create_vector_store(chunks, embeddings)
        st.success("Files processed successfully!")

    user_question = st.text_input("Ask a question about the uploaded PDFs")
    if user_question and st.session_state.vector_store:
        try:
            relevant_docs = st.session_state.vector_store.similarity_search(user_question)
            qa_chain = create_qa_chain(client, chat_model)
            response = qa_chain({"input_documents": relevant_docs, "question": user_question})
            st.write("**Answer:**", response["output_text"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    elif user_question:
        st.warning("Please upload and process PDF files before asking questions.")

if __name__ == "__main__":
    main()
