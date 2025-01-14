from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from typing import List

app = Flask(__name__)

class MistralEmbeddings(Embeddings):
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.chunk_size = 8000

    def chunk_text(self, text: str) -> List[str]:
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

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
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text, chunk_size=5000, chunk_overlap=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


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


# Global variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "ai-research"
pinecone_env = "us-east-1"
embedding_model = "text-embedding-3-large"

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)
print(f"Pinecone Environment: {pinecone_env}")

# Create index if it doesn't exist
if pinecone_index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=pinecone_index_name,
        dimension=3072,  # Based on the embedding model used
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_env
        )
    )
index = pinecone.Index(pinecone_index_name)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_pdfs():
    pdf_files = request.files.getlist("pdf_files")
    if not pdf_files:
        return jsonify({"error": "No PDF files uploaded."}), 400

    try:
        text = get_pdf_text(pdf_files)
        if not text.strip():
            return jsonify({"error": "No text found in PDFs. Please check your files."}), 400

        chunks = get_text_chunks(text)
        for i, chunk in enumerate(chunks):
            index.upsert([(f"chunk_{i}", embeddings.embed_query(chunk))])
        return jsonify({"message": "Files processed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    user_question = request.form.get("question")
    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    try:
        response = index.query(user_question, top_k=5, include_metadata=True)
        relevant_docs = [{"page_content": match["metadata"].get("text", "")} for match in response.matches]
        qa_chain = create_qa_chain(client, embedding_model)
        answer = qa_chain({"input_documents": relevant_docs, "question": user_question})
        return jsonify({"answer": answer["output_text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
