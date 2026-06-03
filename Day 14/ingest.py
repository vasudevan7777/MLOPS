import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


documents = []

pdf_folder = "Railway_Docs"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)

        print(f"Loading {file}")

        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())


print(f"\nTotal Pages Loaded: {len(documents)}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.from_documents(
    chunks,
    embeddings
)


db.save_local("vectorstore")

print("\nVector Database Created Successfully!")