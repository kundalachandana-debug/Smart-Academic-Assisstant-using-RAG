import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR        = "pdfs"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200


def load_documents(data_dir=DATA_DIR):
    documents = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No files found in '{data_dir}/'")

    print("Loading study materials...")

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        # Skip folders
        if not os.path.isfile(file_path):
            continue

        try:
            if file.lower().endswith(".pdf"):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

            elif file.lower().endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(file_path)
                documents.extend(loader.load())

            elif file.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
                documents.extend(loader.load())

            elif file.lower().endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        except Exception as e:
            print(f"Skipping file {file}: {e}")

    for doc in documents:
        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        doc.metadata["subject"] = os.path.splitext(filename)[0].replace("_", " ").title()

    return documents


def load_and_split():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# ✅ FIXED FUNCTION (APPEND INSTEAD OF OVERWRITE)
def build_vectorstore(chunks, persist_path=VECTORSTORE_DIR):
    print("Building vectorstore...")
    embeddings = get_embeddings()

    # If vectorstore already exists → append
    if os.path.exists(persist_path):
        print("Loading existing vectorstore...")
        db = FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        db.add_documents(chunks)

    else:
        print("Creating new vectorstore...")
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(persist_path)
    print("Vectorstore saved")

    return db


def load_vectorstore(persist_path=VECTORSTORE_DIR):
    embeddings = get_embeddings()

    return FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_available_subjects(data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        return []

    subjects = []

    for f in os.listdir(data_dir):
        if f.endswith((".pdf", ".pptx", ".docx", ".txt")):
            subjects.append(
                os.path.splitext(f)[0].replace("_", " ").title()
            )

    return sorted(subjects)


if __name__ == "__main__":
    chunks = load_and_split()
    build_vectorstore(chunks)
    print("Subjects:", ", ".join(get_available_subjects()))
    
    