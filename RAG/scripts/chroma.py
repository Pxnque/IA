import json
from pathlib import Path
from langchain_core.documents import Document  
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


CORPUS_PATH = Path("../salida/corpus_chunks.jsonl")
CHROMA_PATH = Path("../salida/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ Cargar documentos
documents = []
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            documents.append(
                Document(
                    page_content=data["content"],
                    metadata=data.get("metadata", {})
                )
            )

print(f"{len(documents)} chunks")


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=str(CHROMA_PATH),
    collection_name="filosofia_rag"
)

