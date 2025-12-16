import os
import json
from pathlib import Path
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

try:
    nlp_es = spacy.load("es_core_news_sm")
except OSError:
    raise RuntimeError("Ejecuta: python -m spacy download es_core_news_sm")

try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    nlp_en = None

DOCUMENTS_DIR = Path("../pdfs")
OUTPUT_FILE = Path("../salida/corpus_chunks.jsonl")
DOCUMENTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE.parent.mkdir(exist_ok=True)

def preprocess_for_retrieval(text: str, lang="es") -> str:

    nlp = nlp_es if lang == "es" else (nlp_en or nlp_es)
    doc = nlp(text.lower())
    tokens = [
        token.lemma_.strip()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
        and len(token.lemma_) > 2
        and token.pos_ in {"NOUN", "ADJ", "VERB", "PROPN"}  
    ]
    
    return " ".join(tokens)

def detect_language(text: str, threshold=0.7) -> str:
    es_words = sum(1 for w in text.lower().split() if w in {"el", "la", "de", "que", "y", "en", "es", "un", "una"})
    en_words = sum(1 for w in text.lower().split() if w in {"the", "and", "of", "to", "in", "is", "it", "this", "that"})
    return "es" if es_words > en_words * threshold else "en"

all_chunks = []

for filepath in DOCUMENTS_DIR.iterdir():
    if not filepath.is_file():
        continue

    try:
        # Cargar documento
        if filepath.suffix.lower() == ".pdf":
            docs = PyPDFLoader(str(filepath)).load()
        elif filepath.suffix.lower() == ".txt":
            docs = TextLoader(str(filepath), encoding="utf-8").load()
        else:
            continue

        for doc in docs:
            original_text = doc.page_content.strip()
            if len(original_text) < 50:
                continue

            lang = detect_language(original_text)
           
            cleaned_for_retrieval = preprocess_for_retrieval(original_text, lang=lang)
            doc.metadata.update({
                "source": filepath.name,
                "language": lang,
                "cleaned_terms_count": len(cleaned_for_retrieval.split())
            })

        
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=80,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents([doc])

            for chunk in chunks:
                chunk_clean = preprocess_for_retrieval(chunk.page_content, lang=lang)
                chunk.metadata["retrieval_text"] = chunk_clean  
                all_chunks.append(chunk)

        print(f"Processed: {filepath.name} ({lang})")

    except Exception as e:
        print(f"Error with {filepath.name}: {e}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(all_chunks):
        record = {
            "id": f"chunk_{i:06d}",
            "content": chunk.page_content.strip(),                     
            "retrieval_text": chunk.metadata.get("retrieval_text", ""), 
            "metadata": {
                "source": chunk.metadata.get("source", ""),
                "language": chunk.metadata.get("language", "es"),
                "page": chunk.metadata.get("page", None)
            }
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n Done! {len(all_chunks)} chunks with deep cleaning.") 