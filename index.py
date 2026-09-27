# index.py
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once (downloaded from HuggingFace on first run, ~90 MB)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def split_transcript(transcript_text):
    """Split the transcript into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator="end",  # keep ". " at the end of a sentence, not the start of the next chunk
    )
    return text_splitter.create_documents([transcript_text])


def build_faiss_index(transcript_text, k=4):
    """
    Build an in-memory FAISS index for the given transcript text.
    Returns (retriever, docs) so callers can also use the raw chunks (e.g. for summaries).
    """
    # Step 1: Split the transcript into overlapping chunks
    docs = split_transcript(transcript_text)
    if not docs:
        raise ValueError("Transcript is empty — nothing to index.")

    # Step 2 + 3: Embed chunks and store them in FAISS
    vectorstore = FAISS.from_documents(docs, get_embeddings())

    # Step 4: Return retriever object for similarity search
    retriever = vectorstore.as_retriever(search_kwargs={"k": min(k, len(docs))})
    return retriever, docs
