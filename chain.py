# chain.py
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------------------------------------------------------
# LLM CONFIG (set in .env)
#   GOOGLE_API_KEY          Gemini API key (required)
#   LLM_MODEL               main Gemini model
#   GEMINI_FALLBACK_MODELS  comma-separated models tried in order when the main
#                           model is overloaded (503), out of quota (429),
#                           retired (404) or times out
# -----------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL") or "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.5-flash,gemini-3.7-flash,gemini-3.8-flash").split(",")
    if m.strip()
]

# Max characters of transcript sent to the LLM in one call when summarizing.
# Large so most videos (under ~1.5 h) need a single call, saving free-tier quota
SUMMARY_BATCH_CHARS = 100000

# Number of previous question/answer pairs sent with each question
HISTORY_TURNS = 3

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions about a YouTube video "
     "using ONLY the transcript excerpts provided. If the answer is not in the "
     "excerpts, say you could not find it in the video. Use the earlier conversation "
     "only to understand follow-up questions. Answer clearly and concisely."),
    MessagesPlaceholder("history"),
    ("human", "Transcript excerpts:\n{context}\n\nQuestion: {question}"),
])

PARTIAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You summarize parts of YouTube video transcripts accurately and concisely."),
    ("human", "Summarize the key points of this part of a video transcript in a few bullet points:\n\n{text}"),
])

FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You summarize YouTube videos accurately based only on the provided material."),
    ("human", "Write a clear summary of the whole video in 4-6 bullet points, "
              "followed by a one-sentence takeaway.\n\nMaterial:\n\n{text}"),
])


def check_llm():
    """Return None if the Gemini API key is set, else a human-readable problem."""
    if not os.getenv("GOOGLE_API_KEY"):
        return "`GOOGLE_API_KEY` is not set. Add it to your `.env` file (see `.env.example`)."
    return None


def _gemini(model):
    return ChatGoogleGenerativeAI(
        model=model,  # Gemini 3.x uses fixed sampling, so no temperature here
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=45,
        max_retries=1,  # fail fast and move on to a fallback model
    )


def get_llm():
    """Main Gemini model, falling back to GEMINI_FALLBACK_MODELS when it fails."""
    fallbacks = [_gemini(m) for m in GEMINI_FALLBACK_MODELS if m != LLM_MODEL]
    llm = _gemini(LLM_MODEL).with_fallbacks(fallbacks) if fallbacks else _gemini(LLM_MODEL)
    # When every model is busy at once, wait a few seconds and try the whole chain again
    return llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def make_rag_chain(retriever):
    """RAG Chain for YouTube Chatbot (Gemini + FAISS retriever)."""
    llm = get_llm()

    def chain(query, history=None):
        """
        Retrieve relevant chunks and generate a response.
        history: earlier chat as [{"role": "user"|"assistant", "content": str}, ...]
        """
        history = (history or [])[-2 * HISTORY_TURNS:]

        # A follow-up like "explain that more" says little on its own,
        # so retrieval also uses the previous question.
        prev_questions = [m["content"] for m in history if m["role"] == "user"]
        search_query = f"{prev_questions[-1]}\n{query}" if prev_questions else query

        docs = retriever.invoke(search_query)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        messages = [("human" if m["role"] == "user" else "ai", m["content"]) for m in history]
        result = llm.invoke(QA_PROMPT.format_messages(context=context, question=query, history=messages))
        return result.text.strip(), docs

    return chain


def summarize_transcript(docs):
    """
    Summarize the WHOLE transcript (not just the top-k retrieved chunks) with map-reduce:
    batch chunks into ~SUMMARY_BATCH_CHARS pieces, summarize each, then combine.
    """
    llm = get_llm()

    batches, current = [], ""
    for doc in docs:
        if current and len(current) + len(doc.page_content) > SUMMARY_BATCH_CHARS:
            batches.append(current)
            current = ""
        current += doc.page_content + " "
    if current:
        batches.append(current)

    if len(batches) == 1:
        material = batches[0]
    else:
        partials = [
            llm.invoke(PARTIAL_SUMMARY_PROMPT.format_messages(text=b)).text.strip()
            for b in batches
        ]
        material = "\n\n".join(partials)

    return llm.invoke(FINAL_SUMMARY_PROMPT.format_messages(text=material)).text.strip()
