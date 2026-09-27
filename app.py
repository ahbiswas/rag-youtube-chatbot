import os

import streamlit as st
from dotenv import load_dotenv

# Load .env before importing chain.py so LLM settings are picked up
load_dotenv()

# On Streamlit Community Cloud, settings come from the app's Secrets
# (top-level keys or keys inside a [section])
SETTINGS = ("GOOGLE_API_KEY", "LLM_MODEL", "GEMINI_FALLBACK_MODELS")
try:
    for key, value in st.secrets.to_dict().items():
        items = value.items() if isinstance(value, dict) else [(key, value)]
        for k, v in items:
            if k in SETTINGS and not os.getenv(k):
                os.environ[k] = str(v).strip()
except Exception:
    pass  # no secrets.toml (e.g. running locally with .env)

from loader import extract_video_id, fetch_transcript, transcript_to_text
from index import build_faiss_index
from chain import make_rag_chain, summarize_transcript, check_llm, LLM_MODEL

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG-Based YouTube Chatbot",
    layout="centered",
)


def md(text):
    """Escape '$' so Streamlit does not render dollar amounts as LaTeX math."""
    return text.replace("$", "\\$")


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("RAG-Based YouTube Chatbot")
st.markdown(
    f"""
    <div style='text-align:center;'>
    <b>Built using LangChain · FAISS · Google Gemini ({LLM_MODEL})</b><br>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

llm_problem = check_llm()
if llm_problem:
    st.error(llm_problem)
    st.stop()


# -----------------------------------------------------------------------------
# CACHED KNOWLEDGE BASE BUILD (one per video)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=5)
def build_knowledge_base(video_id):
    transcript_text = transcript_to_text(fetch_transcript(video_id))
    retriever, docs = build_faiss_index(transcript_text)
    return transcript_text, retriever, docs


# -----------------------------------------------------------------------------
# INPUT SECTION
# -----------------------------------------------------------------------------
url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=abc123xyz")

if st.button("Build Knowledge Base", disabled=not url):
    try:
        video_id = extract_video_id(url)
        with st.spinner("Fetching transcript & building FAISS index (first run downloads the embedding model)..."):
            transcript_text, retriever, docs = build_knowledge_base(video_id)
        st.session_state.update(
            video_id=video_id,
            transcript_text=transcript_text,
            retriever=retriever,
            docs=docs,
            messages=[],
        )
    except Exception as e:
        st.error(f"Error while building knowledge base: {e}")

if "retriever" not in st.session_state:
    st.info("Paste a YouTube video link above and click **Build Knowledge Base** to begin.")
    st.stop()

# -----------------------------------------------------------------------------
# KNOWLEDGE BASE INFO
# -----------------------------------------------------------------------------
video_id = st.session_state.video_id
col_img, col_info = st.columns([1, 1])
col_img.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", width="stretch")
col_info.success("Knowledge Base ready")
col_info.markdown(
    f"**Video ID:** `{video_id}`  \n"
    f"**Words:** {len(st.session_state.transcript_text.split()):,}  \n"
    f"**Chunks indexed:** {len(st.session_state.docs)}"
)
with st.expander("View transcript"):
    st.write(md(st.session_state.transcript_text))

# -----------------------------------------------------------------------------
# Q&A / SUMMARIZATION SECTION
# -----------------------------------------------------------------------------
st.markdown("### Ask about the video")

c1, c2 = st.columns(2)
summarize_btn = c1.button("Summarize Video", width="stretch")
if c2.button("Clear chat", width="stretch"):
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("Transcript excerpts used"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**{i}.** {md(src)}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(md(msg["content"]))
        if msg.get("sources"):
            show_sources(msg["sources"])

query = st.chat_input("e.g., What are the main points discussed?")


def run_turn(user_text, generate, spinner_text):
    """Show the user message, generate the answer, and store both in the chat history."""
    # Earlier turns (without failed ones) give the LLM context for follow-up questions
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if not m.get("error")
    ]
    user_msg = {"role": "user", "content": user_text}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(md(user_text))

    with st.chat_message("assistant"):
        sources, error = [], False
        with st.spinner(spinner_text):
            try:
                answer, sources = generate(history)
            except Exception as e:
                answer, error = f"LLM error: {e}", True
        st.markdown(md(answer))
        if sources:
            show_sources(sources)

    # A failed answer makes its question useless as context too
    user_msg["error"] = error
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "error": error}
    )


def summarize(history):
    return summarize_transcript(st.session_state.docs), []


def answer_question(history):
    answer, docs = make_rag_chain(st.session_state.retriever)(query, history)
    return answer, [d.page_content for d in docs]


if summarize_btn:
    run_turn("Summarize this video.", summarize, "Gemini is summarizing the whole transcript...")
elif query:
    run_turn(query, answer_question, "Gemini is thinking...")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:14px;'>
        <b>RAG-Based YouTube Chatbot</b> · Powered by <b>LangChain</b>, <b>FAISS</b>,
        <b>HuggingFace</b> & <b>Google Gemini</b><br>
        Originally developed by <b>Sohan Ghosh</b> | MSc Data Science & AI
    </div>
    """,
    unsafe_allow_html=True
)
