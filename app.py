import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import DB_ROOT, GROQ_MODEL
from src.guardrails import check_input_safety, check_output_safety
from src.knowledge import (
    clear_knowledge_base,
    get_retriever,
    ingest_sources,
    unique_sources,
)


load_dotenv()
os.environ.setdefault("USER_AGENT", "AI-Document-Assistant/1.0")

st.set_page_config(
    page_title="AI Document Assistant", page_icon="📘", layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
    }
    .title {
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        color: #16324f;
        margin-top: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #52616b;
        margin-bottom: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return ChatGroq(api_key=key, model_name=GROQ_MODEL, temperature=0)


def get_recent_history(limit=6):
    return st.session_state.messages[-limit:]


def resolve_query(user_query):
    lowered = user_query.lower().strip()
    followup_markers = [
        "it",
        "this",
        "that",
        "they",
        "them",
        "whose",
        "whom",
        "which",
        "refers",
        "refer to",
        "what about",
    ]
    if len(lowered.split()) <= 5 or any(
        marker in lowered for marker in followup_markers
    ):
        previous_user = next(
            (
                msg["content"]
                for msg in reversed(st.session_state.messages)
                if msg["role"] == "user"
            ),
            "",
        )
        if previous_user:
            return f"{user_query}\n\nPrevious question: {previous_user}"
    return user_query


def build_answer(llm, question, context, sources, history_text=""):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a factual RAG assistant for students. "
                "Focus on school and college documents, notices, exam details, course information, admissions, and campus links. "
                "Answer only from the provided context. If the answer is missing, say you don't know. "
                "Use the recent conversation history to resolve follow-up questions. "
                "Keep the answer concise.",
            ),
            (
                "human",
                "Recent conversation:\n{history}\n\nQuestion: {question}\n\nContext:\n{context}\n\nSources:\n{sources}",
            ),
        ]
    )
    result = llm.invoke(
        prompt.format_messages(
            question=question, context=context, sources=sources, history=history_text
        )
    ).content
    return check_output_safety(result)


def ensure_retriever(uploaded_files, link_text, replace_existing):
    retriever = get_retriever()
    if retriever is not None:
        return retriever

    if not uploaded_files and not (link_text or "").strip():
        return None

    count = ingest_sources(uploaded_files, link_text, replace_existing)
    if count:
        st.toast(f"Indexed {count} chunks", icon="✅")
        return get_retriever()
    return None


def extract_institution_items(docs):
    for doc in docs:
        if (doc.metadata or {}).get("page_type") == "institutions":
            items = [
                line.strip()
                for line in doc.page_content.splitlines()
                if line.strip() and line.strip().lower() != "view"
            ]
            return items, (doc.metadata or {}).get("item_count")
    return [], None


def is_count_question(text):
    lowered = text.lower()
    return any(
        word in lowered for word in ["how many", "number", "count", "so many", "total"]
    )


def is_list_question(text):
    lowered = text.lower()
    return any(
        word in lowered
        for word in ["what are", "which are", "mention", "mentioned", "list"]
    )


st.markdown('<div class="title">AI Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Student-focused Q&A from notes, notices, and college links</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload documents or paste links in the sidebar, then ask a question.",
        }
    ]

with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload files", accept_multiple_files=True, type=None
    )
    link_text = st.text_area(
        "Paste URLs", height=120, placeholder="https://example.com"
    )
    replace_existing = st.checkbox("Replace existing knowledge base", value=True)

    if st.button("Build knowledge base"):
        with st.spinner("Indexing your sources..."):
            count = ingest_sources(uploaded_files, link_text, replace_existing)
        if count:
            st.success(f"Indexed {count} chunks.")
            st.rerun()
        else:
            st.warning("No readable documents or links were provided.")

    if st.button("Reset knowledge base"):
        clear_knowledge_base()
        st.success("Knowledge base cleared.")
        st.rerun()

    st.caption(f"DB root: {DB_ROOT}")

llm = get_llm()
if llm is None:
    st.warning("Add `GROQ_API_KEY` to `.env` to enable chat.")
    st.stop()

for message in st.session_state.messages:
    avatar = "👨‍🎓" if message["role"] == "user" else "🏫"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about your documents..."):
    safe, reason = check_input_safety(user_query)
    if not safe:
        with st.chat_message("assistant", avatar="🛡️"):
            st.error(reason)
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👨‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🏫"):
        with st.spinner("Searching sources..."):
            retriever = ensure_retriever(uploaded_files, link_text, replace_existing)
            resolved_query = resolve_query(user_query)
            docs = retriever.invoke(resolved_query) if retriever else []
            if not docs:
                answer = "I don't know based on the indexed documents and links."
            else:
                handled = False
                institution_items, item_count = extract_institution_items(docs)
                if institution_items and (
                    is_count_question(user_query)
                    or is_list_question(user_query)
                    or "institution" in user_query.lower()
                ):
                    count_value = item_count or len(institution_items)
                    if is_count_question(user_query):
                        answer = f"There are {count_value} institutions mentioned in the link."
                    else:
                        answer = "The institutions mentioned are:\n" + "\n".join(
                            f"- {item}" for item in institution_items
                        )
                    sources = unique_sources(docs)
                    source_text = (
                        "\n".join(f"- {source}" for source in sources)
                        if sources
                        else "- Indexed sources"
                    )
                    answer = f"{answer}\n\nSources:\n{source_text}"
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    handled = True

                if not handled:
                    context = "\n\n".join(doc.page_content for doc in docs)
                    sources = unique_sources(docs)
                    history_text = "\n".join(
                        f"{msg['role'].title()}: {msg['content']}"
                        for msg in get_recent_history()
                    )
                    source_text = (
                        "\n".join(f"- {source}" for source in sources)
                        if sources
                        else "- Indexed sources"
                    )
                    answer = build_answer(
                        llm, user_query, context, source_text, history_text=history_text
                    )
                    answer = f"{answer}\n\nSources:\n{source_text}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
