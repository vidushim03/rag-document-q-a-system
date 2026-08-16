import json
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.documents import Document

from src.chunker import chunk_data
from src.data_loader import load_pdf
from src.embedding import get_embeddings
from src.agent import stream_agent
from src.ocr_loader import load_image_text, load_pdf_with_ocr
from src.session_store import SessionVectorStore


APP_DIR = Path(__file__).resolve().parent
CHAT_FILE = APP_DIR / "chats.json"
LEGACY_CHAT_FILE = APP_DIR / "chat_history.json"
CHATS_STORAGE_KEY = "rag_chats_v1"
CHATS_BRIDGE_PARAM = "st_chats_bridge"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

EXAMPLE_PROMPTS = [
    "What is RAG?",
    "Summarize the key points of my documents.",
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-color: #0a0f1c;
        --panel-bg: rgba(148, 163, 184, 0.06);
        --panel-bg-strong: rgba(30, 41, 59, 0.55);
        --panel-border: rgba(148, 163, 184, 0.14);
        --text-main: #e6edf7;
        --text-muted: #8b98ad;
        --accent: #6366f1;
        --accent-2: #a855f7;
        --accent-glow: rgba(99, 102, 241, 0.45);
        --user-bubble: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.12));
        --user-border: rgba(129, 140, 248, 0.32);
    }

    html, body, .stApp {
        background:
            radial-gradient(1100px 600px at 12% -5%, rgba(99, 102, 241, 0.14), transparent 55%),
            radial-gradient(900px 500px at 95% 10%, rgba(168, 85, 247, 0.12), transparent 55%),
            var(--bg-color) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    .block-container {
        max-width: 880px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
    }

    h1, h2, h3, h4, p, label {
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif;
    }

    a {
        color: var(--accent) !important;
    }

    /* Welcome / empty state */
    .welcome-title {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.6rem 0 0.1rem;
    }

    .welcome-hint {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0.5rem 0 0.7rem;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: rgba(148, 163, 184, 0.08) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 12px !important;
        min-height: 2.6rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--user-border) !important;
        background: rgba(99, 102, 241, 0.12) !important;
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 18px var(--accent-glow);
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 26px var(--accent-glow);
        transform: translateY(-2px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--panel-border) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.2rem 0.1rem 0.9rem;
    }

    .brand-logo {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        border-radius: 9px;
        background: linear-gradient(135deg, var(--accent), #7c3aed);
        box-shadow: 0 3px 10px var(--accent-glow);
    }

    .brand-name {
        font-weight: 700;
        font-size: 0.98rem;
        letter-spacing: -0.01em;
        color: var(--text-main);
    }

    .brand-tag {
        font-size: 0.7rem;
        color: var(--text-muted);
    }

    .side-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 0.2rem 0 0.4rem;
    }

    .side-divider {
        border-top: 1px solid var(--panel-border);
        margin: 0.85rem 0;
    }

    .doc-chip {
        font-size: 0.9rem;
        color: var(--text-main);
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid var(--panel-border);
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-family: 'Inter', sans-serif;
    }

    /* Citations */
    .cite-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.6rem;
    }

    .cite-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        color: var(--text-muted);
        background: rgba(148, 163, 184, 0.07);
        border: 1px solid var(--panel-border);
        border-radius: 999px;
        padding: 0.3rem 0.75rem;
        font-family: 'Inter', sans-serif;
    }

    .upload-note {
        font-size: 0.8rem;
        color: #4ade80;
        margin-top: 0.5rem;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.3rem 0 !important;
        margin-bottom: 0.2rem !important;
        backdrop-filter: none !important;
    }

    div[data-testid="stChatMessageAvatarUser"] {
        display: none !important;
    }

    div[data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.35) !important;
    }

    div[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
        background: var(--user-bubble) !important;
        border: 1px solid var(--user-border) !important;
        border-radius: 18px 18px 6px 18px !important;
        padding: 0.5rem 0.9rem !important;
        max-width: 78% !important;
        margin-left: auto !important;
        flex: 0 1 auto !important;
    }

    [data-testid="stChatMessageContent"] {
        padding-top: 0.2rem !important;
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        color: var(--text-main);
        font-size: 0.97rem;
        line-height: 1.7;
    }

    [data-testid="stChatMessageContent"] pre {
        background: rgba(10, 15, 28, 0.7) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 10px !important;
        padding: 0.7rem 0.9rem !important;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1.5px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent) !important;
        box-shadow: 0 0 18px var(--accent-glow) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(148, 163, 184, 0.04) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 14px !important;
    }

    [data-testid="stExpander"] summary {
        color: var(--text-main) !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--panel-border) !important;
        border-left: 4px solid var(--accent) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(30, 41, 59, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 16px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--user-border) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.25);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(148, 163, 184, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_scanned_pdf(docs):
    watermarks = ["scanned by camscanner", "www.camscanner.com"]
    combined_text = " ".join(doc.page_content for doc in docs).lower()
    for watermark in watermarks:
        combined_text = combined_text.replace(watermark, "")
    return len(combined_text.strip()) < 50


def process_files(files):
    processed = []

    for file in files:
        suffix = Path(file.name).suffix.lower().lstrip(".")

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                docs = load_pdf(tmp_path)
                if not docs or is_scanned_pdf(docs):
                    st.info(f"{file.name} appears scanned, so OCR is being used.")
                    docs = load_pdf_with_ocr(tmp_path)
            else:
                text = load_image_text(tmp_path)
                docs = (
                    [Document(page_content=text, metadata={"source": file.name})]
                    if text.strip()
                    else []
                )

            for doc in docs:
                doc.metadata["source"] = file.name

            if docs:
                processed.append((file.name, docs))
            else:
                st.warning(f"No text could be extracted from {file.name}.")
        except Exception as exc:
            st.error(f"Error processing {file.name}: {exc}")
        finally:
            os.remove(tmp_path)

    return processed


def build_chat_history(messages):
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages[-8:]
    ]


def load_chats():
    bridge = st.query_params.get(CHATS_BRIDGE_PARAM)
    if bridge:
        st.query_params.clear()
        try:
            chats = json.loads(bridge)
            if isinstance(chats, dict):
                return chats
        except (json.JSONDecodeError, TypeError):
            pass

    try:
        with CHAT_FILE.open("r", encoding="utf-8") as file:
            chats = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        chats = {}

    if chats:
        has_messages = any(chat.get("messages") for chat in chats.values())
        if has_messages:
            return chats

    legacy_chats = migrate_legacy_history()
    if legacy_chats:
        with CHAT_FILE.open("w", encoding="utf-8") as file:
            json.dump(legacy_chats, file, indent=2, ensure_ascii=False)
        return legacy_chats

    return chats


def save_chats():
    with CHAT_FILE.open("w", encoding="utf-8") as file:
        json.dump(st.session_state.all_chats, file, indent=2, ensure_ascii=False)


def create_chat(title="New Chat", messages=None):
    chat_id = str(uuid.uuid4())[:6]
    st.session_state.all_chats[chat_id] = {
        "title": title,
        "messages": list(messages or []),
    }
    st.session_state.current_chat = chat_id
    save_chats()
    return chat_id


def ensure_current_chat():
    if st.session_state.current_chat not in st.session_state.all_chats:
        st.session_state.current_chat = None


def generate_title(query):
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        return "New Chat"
    return cleaned[:40] + "..." if len(cleaned) > 40 else cleaned


def smart_title(query):
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq

        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Create a short chat title of at most 5 words capturing the topic of the user's first question. Output only the title, no quotes and no punctuation."),
            ("human", "{query}"),
        ])
        title = (
            (prompt | llm | StrOutputParser())
            .invoke({"query": query})
            .strip()
            .strip('"')
            .strip("'")
            .strip()
        )
        if title and len(title) <= 50:
            return title
    except Exception:
        pass
    return generate_title(query)


def format_chat_title(chat_id):
    title = st.session_state.all_chats[chat_id]["title"]
    return title[:34] + "..." if len(title) > 34 else title


def migrate_legacy_history():
    try:
        with LEGACY_CHAT_FILE.open("r", encoding="utf-8") as file:
            legacy_items = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    if not isinstance(legacy_items, list) or not legacy_items:
        return {}

    messages = []
    for item in legacy_items:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})

    if not messages:
        return {}

    first_user_message = next(
        (message["content"] for message in messages if message["role"] == "user"),
        "New Chat",
    )
    chat_id = str(uuid.uuid4())[:6]
    return {chat_id: {"title": generate_title(first_user_message), "messages": messages}}


def get_document_store():
    chat_id = st.session_state.current_chat
    if chat_id and chat_id in st.session_state.chat_stores:
        return st.session_state.chat_stores[chat_id]
    if chat_id is None and st.session_state.draft_store is not None:
        return st.session_state.draft_store
    return None


def render_copy_button(text):
    payload = json.dumps(text).replace("</", "<\\/")
    html = f"""
    <div style="display:flex;justify-content:flex-end;margin-top:6px;">
        <button onclick="copyAnswer()" style="background:rgba(148,163,184,0.08);color:#e6edf7;border:1px solid rgba(148,163,184,0.3);border-radius:10px;padding:6px 16px;font-size:0.85rem;font-family:Inter,sans-serif;cursor:pointer;">Copy</button>
    </div>
    <script>
    function copyAnswer() {{
        var text = {payload};
        if (navigator.clipboard && navigator.clipboard.writeText) {{
            navigator.clipboard.writeText(text).catch(function() {{ fallbackCopy(text); }});
        }} else {{
            fallbackCopy(text);
        }}
    }}
    function fallbackCopy(text) {{
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }}
    </script>
    """
    components.html(html, height=48)


def render_chat_persistence_bridge():
    """If the browser has saved chats, hand them to the app once via a URL redirect."""
    payload = json.dumps(st.session_state.all_chats).replace("</", "<\\/")
    html = f"""
    <script>
    (function () {{
        try {{
            var raw = localStorage.getItem({json.dumps(CHATS_STORAGE_KEY)});
            if (!raw || raw.length > 80000) return;
            var url = new URL(window.location.href);
            if (url.searchParams.has({json.dumps(CHATS_BRIDGE_PARAM)})) return;
            if (raw === JSON.stringify({payload})) return;
            url.searchParams.set({json.dumps(CHATS_BRIDGE_PARAM)}, raw);
            window.location.replace(url.toString());
        }} catch (e) {{}}
    }})();
    </script>
    """
    components.html(html, height=0)


def persist_chats_localstorage():
    """Write the current chats into the browser's localStorage."""
    payload = json.dumps(st.session_state.all_chats).replace("</", "<\\/")
    html = f"""
    <script>
    try {{
        localStorage.setItem({json.dumps(CHATS_STORAGE_KEY)}, {payload});
    }} catch (e) {{}}
    </script>
    """
    components.html(html, height=0)


if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_chats()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "draft_messages" not in st.session_state:
    st.session_state.draft_messages = []

if "chat_stores" not in st.session_state:
    st.session_state.chat_stores = {}

if "draft_store" not in st.session_state:
    st.session_state.draft_store = None

if "upload_sig" not in st.session_state:
    st.session_state.upload_sig = None

ensure_current_chat()

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-logo">🧠</span>
            <span>
                <div class="brand-name">RAG Assistant</div>
                <div class="brand-tag">Document Q&amp;A</div>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("New chat", type="primary", use_container_width=True):
        st.session_state.current_chat = None
        st.session_state.draft_messages = []
        st.session_state.draft_store = None
        st.rerun()

    st.markdown('<div class="side-title">Chats</div>', unsafe_allow_html=True)
    chat_ids = list(reversed(list(st.session_state.all_chats.keys())))
    if chat_ids:
        for chat_id in chat_ids:
            is_active = st.session_state.current_chat == chat_id
            row = st.columns([0.86, 0.14])
            with row[0]:
                marker = "● " if is_active else "   "
                label = f"{marker}{format_chat_title(chat_id)}"
                if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                    st.session_state.current_chat = chat_id
                    st.session_state.draft_messages = []
                    st.rerun()
            with row[1]:
                if st.button("✕", key=f"del_{chat_id}", help="Delete chat", use_container_width=True):
                    st.session_state.all_chats.pop(chat_id, None)
                    st.session_state.chat_stores.pop(chat_id, None)
                    save_chats()
                    if st.session_state.current_chat == chat_id:
                        st.session_state.current_chat = None
                        st.session_state.draft_messages = []
                        st.session_state.draft_store = None
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    with st.expander("Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="doc_uploader",
        )
        st.caption("PDF · PNG · JPG · JPEG")

    store = get_document_store()
    if store and store.source_count() > 0:
        total_chunks = sum(s["chunk_count"] for s in store.list_sources())
        st.markdown(
            f'<div class="upload-note">✓ {store.source_count()} source(s) · {total_chunks} chunks in this chat</div>',
            unsafe_allow_html=True,
        )


if uploaded_files:
    current_signature = tuple((file.name, file.size) for file in uploaded_files)
    if st.session_state.upload_sig != current_signature:
        with st.spinner("Processing documents..."):
            processed = process_files(uploaded_files)

            if processed:
                store = get_document_store()
                if store is None:
                    store = SessionVectorStore(get_embeddings(mode="tfidf"))
                    if st.session_state.current_chat:
                        st.session_state.chat_stores[st.session_state.current_chat] = store
                    else:
                        st.session_state.draft_store = store

                total_chunks = 0
                for source, docs in processed:
                    chunks = chunk_data(docs)
                    result = store.add_documents(chunks, source_name=source)
                    total_chunks += result.get("chunk_count", 0)

                st.session_state.upload_sig = current_signature
                st.rerun()
            else:
                st.session_state.upload_sig = current_signature
                st.rerun()
if st.session_state.current_chat is None:
    chat = {"title": "New Chat", "messages": st.session_state.draft_messages}
else:
    chat = st.session_state.all_chats[st.session_state.current_chat]

messages = chat["messages"]

if not messages:
    st.markdown(
        '<div class="welcome-title">What can I help you with?</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="welcome-hint">Try asking</div>', unsafe_allow_html=True)
    prompt_columns = st.columns(2)
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        with prompt_columns[index % 2]:
            if st.button(f"“{prompt}”", use_container_width=True):
                st.session_state.pending_query = prompt
                st.rerun()

for message in messages:
    avatar = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

pending_query = st.session_state.pop("pending_query", None)
query = st.chat_input("Ask a question about your documents")
if pending_query and not query:
    query = pending_query

if query:
    history = build_chat_history(messages)
    messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="assistant"):
        store = get_document_store()
        has_kb = store is not None and store.source_count() > 0
        source_docs = []

        def answer_generator():
            for item in stream_agent(query, store if has_kb else None, history=history):
                if item[0] == "token":
                    yield item[1]
                else:
                    source_docs.extend(item[1])

        try:
            answer = st.write_stream(answer_generator())
        except Exception as exc:
            st.error(f"Could not generate an answer. Check GROQ_API_KEY. ({exc})")
            answer = "Sorry, I could not generate an answer."

        local_sources = []
        web_used = False
        for doc in source_docs:
            src = (doc.metadata or {}).get("source")
            if src == "web_search":
                web_used = True
            elif src:
                local_sources.append(src)

        if local_sources:
            chips = "".join(
                f'<span class="cite-chip">📄 {s}</span>'
                for s in dict.fromkeys(local_sources)
            )
            st.markdown(f'<div class="cite-row">{chips}</div>', unsafe_allow_html=True)
        elif web_used:
            st.caption("Answer included web search results.")

        render_copy_button(answer)

    messages.append({"role": "assistant", "content": answer})
    if st.session_state.current_chat is None:
        create_chat(title=smart_title(query), messages=messages)
        if st.session_state.draft_store is not None:
            st.session_state.chat_stores[st.session_state.current_chat] = st.session_state.draft_store
            st.session_state.draft_store = None
        st.session_state.draft_messages = []
        st.rerun()
    else:
        if chat["title"] == "New Chat":
            chat["title"] = smart_title(query)
        st.session_state.all_chats[st.session_state.current_chat] = chat
        save_chats()

render_chat_persistence_bridge()
persist_chats_localstorage()
