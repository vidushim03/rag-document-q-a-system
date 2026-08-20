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
            elif suffix == "docx":
                from src.data_loader import load_docx
                docs = load_docx(tmp_path)
            elif suffix == "txt":
                from src.data_loader import load_txt
                docs = load_txt(tmp_path)
            elif suffix == "csv":
                from src.data_loader import load_csv
                docs = load_csv(tmp_path)
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
        st.session_state.persistence_bridge_rendered = True
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
    low = cleaned.lower()
    for lead in (
        "what is ", "what are ", "what's ", "whats ", "how to ", "how do i ",
        "how can i ", "why does ", "why do ", "can you ", "could you ",
        "please ", "tell me ", "summarize ", "summarise ", "explain ",
        "describe ", "give me ", "i need ", "help me ",
    ):
        if low.startswith(lead):
            cleaned = cleaned[len(lead):].strip()
            break
    short = " ".join(cleaned.split()[:6]).strip()
    if not short:
        short = cleaned
    if len(short) > 42:
        short = short[:39].rstrip() + "..."
    return short[0].upper() + short[1:]


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
    btn_id = "btn_" + str(uuid.uuid4()).replace("-", "")
    payload = json.dumps(text).replace("</", "<\\/")
    html = f"""
    <div style="display:flex;justify-content:flex-end;margin-top:10px;margin-bottom:4px;">
        <button id="{btn_id}" onclick="copyAnswer_{btn_id}()" style="background:linear-gradient(135deg,var(--accent),#7c3aed);color:#fff;border:none;border-radius:10px;padding:8px 18px;font-size:0.85rem;font-weight:600;font-family:Inter,sans-serif;cursor:pointer;box-shadow:0 2px 8px var(--accent-glow);transition:transform 0.15s,box-shadow 0.15s;" onmouseover="this.style.transform='translateY(-1px)';this.style.boxShadow='0 4px 14px var(--accent-glow)'" onmouseout="this.style.transform='';this.style.boxShadow='0 2px 8px var(--accent-glow)'">Copy Answer</button>
    </div>
    <script>
    function copyAnswer_{btn_id}() {{
        var text = {payload};
        if (navigator.clipboard && navigator.clipboard.writeText) {{
            navigator.clipboard.writeText(text).then(function() {{
                var btn = document.getElementById('{btn_id}');
                if (btn) {{
                    var orig = btn.textContent;
                    btn.textContent = 'Copied!';
                    btn.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
                    setTimeout(function() {{ btn.textContent = orig; btn.style.background = 'linear-gradient(135deg, var(--accent), #7c3aed)'; }}, 1500);
                }}
            }}).catch(function() {{ fallbackCopy_{btn_id}(text); }});
        }} else {{
            fallbackCopy_{btn_id}(text);
        }}
    }}
    function fallbackCopy_{btn_id}(text) {{
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try {{ document.execCommand('copy'); }} catch (e) {{}}
        document.body.removeChild(ta);
    }}
    </script>
    """
    st.html(html, unsafe_allow_javascript=True)


def render_chat_persistence_bridge():
    """If the browser has saved chats, hand them to the app once via a URL redirect."""
    if st.session_state.get("persistence_bridge_rendered"):
        return
    st.session_state.persistence_bridge_rendered = True
    
    payload = json.dumps(st.session_state.all_chats).replace("</", "<\\/")
    html = f"""
    <script>
    (function () {{
        try {{
            var raw = localStorage.getItem({json.dumps(CHATS_STORAGE_KEY)});
            if (!raw || raw.length > 80000) return;
            var url = new URL(window.location.href);
            if (url.searchParams.has({json.dumps(CHATS_BRIDGE_PARAM)})) return;
            if (raw === {payload}) return;
            url.searchParams.set({json.dumps(CHATS_BRIDGE_PARAM)}, raw);
            window.location.replace(url.toString());
        }} catch (e) {{}}
    }})();
    </script>
    """
    st.html(html, unsafe_allow_javascript=True)


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
    st.html(html, unsafe_allow_javascript=True)


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

if "chat_upload_sigs" not in st.session_state:
    st.session_state.chat_upload_sigs = {}

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
                    st.session_state.draft_store = None
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
            type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"doc_uploader_{st.session_state.current_chat}",
        )
        st.caption("PDF · Word · Text · CSV · Images")

    store = get_document_store()
    if store and store.source_count() > 0:
        total_chunks = sum(s["chunk_count"] for s in store.list_sources())
        st.markdown(
            f'<div class="upload-note">✓ {store.source_count()} source(s) · {total_chunks} chunks in this chat</div>',
            unsafe_allow_html=True,
        )
        if st.button("Summarize Documents", type="primary", use_container_width=True):
            st.session_state.pending_query = "Please provide a comprehensive summary of the uploaded documents."
            st.rerun()

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    if st.session_state.current_chat and st.session_state.all_chats.get(st.session_state.current_chat, {}).get("messages"):
        chat_data = st.session_state.all_chats[st.session_state.current_chat]
        export_text = f"# {chat_data['title']}\n\n"
        for msg in chat_data['messages']:
            role = "User" if msg["role"] == "user" else "Assistant"
            export_text += f"**{role}**:\n{msg['content']}\n\n"
        
        st.download_button(
            label="Download Chat",
            data=export_text,
            file_name=f"{chat_data['title'].replace(' ', '_').lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )


if uploaded_files:
    sig_key = st.session_state.current_chat or "draft"
    current_signature = tuple((file.name, file.size) for file in uploaded_files)
    if st.session_state.chat_upload_sigs.get(sig_key) != current_signature:
        with st.spinner("Processing documents..."):
            processed = process_files(uploaded_files)

            if processed:
                store = get_document_store()
                if store is None:
                    store = SessionVectorStore(get_embeddings(mode="transformer"))
                    if st.session_state.current_chat:
                        st.session_state.chat_stores[st.session_state.current_chat] = store
                    else:
                        st.session_state.draft_store = store

                total_chunks = 0
                for source, docs in processed:
                    chunks = chunk_data(docs)
                    result = store.add_documents(chunks, source_name=source)
                    total_chunks += result.get("chunk_count", 0)

            st.session_state.chat_upload_sigs[sig_key] = current_signature
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
        if message["role"] == "assistant":
            render_copy_button(message["content"])

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
            collected = []
            def display_generator():
                for token in answer_generator():
                    collected.append(token)
                    yield token
            st.write_stream(display_generator())
            answer = "".join(collected) if collected else "Sorry, no answer was generated."
        except Exception as exc:
            st.error(f"Could not generate an answer. Check GROQ_API_KEY. ({exc})")
            answer = "Sorry, I could not generate an answer."
        local_sources = {}
        web_used = False
        for doc in source_docs:
            meta = doc.metadata or {}
            src = meta.get("source")
            page = meta.get("page")
            
            if src == "web_search":
                web_used = True
            elif src:
                if src not in local_sources:
                    local_sources[src] = set()
                if page is not None:
                    local_sources[src].add(str(page))

        if local_sources:
            chip_htmls = []
            for src, pages in local_sources.items():
                if pages:
                    sorted_pages = sorted(list(pages), key=lambda x: int(x) if x.isdigit() else x)
                    page_str = f" (Page{'s' if len(pages) > 1 else ''} {', '.join(sorted_pages)})"
                else:
                    page_str = ""
                chip_htmls.append(f'<span class="cite-chip">📄 {src}{page_str}</span>')
            
            chips = "".join(chip_htmls)
            st.markdown(f'<div class="cite-row">{chips}</div>', unsafe_allow_html=True)
        elif web_used:
            st.caption("Answer included web search results.")

        render_copy_button(answer)

    messages.append({"role": "assistant", "content": answer})
    if st.session_state.current_chat is None:
        create_chat(title=generate_title(query), messages=messages)
        if st.session_state.draft_store is not None:
            st.session_state.chat_stores[st.session_state.current_chat] = st.session_state.draft_store
            st.session_state.draft_store = None
        st.session_state.draft_messages = []
        st.rerun()
    else:
        if chat["title"] == "New Chat":
            chat["title"] = generate_title(query)
        st.session_state.all_chats[st.session_state.current_chat] = chat
        save_chats()

render_chat_persistence_bridge()
persist_chats_localstorage()
