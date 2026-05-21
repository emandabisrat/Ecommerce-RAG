"""
Task 4 – Streamlit Multimodal Chatbot UI
=========================================
Run with:
    streamlit run src/task4_ui.py
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.task3_llm_integration import create_chatbot

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopMind AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #f0eee8;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 6rem 2rem; max-width: 900px; margin: auto; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111118;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #f0eee8 !important; }

/* ── Header ── */
.shopmind-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 0.25rem;
}
.shopmind-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #ffffff 0%, #c084fc 60%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.shopmind-badge {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c084fc;
    border: 1px solid #c084fc44;
    padding: 2px 8px;
    border-radius: 4px;
}
.shopmind-sub {
    font-size: 0.9rem;
    color: #6b6b80;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* ── Chat bubbles ── */
.chat-row {
    display: flex;
    margin-bottom: 1.25rem;
    gap: 12px;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-row.user  { flex-direction: row-reverse; }
.chat-row.assistant { flex-direction: row; }

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    font-weight: 700;
}
.avatar.user-av  { background: linear-gradient(135deg, #c084fc, #f472b6); }
.avatar.bot-av   { background: linear-gradient(135deg, #1e1e2e, #2a2a3e); border: 1px solid #c084fc44; }

.bubble {
    max-width: 72%;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.65;
}
.bubble.user-bubble {
    background: linear-gradient(135deg, #7c3aed22, #db277722);
    border: 1px solid #c084fc33;
    border-top-right-radius: 4px;
    text-align: right;
}
.bubble.bot-bubble {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-top-left-radius: 4px;
    color: #d4d2cc;
}

/* ── Uploaded image preview in chat ── */
.chat-img {
    max-width: 200px;
    border-radius: 10px;
    margin-bottom: 6px;
    border: 1px solid #2a2a3e;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: #111118 !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 12px !important;
    color: #f0eee8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #c084fc !important;
    box-shadow: 0 0 0 2px #c084fc22 !important;
}

.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
/* Primary send button */
div[data-testid="column"]:first-child .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.6rem !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111118 !important;
    border: 1px dashed #2a2a3e !important;
    border-radius: 12px !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #111118 !important;
    border-color: #2a2a3e !important;
    color: #f0eee8 !important;
    border-radius: 10px !important;
}

/* Scrollable chat area */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding-right: 4px;
    margin-bottom: 1.5rem;
    scrollbar-width: thin;
    scrollbar-color: #2a2a3e transparent;
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 4px; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #3a3a4e;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state p { font-size: 0.9rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "chatbot" not in st.session_state:
    with st.spinner("Loading RAG pipeline & model…"):
        st.session_state.chatbot = create_chatbot(prompt_strategy="few_shot")

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, text, image}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("<hr style='border-color:#1e1e2e'>", unsafe_allow_html=True)

    prompt_strategy = st.selectbox(
        "Prompt Strategy",
        ["few_shot", "zero_shot", "multi_shot"],
        help="Controls how the LLM is prompted with retrieved context."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Session")
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.chatbot.reset()
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ℹ️ About")
    st.markdown("""
    <small style='color:#6b6b80; line-height:1.8'>
    <b style='color:#9a9ab0'>ShopMind AI</b> uses CLIP embeddings
    + ChromaDB RAG to retrieve relevant products, then passes context
    to a local <b style='color:#9a9ab0'>LLaVA</b> vision-language model
    for grounded, accurate answers.<br><br>
    Supports text queries, image uploads, and hybrid questions.
    </small>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="shopmind-header">
    <h1 class="shopmind-title">ShopMind AI</h1>
    <span class="shopmind-badge">Beta</span>
</div>
<p class="shopmind-sub">Multimodal product assistant · Powered by CLIP + LLaVA + RAG</p>
<hr class="divider">
""", unsafe_allow_html=True)


# ── Chat history ───────────────────────────────────────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🛍️</div>
        <p>Ask me anything about products.<br>
        You can type a question, upload a product image,<br>or do both at once.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        av_class = "user-av" if role == "user" else "bot-av"
        av_icon  = "U" if role == "user" else "🛍"
        bub_class = "user-bubble" if role == "user" else "bot-bubble"
        row_class = role

        img_html = ""
        if msg.get("image") is not None:
            # show thumbnail inline
            import io, base64
            buf = io.BytesIO()
            msg["image"].save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            img_html = f'<img src="data:image/jpeg;base64,{b64}" class="chat-img"><br>'

        st.markdown(f"""
        <div class="chat-row {row_class}">
            <div class="avatar {av_class}">{av_icon}</div>
            <div class="bubble {bub_class}">{img_html}{msg["text"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Input area ─────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📎 Attach a product image (optional)",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

user_image = None
if uploaded_file:
    user_image = Image.open(uploaded_file).convert("RGB")
    st.image(user_image, caption="Image ready to send", width=180)

user_text = st.text_area(
    "Your message",
    placeholder="e.g. What are the features of this product? or Compare Echo Dot vs Google Nest Mini…",
    height=90,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    send = st.button("Send →", use_container_width=True)
with col2:
    st.markdown("")   # spacer


# ── Handle send ────────────────────────────────────────────────────────────────
if send:
    if not user_text and user_image is None:
        st.warning("Please type a message or upload an image.")
    else:
        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "text": user_text or "*(image only)*",
            "image": user_image,
        })

        # Call chatbot
        with st.spinner("Thinking…"):
            try:
                response, image_path = st.session_state.chatbot.chat(
                    user_text=user_text or None,
                    user_image=user_image,
                    prompt_strategy=prompt_strategy,
                )
            except Exception as e:
                response = f"⚠️ Error: {e}"

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "text": response,
            "image": Image.open(image_path) if image_path else None,
        })

        st.rerun()