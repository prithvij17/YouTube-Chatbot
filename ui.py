import os
import re
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TubeChat — AI Video Assistant",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root & Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
}

/* ── Header ── */
.tube-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 0 24px;
    border-bottom: 1px solid #1e1e30;
    margin-bottom: 28px;
}
.tube-logo {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #ff4444 0%, #ff0066 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.tube-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
    line-height: 1;
    margin: 0;
}
.tube-subtitle {
    font-size: 11px;
    color: #6b6b8a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 0;
    margin-top: 3px;
}

/* ── Input styling ── */
.stTextInput > div > div > input {
    background: #13131f !important;
    border: 1px solid #252538 !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #ff4444 !important;
    box-shadow: 0 0 0 2px rgba(255,68,68,0.15) !important;
}
.stTextInput > label {
    color: #9090b0 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ff4444, #ff0066) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Status / info boxes ── */
.status-box {
    background: #13131f;
    border: 1px solid #252538;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #9090b0;
    margin-bottom: 8px;
}
.status-box.success {
    border-color: #1a3a1a;
    background: #0d1f0d;
    color: #4caf50;
}
.status-box.error {
    border-color: #3a1a1a;
    background: #1f0d0d;
    color: #f44336;
}

/* ── Video info card ── */
.video-card {
    background: linear-gradient(135deg, #13131f 0%, #0f0f1e 100%);
    border: 1px solid #252538;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
}
.video-id-badge {
    display: inline-block;
    background: rgba(255,68,68,0.12);
    border: 1px solid rgba(255,68,68,0.25);
    color: #ff6666;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.video-stat {
    font-size: 12px;
    color: #6b6b8a;
    margin-top: 4px;
}
.video-stat span { color: #c0b0ff; font-weight: 500; }

/* ── Chat area ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #13131f !important;
    border: 1px solid #252538 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #ff4444, #ff0066) !important;
    border-radius: 8px !important;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a5a7a;
    margin-bottom: 10px;
}

/* ── Divider ── */
hr { border-color: #1e1e30 !important; }

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #ff4444 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #252538; border-radius: 4px; }

/* ── Main content padding ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 860px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #3a3a5a;
}
.empty-state-icon { font-size: 48px; margin-bottom: 16px; }
.empty-state-title {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #4a4a6a;
    margin-bottom: 8px;
}
.empty-state-text { font-size: 14px; line-height: 1.6; }

/* ── Suggested chips ── */
.chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
    justify-content: center;
}
.chip {
    background: #13131f;
    border: 1px solid #252538;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #7070a0;
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover { border-color: #ff4444; color: #ff6666; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    """Accept a full YouTube URL or bare video ID."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?\/]|$)",
        r"^([0-9A-Za-z_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id.strip())
        if m:
            return m.group(1)
    return url_or_id.strip()


def format_context(docs):
    return "\n\n".join(d.page_content for d in docs)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "chain": None,
    "video_id": None,
    "chunk_count": 0,
    "transcript_len": 0,
    "messages": [],
    "loading": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="tube-header">
        <div class="tube-logo">▶</div>
        <div>
            <p class="tube-title">TubeChat</p>
            <p class="tube-subtitle">AI Video Assistant</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Load a Video</p>', unsafe_allow_html=True)

    video_input = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://youtube.com/watch?v=... or Gfr50f6ZBvo",
        label_visibility="collapsed",
    )

    load_btn = st.button("⚡  Load & Index Video", use_container_width=True)

    if load_btn and video_input:
        vid = extract_video_id(video_input)
        with st.spinner("Fetching transcript…"):
            try:
                api = YouTubeTranscriptApi()
                transcript_list = api.fetch(vid, languages=["en"])
                transcript = " ".join(chunk.text for chunk in transcript_list)
            except Exception as e:
                st.markdown(f'<div class="status-box error">❌ Could not fetch transcript: {e}</div>',
                            unsafe_allow_html=True)
                st.stop()

        with st.spinner("Building vector index…"):
            splitter = RecursiveCharacterTextSplitter(chunk_size=880, chunk_overlap=150)
            chunks = splitter.create_documents([transcript])

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            store = FAISS.from_documents(chunks, embeddings)
            retriever = store.as_retriever(search_kwargs={"k": 2})

            template = PromptTemplate(
                template="""You are a helpful AI assistant that answers questions based on the provided video transcript context.
If the answer is not found in the context, say "That's outside the scope of this video."

Context:
{context}

Question: {query}

Answer:""",
                input_variables=["context", "query"],
            )

            repo_id = "HuggingFaceH4/zephyr-7b-beta"
            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="conversational",
                max_new_tokens=512,
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            )
            chat_model = ChatHuggingFace(llm=llm)
            parser = StrOutputParser()

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_context),
                "query": RunnablePassthrough(),
            })
            chain = parallel_chain | template | chat_model | parser

        st.session_state.chain = chain
        st.session_state.video_id = vid
        st.session_state.chunk_count = len(chunks)
        st.session_state.transcript_len = len(transcript.split())
        st.session_state.messages = []
        st.rerun()

    # ── Video info ───────────────────────────────────────────────────────────
    if st.session_state.video_id:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Loaded Video</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="video-card">
            <div class="video-id-badge">{st.session_state.video_id}</div>
            <div class="video-stat">Transcript: <span>~{st.session_state.transcript_len:,} words</span></div>
            <div class="video-stat">Chunks indexed: <span>{st.session_state.chunk_count}</span></div>
            <div class="video-stat">Embeddings: <span>all-MiniLM-L6-v2</span></div>
            <div class="video-stat">LLM: <span>Zephyr-7B-β</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#3a3a5a; line-height:1.8;">
        Built with LangChain · FAISS · HuggingFace<br>
        Embeddings: all-MiniLM-L6-v2<br>
        LLM: Zephyr-7B-β
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.chain:
    # Empty / welcome state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">▶</div>
        <div class="empty-state-title">Chat with any YouTube video</div>
        <div class="empty-state-text">
            Paste a YouTube URL or video ID in the sidebar,<br>
            hit <strong>Load &amp; Index Video</strong>, then ask anything.
        </div>
        <div class="chips-row">
            <div class="chip">📝 Summarize the video</div>
            <div class="chip">🔑 Key takeaways</div>
            <div class="chip">❓ Ask a question</div>
            <div class="chip">📌 Important timestamps</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat header ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <span style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:#fff;">
            Ask about the video
        </span>
        <span style="margin-left:10px;font-size:12px;color:#5a5a7a;letter-spacing:1px;">
            · {st.session_state.video_id}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Render history ────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"],
                             avatar="🧑" if msg["role"] == "user" else "▶"):
            st.markdown(msg["content"])

    # ── Suggested prompts (shown only at start) ───────────────────────────────
    if not st.session_state.messages:
        col1, col2 = st.columns(2)
        suggestions = [
            ("📝", "Summarize this video in 5 bullet points"),
            ("🔑", "What are the key takeaways?"),
            ("🎯", "What is the main topic discussed?"),
            ("💡", "What insights does the speaker share?"),
        ]
        for i, (icon, label) in enumerate(suggestions):
            col = col1 if i % 2 == 0 else col2
            if col.button(f"{icon}  {label}", key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": label})
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(label)
                with st.chat_message("assistant", avatar="▶"):
                    with st.spinner("Thinking…"):
                        answer = st.session_state.chain.invoke(label)
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about the video…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="▶"):
            with st.spinner("Searching transcript & generating answer…"):
                answer = st.session_state.chain.invoke(prompt)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()