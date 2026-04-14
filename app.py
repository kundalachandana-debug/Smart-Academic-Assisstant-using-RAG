import os
import io
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from deep_translator import GoogleTranslator
from ingest import load_vectorstore, get_available_subjects
from rag_chain import build_rag_chain, ask
from utils import (
    format_source_citation,
    ingest_files_batch,
    validate_batch,
    ALLOWED_EXTENSIONS,
)

load_dotenv()

st.set_page_config(
    page_title="Smart Academic Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.source-card {
    background: #f8f9fa;
    border-left: 3px solid #1a73e8;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 6px 0;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Japanese": "ja",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-CN",
    "Korean": "ko",
    "Portuguese": "pt",
    "Russian": "ru",
    "Bengali": "bn",
    "Urdu": "ur"
}


@st.cache_resource(show_spinner=False)
def get_chain():
    vs = load_vectorstore()
    return build_rag_chain(vs)


def text_to_speech(text, lang_code="en"):
    try:
        tts = gTTS(text=text[:500], lang=lang_code, slow=False)
        audio_buf = io.BytesIO()
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)
        return audio_buf
    except Exception:
        return None


def translate_text(text, target_lang_code):
    if target_lang_code == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang_code).translate(text)
    except Exception:
        return text


with st.sidebar:
    st.title("🎓 Academic Assistant")
    st.caption("Powered by RAG + Groq LLaMA")
    st.divider()

    subjects = get_available_subjects()
    if subjects:
        st.subheader("📚 Loaded Subjects")
        for s in subjects:
            st.markdown(f"✅ {s}")
    else:
        st.info("No files loaded yet.")

    st.divider()
    st.subheader("🌍 Response Language")
    selected_language = st.selectbox("Choose language", list(LANGUAGES.keys()), index=0)
    lang_code = LANGUAGES[selected_language]

    st.divider()
    st.subheader("🔊 Voice Settings")
    enable_voice = st.toggle("Enable Voice Response", value=True)

    st.divider()
    st.subheader("➕ Add Study Materials")

    accepted_exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
    st.caption(f"Accepted: {accepted_exts} · Max 20 files · Max 50 MB each")

    uploaded_files = st.file_uploader(
        "Upload Study Material(s)",
        type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

        valid_files, errors = validate_batch(uploaded_files)
        if errors:
            for err in errors:
                st.warning(f"⚠️ {err}")
        if valid_files:
            st.success(f"✅ {len(valid_files)} file(s) ready to ingest")

        if st.button("📥 Ingest Files", use_container_width=True, disabled=len(valid_files) == 0):
            with st.spinner(f"Processing {len(valid_files)} file(s)..."):
                try:
                    result = ingest_files_batch(uploaded_files)

                    for err in result["errors"]:
                        st.warning(f"⚠️ {err}")

                    if result["saved"] > 0:
                        st.cache_resource.clear()
                        names = ", ".join(result["saved_names"])
                        st.success(
                            f"✅ {result['saved']} file(s) added: {names}"
                            + (f" ({result['converted']} converted to PDF)" if result["converted"] else "")
                        )
                        st.rerun()
                    else:
                        st.error("No files were saved. Check warnings above.")
                except Exception as e:
                    st.error(f"Batch error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.header("🎓 Ask Anything from Your Study Materials")

if not os.path.exists("vectorstore"):
    st.warning("No knowledge base found. Upload files in the sidebar or run: python ingest.py")
    st.stop()

try:
    chain = get_chain()
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 View Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>📗 {src["subject"]}</b> — Page {src["page"]}<br>'
                        f'<span style="color:#555">{src["snippet"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

query = st.chat_input("💬 Ask a question from any subject...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your materials..."):
            try:
                response = ask(chain, query)
                answer   = response["answer"]
                sources  = [format_source_citation(d) for d in response["sources"]]
            except Exception as e:
                answer  = f"Error: {e}"
                sources = []

        translated_answer = translate_text(answer, lang_code)
        st.markdown(translated_answer)

        if enable_voice:
            audio_out = text_to_speech(translated_answer, lang_code)
            if audio_out:
                st.audio(audio_out, format="audio/mp3")

        if sources:
            with st.expander("📎 View Sources", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>📗 {src["subject"]}</b> — Page {src["page"]}<br>'
                        f'<span style="color:#555">{src["snippet"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": translated_answer,
        "sources": sources,
    })
    
    