import streamlit as st
import tempfile
import os
from ingestpdf import create_embedding_index
from aiapi import ask

st.set_page_config(page_title="Financial Report RAG", page_icon="📈")

st.title("📈 Financial Report Assistant RAG")
st.caption("Upload any company's financial report and ask questions in plain English.")

col1, col2, spacer, col3 = st.columns([0.8, 0.7, 2.5, 1], gap="small")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None

# Init confidence threshold to default
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.7

st.caption(f"Loaded PDF: {st.session_state.ingested_file}")

@st.dialog("Upload Files")
def upload_dialog():
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if st.button("Done"):
        if uploaded:
            if st.session_state.ingested_file != uploaded.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                with st.spinner("Ingesting PDF..."):
                    create_embedding_index(tmp_path)
                st.session_state.ingested_file = uploaded.name
            st.rerun()


    if st.session_state.ingested_file:
        st.divider()
        st.success(f"{st.session_state.ingested_file} has been uploaded and is ready!")

    

with col1:
    with st.popover("Options"):
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold",
            min_value = 0.000, 
            max_value = 1.000,
            value = 0.750,
            step = 0.001,
            format="%.4f",
            help = "Retrieved chunks minimum relevance score, if higher, chunking model must be confident that chunk contains the answer"
        )

with col2:
    if st.button("Clear files"):
        st.session_state.ingested_file = None
        st.rerun()

with col3:
    if st.button("📁 Upload Files"):
        upload_dialog()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the financial report..."):
    if not st.session_state.ingested_file:
        st.error("Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(prompt, st.session_state.confidence_threshold)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})